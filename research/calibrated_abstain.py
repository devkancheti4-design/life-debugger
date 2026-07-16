#!/usr/bin/env python3
"""CALIBRATED ABSTAIN — restore the Life's never-bluff guarantee on top of the net.

Problem (measured in error_space_coverage.py): a single global cosine threshold lets
the net force ~45% of genuinely-novel errors into a known bucket. We want the coverage
AND the honesty. The fix layers the Life's exact strength under the net's fuzzy one:

  Gate A  Life exact error-TYPE membership : never saw this error_type? ABSTAIN. (hard)
  Gate B  per-class radius                 : too far from the nearest pattern's own
                                             training spread? ABSTAIN. (calibrated per class)
  Gate C  margin                           : nearest and 2nd-nearest pattern too close to
                                             call? ABSTAIN. (ambiguous -> don't guess)

We compare 5 gates on ONE fixed encoder (controls for encoder noise) across three tests:
  covered        : held-out instances of TRAINED patterns  -> want high recall
  novel-pattern  : patterns held out, but their error_type IS trained -> want abstain
  novel-category : 3 error_types held out entirely          -> want abstain (~100%)

Run: python3 calibrated_abstain.py
"""
import sys, os, re, glob, json, hashlib
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/Users/kanchetidevieswar/hybrid/life-memory")
from life import Life

def P(m): print(m); sys.stdout.flush()
torch.manual_seed(0); np.random.seed(0)
SD = "/private/tmp/claude-501/-Users-kanchetidevieswar-hybrid/7c960cd0-0a5e-4f03-8c1a-78c428374a75/scratchpad"

def normalize(s):
    s = re.sub(r'"[^"]*"|\'[^\']*\'', "STR", s); return re.sub(r"\b\d+\b", "NUM", s)
def toks(rec):
    out = ["ET:" + rec["error_type"]]
    for fn, ln, src in rec["traceback"]:
        out.append("FN:" + fn); out += ["TB:" + t for t in re.findall(r"[A-Za-z_]+|[^\s]", normalize(src))]
    for line in rec["code"].splitlines():
        out += ["CD:" + t for t in re.findall(r"[A-Za-z_]+|[^\s]", normalize(line))]
    return out
def featurize(rec, D=256):
    v = np.zeros(D, np.float32)
    for t in toks(rec): v[int(hashlib.md5(t.encode()).hexdigest(), 16) % D] += 1.0
    n = np.linalg.norm(v); return v / n if n else v

class Enc(nn.Module):
    def __init__(s, D=256, H=128, E=32):
        super().__init__(); s.n = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, E))
    def forward(s, x): return nn.functional.normalize(s.n(x), dim=-1)
def train_net(X, y, steps=600):
    enc = Enc(); opt = torch.optim.Adam(enc.parameters(), 1e-3); Xt = torch.tensor(X)
    cls = sorted(set(y)); by = {c: [j for j in range(len(y)) if y[j] == c] for c in cls}
    by = {c: v for c, v in by.items() if len(v) >= 2}; keys = list(by); rng = np.random.default_rng(0)
    for _ in range(steps):
        a, p, n = [], [], []
        for _ in range(96):
            c = keys[rng.integers(len(keys))]; ai, pi = rng.choice(by[c], 2, replace=False)
            g = keys[rng.integers(len(keys))]
            while g == c: g = keys[rng.integers(len(keys))]
            a.append(ai); p.append(pi); n.append(rng.choice(by[g]))
        ea, ep, en = enc(Xt[a]), enc(Xt[p]), enc(Xt[n])
        loss = nn.functional.triplet_margin_loss(ea, ep, en, margin=0.4)
        opt.zero_grad(); loss.backward(); opt.step()
    return enc

def main():
    recs = []
    for f in sorted(glob.glob(f"{SD}/errcorpus_*.jsonl")):
        for line in open(f):
            try: recs.append(json.loads(line))
            except: pass
    for r in recs: r["_lab"] = r["category"] + "|" + r["pattern"]; r["_f"] = featurize(r)
    P("=" * 82); P("CALIBRATED ABSTAIN — keep the coverage, restore the never-bluff guarantee"); P("=" * 82)

    # ---- splits: novel category (3 error_types) / novel pattern (30% of remaining patterns) / covered
    novel_cat = {"OverflowError", "StopIteration", "AssertionError"}
    pats = sorted({r["_lab"] for r in recs})
    known_pats = [p for p in pats if p.split("|")[0] not in novel_cat]
    rng = np.random.default_rng(3); rng.shuffle(known_pats)
    n_novelpat = int(len(known_pats) * 0.30)
    novel_pat = set(known_pats[:n_novelpat]); train_pats = set(known_pats[n_novelpat:])

    tr, cov_te, val = [], [], []
    by = {}
    for i, r in enumerate(recs): by.setdefault(r["_lab"], []).append(i)
    for lab, idxs in by.items():
        if lab.split("|")[0] in novel_cat or lab in novel_pat: continue     # OOD sets handled below
        idxs = list(idxs); rng.shuffle(idxs)
        c1, c2 = int(len(idxs)*0.55), int(len(idxs)*0.75)
        tr += idxs[:c1]; val += idxs[c1:c2]; cov_te += idxs[c2:]            # train / calib / test
    npat_te = [i for i, r in enumerate(recs) if r["_lab"] in novel_pat]
    ncat_te = [i for i, r in enumerate(recs) if r["category"] in novel_cat]

    X = np.array([r["_f"] for r in recs], np.float32); y = np.array([r["_lab"] for r in recs])
    et = [r["error_type"] for r in recs]
    P(f"  train {len(tr)} | calib {len(val)} | covered-test {len(cov_te)} | "
      f"novel-pattern {len(npat_te)} | novel-category {len(ncat_te)}")
    P(f"  trained patterns {len(train_pats)} | held-out patterns {len(novel_pat)} | held-out types {sorted(novel_cat)}\n")

    enc = train_net(X[tr], y[tr])
    with torch.no_grad():
        E = {i: enc(torch.tensor(X[i:i+1])).numpy()[0] for i in range(len(recs))}
    # prototypes + per-class calibrated radius, known error types (all from TRAIN only)
    tr_by = {}
    for i in tr: tr_by.setdefault(y[i], []).append(i)
    proto = {c: (lambda v: v/np.linalg.norm(v))(np.mean([E[i] for i in v], 0)) for c, v in tr_by.items()}
    known_types = set(et[i] for i in tr)                          # Life exact membership
    type_of = {c: c.split("|")[0] for c in proto}                  # pattern -> its category/type
    # calibrate on val: per-class radius (5th pct in-class sim) + global thr + margin floor
    radius = {}
    for c, v in tr_by.items():
        sims = [float(E[i] @ proto[c]) for i in v]
        radius[c] = np.percentile(sims, 5) if len(sims) >= 4 else min(sims) - 0.05
    val_same = [float(E[i] @ proto[y[i]]) for i in val if y[i] in proto]
    global_thr = np.percentile(val_same, 5) if val_same else 0.5
    val_margin = []
    for i in val:
        sims = sorted((float(E[i] @ proto[c]) for c in proto), reverse=True)
        if len(sims) >= 2: val_margin.append(sims[0] - sims[1])
    margin_min = np.percentile(val_margin, 5) if val_margin else 0.0

    protos = list(proto); Pmat = np.array([proto[c] for c in protos])
    def nearest(e, same_type=None):
        cand = [k for k, c in enumerate(protos) if same_type is None or type_of[c] == same_type]
        if not cand: return None, -1, -1
        sims = Pmat[cand] @ e; order = np.argsort(-sims)
        c1 = protos[cand[order[0]]]; s1 = float(sims[order[0]])
        s2 = float(sims[order[1]]) if len(order) > 1 else -1.0
        return c1, s1, s2

    # ---- the 5 gates ----
    def g_global(e, etype):                                       # baseline: one cosine threshold
        c, s1, _ = nearest(e); return c if s1 >= global_thr else None
    def g_type(e, etype):                                         # + Life exact error-type gate
        if etype not in known_types: return None
        c, s1, _ = nearest(e, same_type=etype); return c if c and s1 >= global_thr else None
    def g_type_radius(e, etype):                                  # + per-class radius
        if etype not in known_types: return None
        c, s1, _ = nearest(e, same_type=etype)
        return c if c and s1 >= radius[c] else None
    def g_type_margin(e, etype):                                  # + margin
        if etype not in known_types: return None
        c, s1, s2 = nearest(e, same_type=etype)
        return c if c and s1 >= global_thr and (s1 - s2) >= margin_min else None
    def g_full(e, etype):                                         # Life type + radius + margin
        if etype not in known_types: return None
        c, s1, s2 = nearest(e, same_type=etype)
        if not c: return None
        return c if (s1 >= radius[c] and (s1 - s2) >= margin_min) else None

    gates = [("global thr (baseline)", g_global), ("+ Life type gate", g_type),
             ("+ type + radius", g_type_radius), ("+ type + margin", g_type_margin),
             ("+ type+radius+margin (full)", g_full)]

    def evalg(gate):
        cov = sum(1 for i in cov_te if gate(E[i], et[i]) == y[i]) / max(len(cov_te), 1)
        npa = sum(1 for i in npat_te if gate(E[i], et[i]) is None) / max(len(npat_te), 1)   # want abstain
        nca = sum(1 for i in ncat_te if gate(E[i], et[i]) is None) / max(len(ncat_te), 1)   # want abstain
        return cov, npa, nca

    P(f"  {'gate':<32}{'covered recall':>16}{'novel-pat abstain':>19}{'novel-cat abstain':>19}")
    P("  " + "-" * 84)
    for name, g in gates:
        cov, npa, nca = evalg(g)
        P(f"  {name:<32}{f'{100*cov:.0f}%':>16}{f'{100*npa:.0f}%':>19}{f'{100*nca:.0f}%':>19}")

    P(f"\n  calibrated on validation: global_thr={global_thr:.3f}  margin_min={margin_min:.3f}")
    P(f"  HONEST READ: the Life error-type gate is EXACT — it can only lift novel-category abstain to")
    P(f"  100% because a never-seen error_type is knowable with certainty (the Life's real strength).")
    P(f"  Novel-PATTERN within a known type is the hard case (fuzzy) — radius+margin push it up, at")
    P(f"  a measured cost to covered recall. That trade IS the recall<->honesty knob, now explicit.")

if __name__ == "__main__":
    main()
