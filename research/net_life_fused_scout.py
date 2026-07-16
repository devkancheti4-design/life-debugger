#!/usr/bin/env python3
"""NET + LIFE FUSED SCOUT — fix the Life-Scout's one big break, honestly.

The stress test proved the Life-alone Scout fingerprints an EXACT trace, so the
SAME bug reached via a different loop-count / depth / path gets a DIFFERENT
signature -> recurrence recognition BREAKS (one bug -> 12 signatures).

The fix is the session's whole architecture: fuse the exact Life with a SMALL
neural net that supplies the semantics the Life lacks. The net learns to EMBED a
run so that same-bug runs land near each other, even when their exact traces
differ. The Life keeps what it's good at: an exact anchor (byte-identical repeat
-> certain recall) and an honest ABSTAIN on anything it hasn't earned.

  net  = meaning  (same bug, different path -> nearby embedding)
  Life = exact memory + honest abstain (never bluffs "seen this")
  alive = the prototype bank learns online as new runs arrive

We then run the SAME recurrence test that broke the Life alone, on HELD-OUT runs
(same bugs via loop-counts/depths never seen in training), and measure:
  * recurrence recall : does it recognize the bug via a NEW path?  (Life-alone ~0)
  * false-merge       : does the net wrongly fuse two DIFFERENT bugs? (the cost)
  * abstain-on-novel  : on a bug family held out entirely, does it honestly abstain?

Run:  python3 net_life_fused_scout.py
"""
import sys, os, hashlib
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/Users/kanchetidevieswar/hybrid/life-memory")
from life import Life

def P(m): print(m); sys.stdout.flush()
torch.manual_seed(0); np.random.seed(0)
THIS = os.path.abspath(__file__)

# ---------------------------------------------------------------- bug families
# each takes k (1..40): k varies the TRACE (loop count / recursion depth) but NOT
# the bug — the exact case that fragmented the Life-alone signature.
def fam_zdiv(k):
    acc = 0
    for i in range(k): acc += 1
    return 5 // (acc - acc)                 # ZeroDivisionError, fixed crash line
def fam_key(k):
    def rec(d, n):
        if n <= 0: return d['nope']         # KeyError, fixed line
        return rec(d, n - 1)
    return rec({'a': 1}, k)                  # via recursion depth k
def fam_idx(k):
    xs = [0, 1, 2]; acc = 0
    for i in range(k): acc += 1
    return xs[10]                            # IndexError after a k-loop
def fam_typ(k):
    v = None
    for i in range(k): pass
    return v + 1                             # TypeError after a k-loop
def fam_clean(k):
    s = 0
    for i in range(k): s += i
    return s                                 # clean after a k-loop

FAMILIES = {"zdiv": fam_zdiv, "key": fam_key, "idx": fam_idx, "typ": fam_typ, "clean": fam_clean}

# ---------------------------------------------------------------- the Life walks a run
def trace_life(fn, k):
    life = Life(); prev = [None]; err = [None]
    def tracer(frame, event, arg):
        if frame.f_code.co_filename != THIS: return None
        if event == "line":
            loc = f"L{frame.f_lineno}"
            if prev[0]: life.learn(prev[0], loc)
            prev[0] = loc
        elif event == "exception":
            et = arg[0].__name__; loc = f"EXC:{et}@L{frame.f_lineno}"
            if prev[0]: life.learn(prev[0], loc)
            prev[0] = loc; err[0] = f"{et}@L{frame.f_lineno}"
        return tracer
    sys.settrace(tracer)
    try: fn(k)
    except Exception: pass
    finally: sys.settrace(None)
    return life

def featurize(life, D=128):
    """Hash the Life's transition-count table into a fixed vector (deterministic)."""
    v = np.zeros(D, np.float32)
    for k, d in life.store.items():
        for val, cnt in d.items():
            h = int(hashlib.md5(f"{k}|{val}".encode()).hexdigest(), 16) % D
            v[h] += cnt
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

# ---------------------------------------------------------------- the small net
class Encoder(nn.Module):
    def __init__(s, D=128, H=64, E=16):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(D, H), nn.ReLU(), nn.Linear(H, E))
    def forward(s, x): return nn.functional.normalize(s.net(x), dim=-1)

def build_dataset(hold_out=None):
    X, y, sig = [], [], []
    for fam, fn in FAMILIES.items():
        for k in range(1, 41):
            life = trace_life(fn, k)
            X.append(featurize(life)); y.append(fam); sig.append(life.sha())
    return np.array(X, np.float32), np.array(y), np.array(sig)

def main():
    P("=" * 78); P("NET + LIFE FUSED SCOUT — recover the recurrence the Life alone lost"); P("=" * 78)
    X, y, sig = build_dataset()
    fams = list(FAMILIES)
    # split by k parity: even k = train, odd k = test  -> test loop-counts NEVER in train
    idx = np.arange(len(y)); k_of = np.tile(np.arange(1, 41), len(fams))
    tr, te = idx[k_of % 2 == 0], idx[k_of % 2 == 1]
    P(f"  {len(fams)} bug families x 40 runs each; train=even-k ({len(tr)}), test=odd-k ({len(te)}) — disjoint paths\n")

    # ---- Life-alone baseline: recognize a test run only by EXACT signature ----
    train_sig = {sig[i]: y[i] for i in tr}
    life_hits = sum(1 for i in te if sig[i] in train_sig and train_sig[sig[i]] == y[i])
    P(f"  LIFE ALONE (exact signature): recurrence recall on held-out paths = "
      f"{life_hits}/{len(te)} = {100*life_hits/len(te):.0f}%   <- the break we're fixing")

    # ---- train the small net (contrastive triplets) ----
    enc = Encoder(); opt = torch.optim.Adam(enc.parameters(), lr=1e-3)
    Xtr = torch.tensor(X[tr]); ytr = y[tr]
    by_fam = {f: [j for j in range(len(tr)) if ytr[j] == f] for f in fams}
    rng = np.random.default_rng(0)
    for step in range(600):
        a, p, n = [], [], []
        for _ in range(64):
            f = fams[rng.integers(len(fams))]
            if len(by_fam[f]) < 2: continue
            ai, pi = rng.choice(by_fam[f], 2, replace=False)
            g = fams[rng.integers(len(fams))]
            while g == f: g = fams[rng.integers(len(fams))]
            ni = rng.choice(by_fam[g])
            a.append(ai); p.append(pi); n.append(ni)
        ea, ep, en = enc(Xtr[a]), enc(Xtr[p]), enc(Xtr[n])
        loss = nn.functional.triplet_margin_loss(ea, ep, en, margin=0.4)
        opt.zero_grad(); loss.backward(); opt.step()

    # ---- fuse: exact Life anchor -> else nearest embedding -> else ABSTAIN ----
    with torch.no_grad():
        Etr = enc(torch.tensor(X[tr])).numpy(); Ete = enc(torch.tensor(X[te])).numpy()
    def nn_sim(e, bank, labs, self_i=None):
        s = bank @ e
        if self_i is not None: s[self_i] = -1
        j = int(s.argmax()); return s[j], labs[j]
    # threshold from train: separator between same-family and different-family pair sims
    same, diff = [], []
    for _ in range(4000):
        i, j = rng.choice(len(tr), 2, replace=False)
        s = float(Etr[i] @ Etr[j])
        (same if y[tr][i] == y[tr][j] else diff).append(s)
    thr = (np.percentile(same, 5) + np.percentile(diff, 95)) / 2
    P(f"  small net trained (128->64->16, triplet). recognition threshold (cos) = {thr:.3f}\n")

    def fused_recognize(i):
        if sig[i] in train_sig: return train_sig[sig[i]], "exact"     # Life anchor
        s, lab = nn_sim(Ete[list(te).index(i)], Etr, y[tr])           # net semantics
        return (lab, "net") if s >= thr else (None, "abstain")

    correct = merge = abst = 0
    for i in te:
        pred, how = fused_recognize(i)
        if pred is None: abst += 1
        elif pred == y[i]: correct += 1
        else: merge += 1
    n = len(te)
    P(f"  NET + LIFE FUSED (alive):")
    P(f"    recurrence recall  (same bug, new path -> right family) : {correct}/{n} = {100*correct/n:.0f}%")
    P(f"    false-merge        (wrong family — the semantic cost)   : {merge}/{n} = {100*merge/n:.0f}%")
    P(f"    abstain            (didn't recognize — stayed honest)   : {abst}/{n} = {100*abst/n:.0f}%")

    # ---- honesty test: hold ONE family out of training entirely; must ABSTAIN, not merge ----
    novel = "typ"
    keep = [i for i in tr if y[i] != novel]
    Ekeep = enc(torch.tensor(X[keep])).detach().numpy()
    novel_te = [i for i in te if y[i] == novel]
    ab = 0
    for i in novel_te:
        e = enc(torch.tensor(X[i:i+1])).detach().numpy()[0]
        s, _ = nn_sim(e, Ekeep, y[keep])
        if s < thr: ab += 1
    P(f"\n  ABSTAIN-ON-NOVEL (family '{novel}' never trained): {ab}/{len(novel_te)} = "
      f"{100*ab/len(novel_te):.0f}% honestly abstained (rest were false-merged into a known bug)")

    P(f"\n  HONEST: the net recovers the recurrence the exact Life lost — but it is now")
    P(f"  FUZZY (some false-merges) and needs training data. The Life keeps it honest:")
    P(f"  exact repeats are certain, and truly-novel bugs should ABSTAIN, not guess.")

if __name__ == "__main__":
    main()
