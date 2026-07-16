# The Life Debugger

**A four-tier, mostly-local, never-bluff Python debugger.** Catch the certain surface bugs for **$0**, let a
small local model reason the common hard ones, escalate only the genuinely-novel tail to a frontier model —
and an **alive, persistent memory (the "Life") learns every fix a model produces and replays it free forever
after.** The more you run it, the cheaper it gets.

> Honest by design. Every number below comes from running code (`proof_alive.py`, and the research in
> [`ARCHITECTURE.md`](ARCHITECTURE.md)). It does **not** claim to beat a frontier model — it claims to make
> most bugs free, deterministic, and never-invented, and to only pay a reasoner for what actually needs one.

---

## How it works — four tiers

```
              your buggy code
                    │
   ┌────────────────▼─────────────────┐
   │ TIER 1 · FACTS                    │  static anti-patterns + execute-and-catch
   │ detect + explain the certain      │  $0 · deterministic · never bluffs
   │ surface bugs                      │
   └────────────────┬─────────────────┘
                    │  (something it can't be certain about)
   ┌────────────────▼─────────────────┐
   │ TIER 2 · LIFE  (alive cache)      │  has a model already solved this? replay the fix FREE
   └────────────────┬─────────────────┘
                    │  (never seen before)
   ┌────────────────▼─────────────────┐
   │ TIER 3 · NET   (small local model)│  reasons the common intent-dependent fix   (~$0, local)
   └────────────────┬─────────────────┘
                    │  (genuinely novel)
   ┌────────────────▼─────────────────┐
   │ TIER 4 · FRONTIER  (optional)     │  the hard tail
   └───────────────────────────────────┘
        every model solve is LEARNED by the Life → never solved twice
```

## Measured (from running code)

| Property | Result |
|---|---|
| Beginner bugs resolved by the free fact tier | **70%** (never-bluff) |
| Advanced/concurrency bugs by the fact tier | ~34–54% (surface); deep ones escalate |
| Small local net (qwen-7B) on real concurrency code | verified-fixed **8/13**, $0 API |
| **Facts + net together** on that file | **12/13** (each covers the other's blind spot) |
| Reasoning-model calls cut by the alive cache (recurring stream) | **~63%** (70% → 12% over time) |
| Semantic bugs a fact base can catch | **7%** (the rest need reasoning — not a size problem) |

The honest boundaries (where it thins out, what it can't do) are written into
[`ARCHITECTURE.md`](ARCHITECTURE.md) — read it before you trust the pitch.

---

## Install

```bash
git clone git@github.com:<you>/life-debugger.git
cd life-debugger
# pure Python 3.9+, no dependencies for tiers 1–2. (Tier 3 optionally uses ollama.)
python3 proof_alive.py          # see the Life work + learn, live
```

## Use it

```python
from life_debugger import debug, AliveLife

# free fact tier — detect + explain, no model
report = debug(open("buggy.py").read())
for kind, line, why in report["facts"]:
    print(f"line {line}: [{kind}] {why}")

# attach a reasoner for the intent-dependent bugs; the Life learns each fix once
def my_model(code, fact_kinds):        # plug in ollama/qwen, a frontier API, anything
    return call_your_model(code)

report = debug(code, reasoner=my_model)
print(report["handled_by"])            # "reasoner (escalated)" the first time...
report = debug(code, reasoner=my_model)
print(report["handled_by"])            # ..."life-cache (free, $0)" every time after
```

## How the Life works and learns

The Life is a **persistent, exact, honest memory** (`life.py`). It is *not* just a dict:

- **It stores facts** — `learn(key, value)`, O(1) recall, scales to millions.
- **It never overwrites** — revising a key keeps the full history; newest wins recall.
- **It abstains** — recall on an unknown key returns `None`, a real "I don't know", never a guess.
- **It is byte-reproducible** — same facts → same `sha()`, on any machine.
- **It is ALWAYS ALIVE** — it loads from disk on startup and saves after every learn, so it accumulates
  across runs and sessions. It never resets to static.

When a reasoner solves a bug, the Life **learns** `(bug-signature → fix)` and persists it. Next time that
bug appears, tier 2 **replays the fix for free** — the model is never called again for it.

**Honest limits** (also in the code comments): the Life is *exact-key* — a typo misses, like a dict; it
doesn't do fuzzy/semantic matching by itself (pair it with the net for that). The free replay is *full* for
canonical fixes and *recognition-only* for context-specific ones (the net still places the lock). And
genuinely-novel bugs always escalate — the reasoner is a permanent tier, not a crutch removed once you have
"enough" facts. **Semantics can't be stored, only reasoned — the Life makes reasoning cheaper, not optional.**

## Proof it works and learns (run it)

```
$ python3 proof_alive.py
...
[3] LEARN + REPLAY FREE — a reasoner is called ONCE, then the Life replays forever
    1st time  -> handled_by: reasoner (escalated)     reasoner calls so far: 1
    2nd time  -> handled_by: life-cache (free, $0)    reasoner calls so far: 1
    3rd time  -> handled_by: life-cache (free, $0)    reasoner calls so far: 1
[4] ALWAYS ALIVE — a brand-new process reloads the same facts from disk (never resets)
    before restart: 4 facts, sha 33389988e539d84c
    after  restart: 4 facts, sha 33389988e539d84c
    >> byte-identical after restart: the Life is alive, not static.
```

## Layout

```
life_debugger.py    the four-tier debug() + the persistent AliveLife
life.py             the Life: exact, permanent, honest memory (one file, no deps)
proof_alive.py      runnable proof it stores/learns/replays/survives-restart
ARCHITECTURE.md     the full design + every measured number + honest boundaries
research/           the experiments that produced the numbers
```

## License

MIT.
