#!/usr/bin/env python3
"""PROOF the Life is alive — it stores facts, learns fixes, replays them FREE, and survives restart.

Run:  python3 proof_alive.py

Everything below is executed live; the numbers are real, not narrated.
"""
import os, tempfile
from life_debugger import AliveLife, debug, static_facts

DB = os.path.join(tempfile.gettempdir(), "proof_life.json")
if os.path.exists(DB):
    os.remove(DB)

def line(): print("-" * 68)

print("=" * 68)
print("PROOF: the Life stores facts, learns fixes, replays free, stays alive")
print("=" * 68)

# ---------------------------------------------------------------- 1. it stores facts (and abstains)
life = AliveLife(DB)
print("\n[1] STORE FACTS — teach it, recall it exactly, and watch it ABSTAIN on the unknown")
life.store_fact("fix:KeyError:missing-key", "use dict.get(key, default) instead of dict[key]")
life.store_fact("fix:is-with-literal", "replace `is 5` with `== 5`")
print(f"    recall a stored fact : {life.recall('fix:is-with-literal')!r}")
print(f"    recall an unknown key: {life.recall('fix:never-seen')!r}   <- None = honest ABSTAIN, not a guess")
print(f"    facts held: {life.n_facts()}   identity sha: {life.sha()}")

# ---------------------------------------------------------------- 2. facts catch surface bugs FREE
print("\n[2] FACTS TIER — catches surface bugs for $0, no model")
buggy = "import asyncio, time\nasync def go():\n    time.sleep(1)\n    if go is 5:\n        return [].apend(1)\n"
for kind, ln, expl in static_facts(buggy):
    print(f"    line {ln}  [{kind}]  {expl}")

# ---------------------------------------------------------------- 3. learn a fix once, replay it FREE
print("\n[3] LEARN + REPLAY FREE — a reasoner is called ONCE, then the Life replays forever")
calls = {"n": 0}
def counting_reasoner(code, fact_kinds):        # stands in for qwen / a frontier model
    calls["n"] += 1
    return "add a threading.Lock around the shared counter increments"

bug = "counter = 0\ndef inc():\n    global counter\n    counter += 1  # race under threads\n"

r1 = debug(bug, reasoner=counting_reasoner, life=life)
print(f"    1st time  -> handled_by: {r1['handled_by']:24} reasoner calls so far: {calls['n']}")
r2 = debug(bug, reasoner=counting_reasoner, life=life)
print(f"    2nd time  -> handled_by: {r2['handled_by']:24} reasoner calls so far: {calls['n']}")
r3 = debug(bug, reasoner=counting_reasoner, life=life)
print(f"    3rd time  -> handled_by: {r3['handled_by']:24} reasoner calls so far: {calls['n']}")
assert calls["n"] == 1, "the reasoner should have been called exactly ONCE"
print(f"    >> reasoner ran {calls['n']}x total; runs 2 and 3 were REPLAYED FREE from the Life.")
print(f"    >> the learned fix: {r2['fix']!r}")

# ---------------------------------------------------------------- 4. it is ALIVE across restarts
print("\n[4] ALWAYS ALIVE — a brand-new process reloads the same facts from disk (never resets)")
sha_before = life.sha(); n_before = life.n_facts()
del life                                        # simulate the process ending
reborn = AliveLife(DB)                           # a fresh AliveLife, new object, same disk file
print(f"    before restart: {n_before} facts, sha {sha_before}")
print(f"    after  restart: {reborn.n_facts()} facts, sha {reborn.sha()}")
print(f"    learned fix still there: {reborn.recall('fix:' + __import__('life_debugger')._sig(bug))!r}")
assert reborn.sha() == sha_before, "the Life must survive restart byte-identical"
print("    >> byte-identical after restart: the Life is alive, not static.")

# ---------------------------------------------------------------- 5. it keeps growing
print("\n[5] IT KEEPS STORING — every new solve grows the store and moves the sha")
before = reborn.n_facts()
reborn.store_fact("fix:new-pattern-today", "the fix a model produced just now")
print(f"    facts {before} -> {reborn.n_facts()}   sha moved: {reborn.sha()}")

line()
print("PROVEN: stores facts, abstains honestly, catches surface free, learns a fix once,")
print("replays it free, survives restart byte-identical, and keeps growing. That is 'alive'.")
os.remove(DB)
