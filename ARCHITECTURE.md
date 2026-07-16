# The Life Debugger — a cost-optimized, never-bluff debugging architecture

**One line:** handle the *finite, common* bugs exactly and for **$0** with an execute‑and‑recognize
front‑end (the "Life"), and escalate only the *genuinely‑novel reasoning* to an LLM — so a Python
debugger costs a fraction of pure‑LLM, is deterministic, and never makes an answer up.

> Every number in this document comes from **running code**, not from claims. The measurements and the
> honest boundaries are both part of the design — a debugger that lies about its own limits isn't one.

---

## 1. The idea

Debugging with an LLM on *every* bug is expensive and non‑deterministic. But most real bugs — especially
for students and everyday development — are **mechanical**: the shape of the code *forces* an error
(`NameError`, `TypeError`, `IndexError`, syntax). That class is **finite and coverable**. The genuinely
hard part — **semantic** bugs, where the code runs fine but does the *wrong thing* — is **infinite** and
needs reasoning.

This architecture splits the two:

- A **cheap path** (execute + parse + a curated fact‑memory, "the Life") that resolves the finite
  mechanical class **exactly, instantly, for $0, and never bluffs**.
- An **escalation** to an LLM reasoner for the semantic / novel residue.

The output quality matches the LLM (the hard bugs go to it); the *cost* is a fraction of it (the
mechanical majority is free).

---

## 2. Architecture

```
                 failing code / test
                         │
          ┌──────────────▼───────────────┐
          │   CHEAP PATH  (the Life)      │   $0 · instant · deterministic · never‑bluff
          │   • parse        → SyntaxError │
          │   • execute      → catch type  │
          │   • call‑graph localize        │   193 functions → the 6 on the failing chain (exact)
          │   • fact‑memory: error → a      │
          │     friendly explanation + fix  │   curated; per‑user "alive", learns YOUR recurring bugs
          └──────────────┬───────────────┘
                         │
        recognized mechanical bug ──► EXPLAIN + FIX      ($0, deterministic)
                         │
             abstain (ran clean but wrong, or a novel pattern)
                         │
          ┌──────────────▼───────────────┐
          │   SMALL REASONING NET (~3 GB)  │   local, cheap; reasons the fix from context
          │   fixes ~88% of common          │   (measured: qwen-coder-7B, 35/40 = 88%)
          │   intent-dependent bugs         │
          └──────────────┬───────────────┘
                         │ (novel / subtle / hard tail only)
          ┌──────────────▼───────────────┐
          │   FRONTIER REASONER (Opus…)    │   the genuinely-novel reasoning
          └──────────────────────────────┘
```

**Three tiers, most of it local and free:** (1) facts + execute explain ~70% and auto-fix the canonical
13% for $0; (2) a *small* local net (~3 GB) *reasons* the fix for ~88% of the common intent-dependent bugs
(measured on a 7B model, ~2 s each, no API cost); (3) escalate to a frontier model only for the novel tail.
The facts power tier 1 (and personalize/RAG the others); the *reasoning* — not the facts — does the fixing
in tiers 2–3. A small net suffices for the *common* cases; the frontier model still wins the *novel* tail.

**The gate is the whole trick:** the Life resolves what it *exactly* recognizes and **abstains** on
everything else — it never guesses. Abstention is what makes the escalation trustworthy.

---

## 3. Measured results

### Coverage across the Python difficulty spectrum
What the cheap path resolves for **$0** vs what it escalates to the LLM (each bug validated by running):

| Level | bugs tested | cheap path ($0) | → LLM |
|---|---:|---:|---:|
| **Beginner**     | 166 | **70%** | 30% |
| **Intermediate** |  80 | **55%** | 45% |
| **Advanced**     |  77 | **34%** | 66% |
| **Research**     |  —  | ~0%     | ~100% |

Coverage **declines with difficulty** because harder bugs are increasingly *silent‑logic* (they raise no
exception) — and only the reasoner catches those. This is expected and honest: the architecture is most
valuable at the beginner/student end, where both bug *volume* and the mechanical *fraction* are highest.

### Never bluffs
Across the **47** silent‑logic beginner bugs, the cheap path abstained **47/47**. It never once claimed to
have caught a bug it hadn't. **Zero false‑handles.** That property is the point.

### Can you store *semantic* bugs as facts too?
Tested on **133 validated silent-logic bugs** across the spectrum. A semantic anti-pattern recognizer
(mutable-default-arg, bare-except, float `==`, `is`-with-literal, mutate-while-iterating, missing-`await`,
return-in-finally) — the "semantic fact base" — caught **9 / 133 = 7%**. The other **93% are
intent-dependent** (off-by-one, wrong variable, wrong operator): the buggy code is *pattern-identical to
correct code*, so "wrong" is only definable relative to the program's intent — which is not in any fact
base. Storing every past semantic bug does **not** cover them. (And the 7% that *are* catchable are exactly
the rules `pylint`/`flake8`/`mypy` already ship.) **Conclusion: the reasoner is a permanent part of the
design, not a crutch removed once you have "enough" facts. Facts can't hold intent — at any size.**

### Detect vs. explain vs. auto-fix (the number that reframes the product)
On the validated mechanical bugs, verified end-to-end **auto-fix** (a pattern fix that makes the code
actually run) was **beginner 13% · intermediate 2% · advanced 0%** — far below the ~70% *detection* rate.
Only **typos** (fuzzy-match, name/method) and **missing colons** have a *canonical* fix that needs no
intent. For everything else — `TypeError`, `IndexError`, `KeyError`, `ZeroDivisionError` — the crash is
obvious but the *fix* is intent-dependent (convert which operand? which index? guard or default?).
**Fixing is a reasoning act even when detecting isn't.** So this system is a **teaching *explainer***, not
an auto-fixer: it tells a student *what went wrong and where* for ~70% of their bugs (free, deterministic,
never-bluff), auto-fixes the ~13% that are typos/syntax, and hands the rest — *with a clear explanation* —
to the student's own reasoning or an LLM. For beginners that explanation is most of the value.

### Component measurements
- **Mechanical detector** on a 500‑line multi‑bug file: **23 / 29 bugs caught in 1.1 s, $0, 0 tokens.**
- **Call‑graph localizer (the Life):** narrows a **193‑function / 3,875‑line repo → the 6 functions on the
  failing call chain, 100% recall, deterministic, no training.**
- **vs a real code LLM** (qwen2.5‑coder‑7b) on mechanical bugs: the LLM scored 87% but was
  **non‑deterministic** (same diagnosis on rerun only 5/15); the exact path is deterministic and $0.
- **Token cost of localization** (frontier LLM debugging one bug in a repo): a function+variable hint cut
  **~10.8%** of the search tokens — a real but bounded saving (the reasoning, not the search, is ~90% of
  the cost).

---

## 4. Honest boundaries (read before believing the pitch)

- **This is a cost + reliability layer, not a smarter debugger.** The reasoning lives in the LLM. The
  cheap path only handles what is *finite and recognizable*.
- **Coverage declines with difficulty** (table above). Advanced/semantic bugs mostly escalate.
- **"Handled" = detect + classify + explain/fix a mechanical bug.** Raw detection is largely *free from the
  Python interpreter itself*; the Life's added value is the **friendly, personalized explanation + fix**
  layer and the per‑user memory.
- **A fact base covers bug *types*, not *locations*.** Knowing "off‑by‑one is common" doesn't tell you
  *which* `range()` is wrong in a novel program — that's reasoning, and it stays in the LLM. (Measured: a
  net trained on common‑mistake patterns scored 88% on its own distribution but **0/5** at pinpointing the
  buggy *line* in real code.)
- **Research‑frontier / novel‑algorithm code is out of scope** — it is all novel reasoning.

---

## 4b. The alive learning loop (the flywheel)

Every time a bug escalates to the frontier reasoner, the Life **caches the fix**. Next time that bug — or,
via the net's fuzzy match, a *similar* one — appears, the cache handles it for free. Measured on a replayed
stream: escalation fell **100% → 0%** for anything already seen; the frontier is called **less over time.**

Measured on a 600-bug recurring stream: reasoning-model calls fell **70% → 12%** as the Life learned each
pattern, cutting total model calls **~63%** (218 vs 600) — each pattern is solved *once* by a model, then
replayed free. Three honest caveats: (1) the **exact** Life cache only hits *identical* recurrences —
generalizing a cached fix to a *new-but-similar* bug needs the **net** (net+Life, not Life alone); (2) the
free replay is *full* for **canonical** fixes (`is`→`==`, closure→default-arg) but *partial* for
**context-specific** ones (a lock/timeout must still be *placed* by the net — the Life saves the recognition,
not the application); (3) **genuinely-novel bugs always escalate**, so the call rate plateaus at the
novel-bug rate, not zero. But across a user base the *first*
user pays the frontier and everyone after gets it free, so the aggregate escalation rate falls toward the
rate of bugs *new to everyone* — low for common bugs, nonzero only on the long tail. **The system gets
cheaper the more it is used, without ever pretending the reasoner is gone.**

## 5. Why it works (the principle)

Every bug is either **mechanical** — a deterministic consequence of code shape, therefore *finite,
storable, recognizable* — or **semantic** — behavior ≠ intent, therefore *infinite* and requiring
reasoning. The Life owns the first class exactly and honestly; the LLM owns the second.

**Personalization shrinks the second class for a given user.** The "alive" Life accumulates *your*
libraries, idioms, and recurring mistakes as you code, so more of *your* everyday bugs fall into the free
path over time. You are not covering infinite Python — you are making the *finite recurring slice of one
person's world* free, and escalating only what's genuinely new to them.

---

## 6. Components (this repo)

| Component | File | Role |
|---|---|---|
| Mechanical detector | (execute+catch) | run code, catch the error type — the finite class |
| Call‑graph localizer | `net_life_localizer.py` | Life traverses calls → narrows repo to the failing chain (exact) |
| Error → explain/fix memory | (the Life) | curated, per‑user "alive" fact base; friendly fixes |
| Abstain / escalate gate | (the Life) | never‑bluff: resolve if recognized, else hand to the LLM |
| Reasoner adapter | (LLM) | Opus / any model for the semantic/novel residue |
| Fusion experiments | `net_life_fused_scout.py`, `error_space_coverage.py`, `calibrated_abstain.py` | the research that produced the measured numbers above |

---

## 7. Status

Research prototype, honestly scoped and measured:

> **A debugging front‑end that resolves ~70% of a beginner's bugs and ~34% of advanced ones for free,
> deterministically, never bluffs, and escalates the rest to an LLM at the LLM's own quality.**

Cheapest exactly where the volume is (students / everyday dev), positive through intermediate, thin at
advanced, out at research. That's the whole honest story — and it's a real one.
