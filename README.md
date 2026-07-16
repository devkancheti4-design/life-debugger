# The Life Debugger

A small Python debugger that explains most of your bugs **instantly and for free**, and only asks a bigger
AI model for help on the genuinely hard ones. It **remembers every fix it learns**, so it gets more useful
the more you use it — and it never makes an answer up.

```bash
git clone https://github.com/devkancheti4-design/life-debugger.git
cd life-debugger
python3 proof_alive.py     # watch it learn a fix once, then reuse it for free
```

## Why

Most bugs are the same handful of mistakes — a typo, a missing colon, comparing with `is` instead of `==`,
the wrong kind of value. You shouldn't need a slow, paid AI model to point those out. The Life Debugger
catches them instantly for free, explains them in plain English, and saves the expensive model for the
tricky stuff. When it *does* ask a model, it remembers the answer — so next time, that fix is free too.

## Use it

No dependencies, no API key needed for the basics:

```python
from life_debugger import debug

report = debug(open("my_script.py").read())
for kind, line, why in report["facts"]:
    print(f"line {line}: {why}")
```

Want it to fix the harder bugs too? Plug in any model you have:

```python
def my_model(code, hints):
    return call_whatever_model_you_like(code)

debug(code, reasoner=my_model)   # asks the model once, then remembers the fix forever
```

## How it works

It tries the cheapest thing first and only escalates when it has to:

1. **Known patterns** — catches common bugs instantly, for free, and never guesses.
2. **Memory** — if it has solved this before, it reuses the fix for free.
3. **A model** — for something new, it asks an AI, then *remembers* the answer.

The memory is the "alive" part: it saves to disk and reloads every run, so it never forgets — it just keeps
getting more helpful over time.

## Honest about the limits

It won't fix everything, and it won't pretend to. It's genuinely good at the common, mechanical bugs — most
of what you hit day to day — and honest about the rest: it says *"I'm not sure, ask the model"* instead of
inventing a fix. If you want the details — how it works, what it's been measured to do, and exactly where it
falls short — it's all in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

## What's in here

```
life_debugger.py   the debugger you import
life.py            the memory that remembers fixes (one file, no dependencies)
proof_alive.py     run it to see the memory learn and reuse a fix
ARCHITECTURE.md    the full design, the numbers, and the honest limits
```

## License

[AGPL-3.0](LICENSE) — free and open. If you run a modified version as a service, please share your changes.
