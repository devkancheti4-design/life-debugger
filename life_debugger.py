#!/usr/bin/env python3
"""The Life Debugger — a four-tier, mostly-local, never-bluff Python debugger.

    TIER 1  FACTS     static anti-patterns + execute-and-catch — the finite, certain surface ($0)
    TIER 2  LIFE      an ALIVE, persistent cache of every fix the reasoners have produced (replays free)
    TIER 3  NET       a small local reasoning model (e.g. ollama qwen) for common intent-dependent bugs
    TIER 4  FRONTIER  a frontier model (optional) for the genuinely-novel tail

The Life is ALWAYS ALIVE: it loads its facts from disk on startup and saves after every learn, so it
accumulates across runs and sessions — it never resets to static. Each fix a reasoner produces is learned
ONCE and replayed free forever after (the flywheel). Honest boundaries are in ARCHITECTURE.md.

Usage:
    from life_debugger import debug, AliveLife
    report = debug(open("buggy.py").read())          # facts only, free
    report = debug(code, reasoner=my_model_fn)        # escalate+learn the intent-dependent ones
"""
import ast, re, os, sys, subprocess, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from life import Life

DEFAULT_DB = os.path.expanduser("~/.life_debugger.json")

# ============================================================ the ALIVE, PERSISTENT Life
class AliveLife:
    """A persistent fact + fix store. Loads on init, saves after every learn — never goes static."""
    def __init__(self, path=DEFAULT_DB):
        self.path = path
        self.mem = Life(path)                    # <- loads existing facts if the file exists (ALIVE across runs)

    def known_fix(self, sig):                    # has a reasoner already solved this exact bug?
        return self.mem.recall("fix:" + sig)

    def learn_fix(self, sig, fix, source="reasoner"):
        self.mem.learn("fix:" + sig, fix)        # remember the fix
        self.mem.learn("src:" + sig, source)     # remember who solved it (net / frontier)
        self.mem.save()                          # persist NOW -> survives restart, never static

    def store_fact(self, key, value):            # teach it a fact directly (it keeps everything)
        self.mem.learn(key, value); self.mem.save()

    def recall(self, key):  return self.mem.recall(key)   # exact recall, or None = honest ABSTAIN
    def n_facts(self):      return len(self.mem.store)
    def sha(self):          return self.mem.sha()          # deterministic identity of the whole memory


# ============================================================ TIER 1: FACTS (static, $0, never-bluff)
def static_facts(code):
    """Return [(kind, line, explanation)] for the finite anti-pattern set. Never guesses."""
    out = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [("SyntaxError", e.lineno, f"syntax error: {e.msg}")]
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare):
            for op, c in zip(n.ops, n.comparators):
                if isinstance(op, (ast.Is, ast.IsNot)) and isinstance(c, ast.Constant) and not isinstance(c.value, (bool, type(None))):
                    out.append(("is-with-literal", n.lineno, "use `==` not `is` to compare values (identity vs equality)"))
        if isinstance(n, ast.ExceptHandler) and (n.type is None or (isinstance(n.type, ast.Name) and n.type.id in ("Exception", "BaseException"))):
            out.append(("broad-except", n.lineno, "broad `except` hides real errors; catch a specific type and log the traceback"))
        if isinstance(n, ast.FunctionDef) and n.name == "__del__":
            out.append(("del-cleanup", n.lineno, "`__del__` is unreliable for cleanup; use a context manager (`with`)"))
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in n.args.defaults:
                if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                    out.append(("mutable-default-arg", n.lineno, "mutable default arg is shared across calls; use `None` then create inside"))
        if isinstance(n, ast.AsyncFunctionDef):
            for x in ast.walk(n):
                if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and getattr(x.func.value, "id", "") == "time" and x.func.attr == "sleep":
                    out.append(("blocking-in-async", x.lineno, "`time.sleep` blocks the event loop; use `await asyncio.sleep`"))
        if isinstance(n, ast.For):
            its = ast.unparse(n.iter); tgt = getattr(n.target, "id", None)
            for x in ast.walk(n):
                if isinstance(x, ast.Lambda):
                    for y in ast.walk(x):
                        if isinstance(y, ast.Name) and y.id == tgt:
                            out.append(("late-binding-closure", x.lineno, "closure captures the loop var late; bind it as a default arg `lambda p, r=rule:`"))
                if isinstance(x, ast.Call) and isinstance(x.func, ast.Attribute) and x.func.attr in ("remove", "append", "pop", "insert") and ast.unparse(x.func.value) == its:
                    out.append(("mutate-while-iterating", x.lineno, "modifying a list while iterating it skips elements; iterate over a copy"))
    return out

def runtime_fact(code, test):
    """Execute-and-catch: run the code and report the exact error it raises (mechanical class). $0."""
    script = code + (f"\n{test}" if test else "")
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=6)
    except subprocess.TimeoutExpired:
        return ("Timeout", "code hangs (possible infinite loop / deadlock)")
    if r.returncode != 0:
        last = [l for l in r.stderr.strip().split("\n") if l]
        return ((last[-1].split(":")[0].strip() if last else "Error"), last[-1] if last else "runtime error")
    return None


def _sig(code):
    return hashlib.md5(re.sub(r"\s+", " ", code).encode()).hexdigest()[:16]


# ============================================================ the orchestrator
def debug(code, test=None, reasoner=None, life=None):
    """Four-tier debug. FACTS are always free. If a reasoner already solved this bug, the Life REPLAYS
    the fix free; otherwise the reasoner solves it and the Life LEARNS it for next time.

    reasoner: callable(code, fact_kinds) -> fix_string. Plug in ollama/qwen or a frontier model.
    Returns {facts, fix, handled_by, escalated, life_facts, life_sha}.
    """
    life = life or AliveLife()
    report = {"facts": [], "fix": None, "handled_by": None, "escalated": False}

    # TIER 1 — FACTS: free, deterministic, never-bluff
    report["facts"] = static_facts(code)
    rt = runtime_fact(code, test)
    if rt:
        report["facts"].append((rt[0], None, rt[1]))

    sig = _sig(code)

    # TIER 2 — ALIVE LIFE: has a reasoner solved this exact bug before? replay it free.
    cached = life.known_fix(sig)
    if cached:
        report["fix"] = cached
        report["handled_by"] = "life-cache (free, $0)"
    elif reasoner is not None:
        # TIER 3/4 — escalate, then LEARN so it never costs again
        fix = reasoner(code, [f[0] for f in report["facts"]])
        report["fix"] = fix
        report["handled_by"] = "reasoner (escalated)"
        report["escalated"] = True
        life.learn_fix(sig, fix, source="reasoner")
    else:
        report["handled_by"] = "facts-only (no reasoner attached)"

    report["life_facts"] = life.n_facts()
    report["life_sha"] = life.sha()
    return report


if __name__ == "__main__":
    src = open(sys.argv[1]).read() if len(sys.argv) > 1 else "def f(x):\n    if x is 5:\n        return x\n"
    rep = debug(src)
    print("FACTS (free):")
    for kind, line, expl in rep["facts"]:
        print(f"  line {line}  [{kind}]  {expl}")
    print(f"handled_by: {rep['handled_by']}  |  Life holds {rep['life_facts']} facts  |  sha {rep['life_sha']}")
