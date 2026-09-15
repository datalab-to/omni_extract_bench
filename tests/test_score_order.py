#!/usr/bin/env python3
"""Edge cases for order-dependent array scoring (`grade(..., order_matters=...)`).

Declaring an array ordered says its index is an ADDRESS rather than a position, which touches
three things at once: `align` stops solving that node, `_prepare` folds its leaves in with the
named ones, and the pairing weight therefore compares it positionally too. That last one is
where the first attempt went wrong -- rows distinguished only by an ordered inner array all
tied, and a perfect extraction scored 0.00 -- so the interactions get their own file.

Run: python3 tests/test_order_edges.py
"""
import copy
import itertools
import os as _os
import random
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench import matching as OM                              # noqa: E402
from omni_extract_bench.score import (                             # noqa: E402
    INDEX, KEY, explain, flatten, show, grade, node_key,
    format_node, _find_arrays)

# ── the scorer now requires a schema ──────────────────────────────────────────────────
# These tests are about scoring, not schema plumbing, so derive one from the ground truth.
# That is the realistic case anyway: gold conforms to the schema that was sent. Deriving it
# from gold alone is deliberate -- a key the PREDICTION invented genuinely is not a slot the
# model was offered, which is what tells `invented field` from `fabricated`.
from omni_extract_bench.score import grade as _grade_impl          # noqa: E402
from omni_extract_bench.score import explain as _explain_impl      # noqa: E402


def _schema_from(doc):
    if isinstance(doc, dict):
        return {"type": "object", "properties": {k: _schema_from(v) for k, v in doc.items()}}
    if isinstance(doc, list):
        merged = {}
        for e in doc:
            if isinstance(e, dict):
                merged.update(e)
        return {"type": "array", "items": _schema_from(merged) if merged else {}}
    return {}


def grade(pred, gt, schema=None, *a, **kw):
    return _grade_impl(pred, gt, schema or _schema_from(gt), *a, **kw)


def explain(pred, gt, schema=None, *a, **kw):
    return _explain_impl(pred, gt, schema or _schema_from(gt), *a, **kw)
# ──────────────────────────────────────────────────────────────────────────────────────


FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


def _raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return True
    except Exception:                                               # noqa: BLE001
        return False
    return False


def acc(pred, gold, ordered=(), schema=None):
    return grade(copy.deepcopy(pred), copy.deepcopy(gold), schema, ordered)["accuracy"]


def parts(pred, gold, ordered=(), schema=None):
    r = grade(copy.deepcopy(pred), copy.deepcopy(gold), schema, ordered)
    return r["matched"], r["total"]


# ═══════════════════════════════════════════════════════════════════════════════
print("\nLENGTH MISMATCH IN AN ORDERED ARRAY")
G = {"steps": ["mix", "bake", "cool"]}
STEPS = ["steps"]
short = parts({"steps": ["mix", "bake"]}, G, STEPS)
long_ = parts({"steps": ["mix", "bake", "cool", "eat"]}, G, STEPS)
inserted = parts({"steps": ["prep", "mix", "bake", "cool"]}, G, STEPS)
report("a short prediction leaves gold's tail missing", short == (2, 3), f"got {short}")
report("a long prediction has its tail charged as spurious", long_ == (3, 4), f"got {long_}")
report("an inserted element shifts everything after it, and that is the point",
       inserted == (0, 4), f"got {inserted}")
note("order-free would score the insertion 3/4; ordered charges the shift, by design")
report("an empty prediction charges every gold element",
       parts({"steps": []}, G, STEPS) == (0, 3))
report("an empty gold charges every predicted element",
       parts(G, {"steps": []}, STEPS) == (0, 3))

print("\nDUPLICATES AND OBJECTS INSIDE AN ORDERED ARRAY")
DUP = {"xs": ["x", "x", "y"]}
XS = ["xs"]
dup = parts({"xs": ["x", "y", "x"]}, DUP, XS)
report("duplicate values compare position by position", dup == (1, 3), f"got {dup}")
report("an ordered array of OBJECTS is positional too",
       parts({"rows": [{"a": 2}, {"a": 1}]}, {"rows": [{"a": 1}, {"a": 2}]},
             ["rows"]) == (0, 2))
report("...and identity still holds for one",
       abs(acc({"rows": [{"a": 1}, {"a": 2}]},
               {"rows": [{"a": 1}, {"a": 2}]}, ["rows"]) - 100) < 1e-9)
nulls = parts({"xs": ["a", None, "c"]}, {"xs": ["a", None, "c"]}, XS)
report("a null inside an ordered array occupies no address", nulls == (2, 2), f"got {nulls}")
note("index 1 yields no path on either side, so index 2 keeps its own address")

print("\nARRAYS OF ARRAYS, ONE LEVEL ORDERED AT A TIME")
GRID = {"grid": [[1, 2], [3, 4]]}
SWAP_OUTER = {"grid": [[3, 4], [1, 2]]}
SWAP_INNER = {"grid": [[2, 1], [4, 3]]}
report("nothing ordered: both levels are free",
       abs(acc(SWAP_OUTER, GRID) - 100) < 1e-9 and abs(acc(SWAP_INNER, GRID) - 100) < 1e-9)
oo, oi = acc(SWAP_OUTER, GRID, ["grid"]), acc(SWAP_INNER, GRID, ["grid"])
io, ii = acc(SWAP_OUTER, GRID, ["grid[*]"]), acc(SWAP_INNER, GRID, ["grid[*]"])
report("outer ordered only: the outer swap is charged, the inner is not",
       oo < 1e-9 and abs(oi - 100) < 1e-9, f"outer {oo:.2f} (want 0), inner {oi:.2f} (want 100)")
report("inner ordered only: the inner swap is charged, the outer is not",
       abs(io - 100) < 1e-9 and ii < 1e-9, f"outer {io:.2f} (want 100), inner {ii:.2f} (want 0)")
report("both ordered: a matrix, fully positional",
       abs(acc(GRID, GRID, ["grid", "grid[*]"]) - 100) < 1e-9
       and acc(SWAP_OUTER, GRID, ["grid", "grid[*]"]) < 1e-9)

# ═══════════════════════════════════════════════════════════════════════════════
print("\nTHE PAIRING WEIGHT MUST AGREE WITH THE SCORE")
# The first implementation only stopped ALIGNING ordered nodes. The weight still compared them
# order-free, so rows whose only difference was inside an ordered array all tied at (2,2), the
# matcher broke the tie arbitrarily, and a perfect extraction scored 0.00.
NEST_G = {"books": [{"chapters": ["a", "b"]}, {"chapters": ["b", "a"]}]}
NEST_P = {"books": [{"chapters": ["b", "a"]}, {"chapters": ["a", "b"]}]}
nest = acc(NEST_P, NEST_G, ["books[*].chapters"])
report("rows distinguished ONLY by an ordered inner array still pair correctly",
       abs(nest - 100) < 1e-9, f"got {nest:.2f}, want 100 (was 0.00)")
# and the same when the discriminator is deeper still
DEEP_G = {"a": [{"b": [{"c": ["x", "y"]}]}, {"b": [{"c": ["y", "x"]}]}]}
DEEP_P = {"a": [{"b": [{"c": ["y", "x"]}]}, {"b": [{"c": ["x", "y"]}]}]}
deep = acc(DEEP_P, DEEP_G, ["a[*].b[*].c"])
report("the same holds three levels down", abs(deep - 100) < 1e-9, f"got {deep:.2f}")
report("declaring an array ordered never LOWERS a correct answer's score",
       all(abs(acc(d, d, [format_node(n) for n in _find_arrays(list(flatten(d)))]) - 100) < 1e-9
           for d in (NEST_G, DEEP_G, GRID, G, DUP)))

# ═══════════════════════════════════════════════════════════════════════════════
print("\nPATTERNS NAME SCHEMA LOCATIONS, AND ARE UNAMBIGUOUS")
report("one pattern covers every instance of a nested array",
       format_node(((KEY, "b"), (INDEX, 0), (KEY, "c")))
       == format_node(((KEY, "b"), (INDEX, 99), (KEY, "c"))) == "b[*].c")
# A key containing the path delimiters must not be mistakable for structure -- the same hazard
# the tagged key entries exist to prevent, re-established after formatting throws the tags away.
report("a nested path and a dotted key are DIFFERENT names, structurally",
       node_key(((KEY, "a"), (KEY, "b"))) == ((KEY, "a"), (KEY, "b"))
       and node_key(((KEY, "a.b"),)) == ((KEY, "a.b"),)
       and node_key(((KEY, "a"), (KEY, "b"))) != node_key(((KEY, "a.b"),)))
report("a key named \"*\" is not an index step",
       node_key(((KEY, "*"),)) == ((KEY, "*"),)
       and node_key(((INDEX, 0),)) == ((INDEX, None),)
       and node_key(((KEY, "*"),)) != node_key(((INDEX, 0),)))
note("a formatted string collided on both of these; a tuple cannot")
DOTTED_G, DOTTED_P = {"a.b": ["x", "y"]}, {"a.b": ["y", "x"]}
NESTED_G, NESTED_P = {"a": {"b": ["x", "y"]}}, {"a": {"b": ["y", "x"]}}
report("the dotted key and the nested path get DIFFERENT names, each charged on its own",
       acc(DOTTED_P, DOTTED_G, ["['a.b']"]) < 1e-9
       and acc(NESTED_P, NESTED_G, ["a.b"]) < 1e-9)
note("the quoting is what keeps these apart: \"['a.b']\" is the key, \"a.b\" is the path")
report("and naming one when the document holds the other is an error, not a no-op",
       all(_raises(ValueError, lambda: acc(pp, gg, [nm]))
           for pp, gg, nm in ((DOTTED_P, DOTTED_G, "a.b"),
                              (NESTED_P, NESTED_G, "['a.b']"))))
report("format_node remains available for display, and quotes the dotted key",
       format_node(((KEY, "a"), (KEY, "b"))) == "a.b"
       and format_node(((KEY, "a.b"),)) == "['a.b']")

print("\nNODE KEYS SURVIVE A SECOND COPY OF THE MODULE")
# There used to be an EACH sentinel object standing in for an index inside a node key, and it
# compared by IDENTITY. A second copy of this module (vendored, reloaded, imported by two
# paths) built its own, so a node key from one copy matched nothing in the other: the array
# was aligned instead of held in order, silently, with no error. Found exactly that way,
# comparing this module against a refactored copy of itself.
#
# A node key is now an address with the index VALUES blanked and the ('i', ...) tag kept, so
# it is plain tuples of strings and None all the way down. Two copies cannot disagree about
# those. The class of bug is gone rather than guarded, but the check stays: it is the thing
# that would notice a sentinel creeping back in.
import importlib.util as _ilu                                              # noqa: E402

_spec = _ilu.spec_from_file_location(
    "omni_extract_bench._score_second_copy",
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                  "omni_extract_bench", "score.py"))
_copy = _ilu.module_from_spec(_spec)
_copy.__package__ = "omni_extract_bench"          # so its relative imports resolve
_sys.modules[_spec.name] = _copy
_spec.loader.exec_module(_copy)

BOOKS = ((KEY, "books"), (INDEX, 0), (KEY, "chapters"))
report("a node key holds nothing but strings and None",
       all(isinstance(part, tuple) and (part[1] is None or isinstance(part[1], str))
           for part in node_key(BOOKS)),
       f"got {node_key(BOOKS)!r}")
report("so a node key built by one copy matches a frozenset built by the other",
       _copy.node_key(BOOKS) in frozenset({node_key(BOOKS)}))
CROSS_G = {"books": [{"chapters": ["a", "b"]}]}
CROSS_P = {"books": [{"chapters": ["b", "a"]}]}
report("and a name resolved by one copy applies in the other",
       _copy.grade(copy.deepcopy(CROSS_P), copy.deepcopy(CROSS_G),
                   _schema_from(CROSS_G), ["books[*].chapters"])["accuracy"] < 1e-9
       and acc(CROSS_P, CROSS_G, ["books[*].chapters"]) < 1e-9)
note("under the old identity comparison this scored 100.00: the config silently did not apply")

# The blanked index must stay distinguishable from every key a document can hold.
report("a blanked index equals no key a document can contain",
       all(node_key(((INDEX, 0),)) != node_key(((KEY, v),))
           for v in ("*", "EACH", "None", "", 0, 1, True, False, None, (), 3.5)))
report("...including a key literally named \"*\"",
       node_key(((KEY, "*"),)) != node_key(((INDEX, 0),)))

print("\nA NAME THAT FITS NO ARRAY IS AN ERROR, NOT A NO-OP")
# This was silent. A typo left the array order-free -- which is the DEFAULT -- so the run
# finished and reported a number that looked entirely plausible. Every mistake below scored
# 100.00 before, on a document whose steps are reversed.
SG, SP = {"steps": ["a", "b"]}, {"steps": ["b", "a"]}
for why, name in (("a misspelt key", "stepz"),
                  ("the right key in the wrong case", "Steps"),
                  ("a key that is not an array at all", "nope"),
                  ("the ELEMENTS of steps rather than steps itself", "steps[*]"),
                  ("a name that forgot its nesting", "w.steps")):
    report(f"{why} is refused", _raises(ValueError, lambda n=name: acc(SP, SG, [n])))
try:
    acc(SP, SG, ["stepz"])
except ValueError as exc:
    report("...and the error lists the arrays that DO exist", "'steps'" in str(exc), str(exc))

report("a correct name still charges the reordering", acc(SP, SG, ["steps"]) < 1e-9)
report("an empty configuration equals no configuration", acc(SP, SG, []) == acc(SP, SG))
for kind, value in (("list", ["steps"]),
                    ("tuple", ("steps",)),
                    ("set", {"steps"}),
                    ("frozenset", frozenset({"steps"})),
                    ("generator", (x for x in ["steps"]))):
    report(f"accepts a {kind}", acc(SP, SG, value) < 1e-9)
note("a name is a string now, so a tuple of names is no longer mistakable for one name")

# A bare string is iterable, so without a guard `order_matters="steps"` would iterate as
# characters and complain about 's'.
report("a bare string is refused, with the fix in the message",
       _raises(TypeError, lambda: acc(SP, SG, "steps")))
try:
    acc(SP, SG, "steps")
except TypeError as exc:
    report("...naming the list the caller meant", "['steps']" in str(exc), str(exc))

# ═══════════════════════════════════════════════════════════════════════════════
print("\nPATTERNS ARE ABSOLUTE, SO DEPTH IS NOT INVARIANT (by design)")
FLAT_G, FLAT_P = {"steps": ["a", "b"]}, {"steps": ["b", "a"]}
report("wrapping a document makes the old name an error rather than a quiet no-op",
       acc(FLAT_P, FLAT_G, ["steps"]) < 1e-9
       and _raises(ValueError, lambda: acc({"w": FLAT_P}, {"w": FLAT_G}, ["steps"])))
report("the wrapped name restores the behaviour",
       acc({"w": FLAT_P}, {"w": FLAT_G}, ["w.steps"]) < 1e-9)
note("a name locates an array; move the array and the name must move with it -- but now")
note("the stale name says so, instead of scoring the document order-free")
note("nesting invariance still holds with NO configuration, which is the tested property")

# ═══════════════════════════════════════════════════════════════════════════════
print("\nPROPERTIES THAT MUST SURVIVE A CONFIGURATION")
def rand_doc(rnd, depth=0):
    d = {}
    for i in range(rnd.randint(2, 4)):
        r = rnd.random()
        if r < 0.35 and depth < 2:
            d[f"arr{i}"] = [{f"f{j}": rnd.choice([1, 2, "x", "y"]) for j in range(3)}
                            for _ in range(rnd.randint(1, 4))]
        elif r < 0.6:
            d[f"sarr{i}"] = [rnd.choice([1, 2, "x", "y"]) for _ in range(rnd.randint(1, 4))]
        elif r < 0.75 and depth < 2:
            d[f"obj{i}"] = rand_doc(rnd, depth + 1)
        else:
            d[f"s{i}"] = rnd.choice([1, 2, "x", "y", None])
    return d

ok_id = ok_det = ok_mono = True
for s_ in range(150):
    rnd = random.Random(30000 + s_)
    gold = rand_doc(rnd)
    if not flatten(gold):
        continue
    every = [format_node(n) for n in _find_arrays(list(flatten(gold)))]
    subset = [p for p in every if rnd.random() < 0.5]
    if abs(acc(gold, gold, every) - 100) > 1e-9 or abs(acc(gold, gold, subset) - 100) > 1e-9:
        ok_id = False
        report("identity", False, f"seed {s_} with ordered={subset}")
        break
    runs = {repr(sorted(grade(copy.deepcopy(gold), copy.deepcopy(gold), None,
                                    subset).items(), key=str)) for _ in range(3)}
    if len(runs) != 1:
        ok_det = False
        break
report("identity holds with any subset of arrays declared ordered", ok_id)
report("determinism holds with a configuration", ok_det)

for s_ in range(200):
    rnd = random.Random(31000 + s_)
    gold = [{f"f{j}": rnd.choice(["A", "B", "C"]) for j in range(3)} for _ in range(3)]
    pred = copy.deepcopy(gold)
    wrong = []
    for i, row in enumerate(pred):
        for f in row:
            if rnd.random() < 0.4:
                row[f] = "WRONG"
                wrong.append((i, f))
    if not wrong:
        continue
    before = acc({"rows": pred}, {"rows": gold}, ["rows"])
    for i, f in wrong:
        fixed = copy.deepcopy(pred)
        fixed[i][f] = gold[i][f]
        if acc({"rows": fixed}, {"rows": gold}, ["rows"]) < before - 1e-9:
            ok_mono = False
            report("monotonicity", False, f"seed {s_}: fixing row {i}.{f} lowered the score")
            break
    if not ok_mono:
        break
report("correcting a leaf never lowers the score of an ordered array", ok_mono)

# ═══════════════════════════════════════════════════════════════════════════════
print("\nINTERACTION WITH EVERYTHING ELSE")
OM_SCH = {"type": "object", "properties": {
    "bag": {"type": "object", "additionalProperties": {"type": "array",
                                                       "items": {"type": "string"}}}}}
OM_D = {"bag": {"H": ["a", "b"]}}
report("an ordered array inside a skipped open map is simply never reached",
       grade(copy.deepcopy(OM_D), copy.deepcopy(OM_D), OM_SCH,
                   [])["skipped_open_maps"] == ["bag"])
report("explain reports positional verdicts for an ordered array",
       [v.verdict for v in explain({"steps": ["b", "a"]}, {"steps": ["a", "b"]},
                                      None, ["steps"])] == ["misread", "misread"])
report("show and format_node agree except for the blanked indices",
       show(((KEY, "b"), (INDEX, 3), (KEY, "c"))) == "b[3].c"
       and format_node(((KEY, "b"), (INDEX, 3), (KEY, "c"))) == "b[*].c")

# An inner array is re-priced once per candidate outer pairing, so an approximate inner solve
# would otherwise be reported once per evaluation -- 20 entries for one array in a 4-row doc.
BIG_G = {"rows": [{"k": i % 2, "xs": [{"v": j % 3, "w": "s"} for j in range(12)]}
                  for i in range(4)]}
BIG_P = {"rows": [{"k": i % 2, "xs": [{"v": j % 3, "w": "s"} for j in reversed(range(12))]}
                  for i in range(4)]}
restore = OM.force_approximate()
big = grade(BIG_P, BIG_G)
restore()
report("greedy_blocks reports distinct sizes, not one entry per weight evaluation",
       len(big["approximated"]) == 1 and big["matching_exact"] is False,
       f"got {big['approximated']}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\nDEGENERATE SHAPES, WITH A CONFIGURATION")
DEGEN = [({}, {}), ({"a": []}, {"a": []}), ({"a": [[]]}, {"a": [[]]}),
         ({"a": [{}]}, {"a": [{}]}), ({"a": []}, {"a": ["x"]}), ({"a": ["x"]}, {"a": []}),
         ({"a": [None, None]}, {"a": [None, None]}), ({}, {"a": ["x"]}), ({"a": ["x"]}, {})]
ok, detail = True, None
for pred, gold in DEGEN:
    try:
        names = [format_node(n) for n in
                 _find_arrays(list(flatten(gold)) + list(flatten(pred)))]
        r = grade(copy.deepcopy(pred), copy.deepcopy(gold), None, names)
        if not (0.0 <= r["accuracy"] <= 100.0):
            ok, detail = False, f"{pred} vs {gold} -> {r['accuracy']}"
            break
    except Exception as exc:                                        # noqa: BLE001
        ok, detail = False, f"{pred} vs {gold} raised {type(exc).__name__}: {exc}"
        break
report("empty and degenerate arrays survive being declared ordered", ok, detail)

print(f"\n{'ORDER EDGE CASES HOLD' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
