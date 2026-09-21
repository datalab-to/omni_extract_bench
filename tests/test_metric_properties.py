#!/usr/bin/env python3
"""Property-based tests for the metric spec (METRIC_SPEC.md P1..P23), against `metric.py`.

Generative, not example-based: random nested documents are built and perturbed, so the
properties are checked over a space of shapes rather than a handful of cases I happened to
think of. Deterministic seeds — a failure is always reproducible.

This file was written against the paired walker that `metric.py` replaced, and was pointed at
`metric.py` when that walker was deleted. Worth knowing, because the spec's property table had
until then been asserted only against an implementation the benchmark no longer ran: the
properties passed unchanged, but they had never been checked against the scorer that produces
the published numbers.

One property did not survive the move. P18c asserted `pair_object_keys`, an internal of the
walker; the path scorer has no key-pairing step, because an object key IS the address. It is
noted in place rather than quietly dropped.

Run: python3 tests/test_metric_properties.py
"""
import copy, json, os, random, sys

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench import matching as OM         # noqa: E402
from tests.approximate import approximate                  # noqa: E402
from omni_extract_bench.metric import score as _grade   # noqa: E402
from omni_extract_bench.values import canon_key, cmp_leaf   # noqa: E402


def _schema_from(doc):
    """Derive a schema from gold, so these property tests stay about scoring.

    `score.score` requires one -- without it an `additionalProperties` subtree cannot be
    found, and a fabricated value cannot be told from an invented one. Deriving it from GOLD
    alone is deliberate: a key the PREDICTION invented genuinely is not a slot the model was
    offered.
    """
    if isinstance(doc, dict):
        return {"type": "object", "properties": {k: _schema_from(v) for k, v in doc.items()}}
    if isinstance(doc, list):
        merged = {}
        for e in doc:
            if isinstance(e, dict):
                merged.update(e)
        return {"type": "array", "items": _schema_from(merged) if merged else {}}
    return {}


def fair_grade(pred, gt, schema=None):
    """`score.score` under the name this file was written against."""
    return _grade(pred, gt, schema or _schema_from(gt))

FAILS = []
N_CASES = 120


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(f"{name} {detail}")
    return cond


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if not ok else ""))


WORDS = ["alpha", "beta", "gamma", "Acme Ltd", "J. Smith", "2024-03-01", "Q2", "USD"]


def rand_scalar(rnd):
    return rnd.choice([
        rnd.randint(1, 9999),
        round(rnd.uniform(0, 5000), 2),
        rnd.choice(WORDS),
        rnd.choice([True, False]),
        None,
    ])


def rand_row(rnd, nf=None):
    return {f"f{i}": rand_scalar(rnd) for i in range(nf or rnd.randint(2, 5))}


def rand_doc(rnd, depth=0):
    """Generate a document.

    The array shapes here are load-bearing. This generator used to emit ONLY homogeneous
    arrays of dicts, so three defects in the array branch -- mixed arrays silently dropping
    their scalar elements, arrays of arrays compared by string repr, and shape disagreement
    erasing one side's leaves -- were invisible to every property below. A generator that
    only produces well-shaped input cannot test a scorer whose job is malformed input.
    """
    d = {}
    for i in range(rnd.randint(2, 5)):
        r = rnd.random()
        if r < 0.28 and depth < 2:
            d[f"arr{i}"] = [rand_row(rnd, 4) for _ in range(rnd.randint(1, 6))]
        elif r < 0.38:
            d[f"sarr{i}"] = [rand_scalar(rnd) for _ in range(rnd.randint(1, 5))]
        elif r < 0.46 and depth < 2:
            d[f"marr{i}"] = ([rand_row(rnd, 3) for _ in range(rnd.randint(1, 3))]
                             + [rand_scalar(rnd) for _ in range(rnd.randint(1, 3))])
        elif r < 0.54:
            d[f"aarr{i}"] = [[rand_scalar(rnd) for _ in range(rnd.randint(1, 3))]
                             for _ in range(rnd.randint(1, 4))]
        elif r < 0.66 and depth < 2:
            d[f"obj{i}"] = rand_doc(rnd, depth + 1)
        else:
            d[f"s{i}"] = rand_scalar(rnd)
    return d


def reshape(node, rnd):
    """Perturb a document's SHAPE, not its values: swap a container for another kind.

    This is the mutation no existing test performed. A vendor really does return `{}` where
    gold has `[]`, or a bare string where gold has an array, and the scorer's answer used to
    be to coerce the offending side to an empty container -- discarding the other side's
    leaves rather than charging them.
    """
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if rnd.random() < 0.15:
                out[k] = [] if isinstance(v, dict) else ({} if isinstance(v, list) else "s")
            else:
                out[k] = reshape(v, rnd)
        return out
    if isinstance(node, list):
        return [reshape(x, rnd) for x in node]
    return node


def gold_leaf_paths(node, prefix=()):
    """Every gold leaf path, enumerated WITHOUT consulting the prediction.

    An independent count is the point: it is the one denominator the scorer cannot influence,
    so comparing against it catches erasure that self-referential properties (P11) cannot.
    """
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            out += gold_leaf_paths(v, prefix + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out += gold_leaf_paths(v, prefix + (i,))
    elif node is not None:
        out.append(prefix)
    return out


def leaves_of(node):
    """Every scalar path in a document (mirrors the metric's leaf definition)."""
    out = []

    def w(n, p=()):
        if isinstance(n, dict):
            for k, v in n.items():
                w(v, p + (k,))
        elif isinstance(n, list):
            for i, v in enumerate(n):
                w(v, p + (i,))
        else:
            out.append(p)
    w(node)
    return out


def set_at(doc, path, val):
    cur = doc
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = val


def get_at(doc, path):
    cur = doc
    for p in path:
        cur = cur[p]
    return cur


ok = True
for s in range(N_CASES):
    rnd = random.Random(s)
    g = rand_doc(rnd)
    r = fair_grade(copy.deepcopy(g), copy.deepcopy(g), {})
    if r["total"] and abs(r["accuracy"] - 1.0) > 1e-9:
        ok = check("P1 identity", False, f"seed {s} -> {r['accuracy']}")
        break
report("P1  identity: score(G,G) == 100", ok)

ok = True
for s in range(N_CASES):
    rnd = random.Random(1000 + s)
    g, p = rand_doc(rnd), None
    p = copy.deepcopy(g)
    for k, v in p.items():
        if isinstance(v, list) and v:
            rnd.shuffle(v)
    base = fair_grade(copy.deepcopy(g), copy.deepcopy(g), {})["accuracy"]
    perm = fair_grade(p, copy.deepcopy(g), {})["accuracy"]
    if abs(base - perm) > 1e-9:
        ok = check("P2 permutation", False, f"seed {s}: {base} vs {perm}")
        break
report("P2  permutation invariance: array order is irrelevant", ok)

ok, tested = True, 0
for s in range(N_CASES * 2):
    rnd = random.Random(2000 + s)
    g = rand_doc(rnd)
    p = copy.deepcopy(g)
    paths = [x for x in leaves_of(g) if get_at(g, x) is not None]
    if len(paths) < 3:
        continue
    broken = rnd.sample(paths, min(3, len(paths)))
    for b in broken:
        set_at(p, b, "___WRONG___")
    before = fair_grade(copy.deepcopy(p), copy.deepcopy(g), {})["accuracy"]
    set_at(p, broken[0], get_at(g, broken[0]))
    after = fair_grade(copy.deepcopy(p), copy.deepcopy(g), {})["accuracy"]
    tested += 1
    if after < before - 1e-9:
        ok = check("P3 monotonicity", False, f"seed {s}: repairing a leaf dropped {before}->{after}")
        break
report(f"P3  monotonicity: fixing a leaf never lowers the score  ({tested} cases)", ok)

ok = True
for s in range(N_CASES):
    rnd = random.Random(3000 + s)
    g = rand_doc(rnd)
    r = fair_grade(copy.deepcopy(g), copy.deepcopy(g), {})
    expected = len([x for x in leaves_of(g) if get_at(g, x) is not None])
    if r["total"] != expected:
        ok = check("P4 no double counting", False,
                   f"seed {s}: denominator {r['total']} vs {expected} non-null gold leaves")
        break
report("P4  no double counting: denominator == count of gold leaves", ok)

ok = True
for s in range(60):
    rnd = random.Random(4000 + s)
    g, p = rand_doc(rnd), rand_doc(random.Random(4000 + s + 7))
    a = fair_grade(copy.deepcopy(p), copy.deepcopy(g), {})
    b = fair_grade(copy.deepcopy(p), copy.deepcopy(g), {})
    c = fair_grade(copy.deepcopy(p), copy.deepcopy(g), {})
    if not (a == b == c):
        ok = check("P5 determinism", False, f"seed {s}")
        break
report("P5  determinism: repeated grading is identical", ok)

ok = True
for s in range(400):
    rnd = random.Random(5000 + s)
    a, b = rand_scalar(rnd), rand_scalar(rnd)
    if a is None or b is None:
        continue
    if cmp_leaf(a, b) != cmp_leaf(b, a):
        ok = check("P7 symmetry", False, f"{a!r} vs {b!r}")
        break
report("P7  symmetry: cmp(p,g) == cmp(g,p)", ok)

def brute_force_best(P, G, w):
    """Exhaustive maximum-weight matching, for small n only."""
    best = [0]
    def rec(i, used, tot):
        if i == len(P):
            best[0] = max(best[0], tot); return
        rec(i + 1, used, tot)
        for j in range(len(G)):
            if j in used:
                continue
            rec(i + 1, used | {j}, tot + w(P[i], G[j]))
    rec(0, frozenset(), 0)
    return best[0]


ok, tested = True, 0
for s in range(N_CASES):
    rnd = random.Random(6000 + s)
    n, m = rnd.randint(1, 5), rnd.randint(1, 5)
    P = [rand_row(rnd, 3) for _ in range(n)]
    G = [rand_row(rnd, 3) for _ in range(m)]
    w = lambda a, b: sum(1 for k in set(a) | set(b)
                         if a.get(k) is not None and a.get(k) == b.get(k))
    pairs, _, _, exact = OM.match_rows(P, G, w)
    got = sum(w(a, b) for a, b in pairs)
    best = brute_force_best(P, G, w)
    tested += 1
    if exact and got < best:
        ok = check("P8 optimality", False, f"seed {s}: got {got}, optimum {best}")
        break
report(f"P8  optimality: assignment matches brute-force optimum  ({tested} cases)", ok)

r = fair_grade({"a": None, "b": None}, {"a": None, "b": None}, {})
report("P9  null neutrality: null==null contributes nothing", check(
    "P9", r["total"] == 0, str(r)))

free = all(cmp_leaf(a, b) >= 1.0 for a, b in
           [("(98.2)", "-98.2"), ("−98.2", "-98.2"), ("98.2-", "-98.2"), ("+98.2", "98.2")])
strict = all(cmp_leaf(a, b) < 1.0 for a, b in
             [("-98.2", "98.2"), ("(98.2)", "98.2"), ("-1.0", "1.0")])
report("P10 sign fidelity: notation free, wrong sign never free",
       check("P10", free and strict, f"notation_ok={free} strict_ok={strict}"))


ok = True
for s in range(60):
    rnd = random.Random(9000 + s)
    n = rnd.randint(2, 40)
    keep = rnd.randint(0, n)
    gold = [{"id": i, "v": i * 2} for i in range(n)]
    got = [{"id": i, "v": i * 2} for i in range(keep)]
    flat = fair_grade({"rows": got}, {"rows": gold},
                         {"properties": {"rows": {"type": "array"}}})["accuracy"]
    nested = fair_grade({"w": {"rows": got}}, {"w": {"rows": gold}}, {})["accuracy"]
    if abs(flat - nested) > 0.05:
        ok = check("P11 nesting invariance", False,
                   f"seed {s}: {keep}/{n} rows -> top-level {flat:.1f} vs nested {nested:.1f}")
        break
report("P11 nesting invariance: depth must not change the score", ok)

ok = True
for s in range(60):
    rnd = random.Random(9500 + s)
    n = rnd.randint(4, 40)
    gold = [{"id": i, "v": i} for i in range(n)]
    half = fair_grade({"rows": gold[: n // 2]}, {"rows": gold},
                         {"properties": {"rows": {"type": "array"}}})["accuracy"]
    if half > 99.0:
        ok = check("P12 omission charged", False, f"seed {s}: half of {n} rows scored {half:.1f}")
        break
report("P12 omission is charged, not free", ok)

g = {"tags": ["a", "b", "c", "d"]}
p_ = {"tags": ["a", "WRONG"]}
flat = fair_grade(p_, g, {"properties": {"tags": {"type": "array"}}})
nested = fair_grade({"w": p_}, {"w": g}, {})
report("P13 top-level scalar arrays are scored",
       check("P13", flat["total"] > 0 and flat["total"] == nested["total"],
             f"top denom={flat['total']} nested denom={nested['total']}"))

SCH_ROWS = {"properties": {"rows": {"type": "array"}}}
ok = True
variants = [
    ("date format",   [{"d": "2024-01-15", "v": 5.0}, {"d": "2024-02-20", "v": 5.0}],
                      [{"d": "02/20/2024", "v": 5.0}, {"d": "01/15/2024", "v": 5.0}]),
    ("float precision", [{"k": "a", "v": 33.3333333}, {"k": "a", "v": 66.6666666}],
                        [{"k": "a", "v": 66.66666660001}, {"k": "a", "v": 33.33333330001}]),
    ("sign notation", [{"k": "a", "v": "-98.2"}, {"k": "a", "v": "-15.0"}],
                      [{"k": "a", "v": "(15.0)"}, {"k": "a", "v": "(98.2)"}]),
]
for nm, gold, got in variants:
    sc = fair_grade({"rows": got}, {"rows": gold}, SCH_ROWS)["accuracy"]
    if abs(sc - 1.0) > 1e-6:
        ok = check("P14 pairing/scoring agreement", False, f"{nm} scored {sc:.4f}, want 1.0")
        break
report("P14 pairing honours the same equivalences as scoring", ok)

NUMERIC_FORMS = [5, 5.0, "5", "5.0", "5.00", 0.5, ".5", "0.50", 1000, "1,000", "1000.0",
                 -3, -3.0, "-3", "(3)", 33.3333333, 33.33333330001, 8303911426,
                 "2024-01-15", "01/15/2024", "2024-1-15", True, False, "x", "", None]
ok, pairs_checked = True, 0
for a in NUMERIC_FORMS:
    for b in NUMERIC_FORMS:
        if a is None or b is None:
            continue
        pairs_checked += 1
        if cmp_leaf(a, b) >= 1.0 and canon_key(a) != canon_key(b):
            ok = check("P14b key/comparator agreement", False,
                       f"cmp({a!r},{b!r})=1 but keys differ: "
                       f"{canon_key(a)!r} vs {canon_key(b)!r}")
            break
    if not ok:
        break
if ok:
    for s_ in range(4000):
        rnd = random.Random(20000 + s_)
        a, b = rand_scalar(rnd), rand_scalar(rnd)
        if a is None or b is None:
            continue
        pairs_checked += 1
        if cmp_leaf(a, b) >= 1.0 and canon_key(a) != canon_key(b):
            ok = check("P14b key/comparator agreement", False,
                       f"seed {s_}: cmp({a!r},{b!r})=1 but keys differ")
            break
report(f"P14b equal values always share a pairing key  ({pairs_checked} pairs)", ok)

small = {"rows": [{"i": n} for n in range(20)]}
r_small = fair_grade(small, small, SCH_ROWS)
huge = {"rows": [{"i": n, "v": n % 7} for n in range(40)]}
with approximate():
    r_huge = fair_grade(huge, huge, SCH_ROWS)
SCH_R = {"properties": {"rows": {"type": "array"}}}
gold_b = [{"segment_type": "Company", "v": 10}, {"segment_type": "Segment", "v": 20}]
pred_b = [{"segment_type": "company", "v": 10}, {"segment_type": "segment", "v": 20}]
r_case = fair_grade({"rows": pred_b}, {"rows": gold_b}, SCH_R)
gold_d = [{"segment_type": "company", "v": 10}, {"segment_type": "segment", "v": 99}]
pred_d = [{"segment_type": "company", "v": 10}, {"segment_type": "segment", "v": 20}]
r_diff = fair_grade({"rows": pred_d}, {"rows": gold_d}, SCH_R)
report("P16 blocking uses the scorer's equality, and still separates real differences",
       check("P16", abs(r_case["accuracy"] - 1.0) < 1e-6
             and abs(r_diff["accuracy"] - 0.75) < 1e-6,
             f"casing={r_case['accuracy']:.4f} (want 1.0) "
             f"distinct={r_diff['accuracy']:.1f} (want 75)"))

report("P15 greedy fallback is reported, not silent",
       check("P15", r_small["matching_exact"] is True and r_huge["matching_exact"] is False,
             f"small={r_small['matching_exact']} huge={r_huge['matching_exact']}"))


SCH_P17 = {"properties": {"rows": {"type": "array"}}}
doc_with_empty = {"rows": [
    {"id": 1, "value": 10},
    {"id": None, "value": None},
    {"id": 2, "value": 20},
    {"id": None, "value": None},
]}
r_id = fair_grade(copy.deepcopy(doc_with_empty), copy.deepcopy(doc_with_empty), SCH_P17)
partial = {"rows": [{"id": 1, "value": 10}]}
r_part = fair_grade(partial, copy.deepcopy(doc_with_empty), SCH_P17)
report("P17 identity holds when a document contains all-null rows",
       check("P17", abs(r_id["accuracy"] - 1.0) < 1e-6 and r_part["accuracy"] < 0.99,
             f"identity={r_id['accuracy']:.4f} (want 1.0) partial={r_part['accuracy']:.4f} (want <1.0)"))


SCH_P18 = {"type": "object", "properties": {"total": {"type": "number"}}}
same = fair_grade({"total": 5.0}, {"total": 5.0}, SCH_P18)
cased = fair_grade({"Total": 5.0}, {"total": 5.0}, SCH_P18)
report("P18 an exact key match scores 1.0",
       check("P18", abs(same["accuracy"] - 1.0) < 1e-9, f"got {same['accuracy']:.4f}"))
report("P18b a key that is not exactly gold's is charged, both ways",
       check("P18b", cased["matched"] == 0 and cased["total"] == 2,
             f"got {cased['matched']}/{cased['total']}, want 0/2 "
             f"(1 gold key missing, 1 predicted key spurious)"))

SCH_OM = {"type": "object", "properties": {
    "invoice_no": {"type": "string"},
    "groups": {"type": "object", "additionalProperties": {"type": "array",
                                                          "items": {"type": "string"}}}}}
G_OM = {"invoice_no": "INV-1", "groups": {"Core Competencies": ["SEO"]}}
r_om = fair_grade({"invoice_no": "INV-1"}, copy.deepcopy(G_OM), SCH_OM)
report("P23 an open map reaches neither numerator nor denominator, and is counted",
       check("P23", (r_om["total"], r_om["matched"]) == (1, 1)
             and len(r_om["skipped_open_maps"]) == 1,
             f"got {r_om['matched']}/{r_om['total']}, "
             f"skipped={r_om['skipped_open_maps']}"))

print(f"\n{'ALL METRIC PROPERTIES HOLD' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
sys.exit(1 if FAILS else 0)
