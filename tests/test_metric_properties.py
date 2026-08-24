#!/usr/bin/env python3
"""C3 — property-based tests for the metric spec (METRIC_SPEC.md P1..P10).

Generative, not example-based: random nested documents are built and perturbed, so the
properties are checked over a space of shapes rather than a handful of cases I happened to
think of. Deterministic seeds — a failure is always reproducible.

P3 (monotonicity), P4 (no double counting) and P8 (optimality) are NEW: the benchmark plan
flagged them as unasserted, and optimality had been claimed before it was ever tested.

Run: python3 tests/test_metric_properties.py
"""
import copy, json, os, random, sys

# run from anywhere: `python tests/x.py` puts tests/ on the path, not the repo root
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench import grading as FG          # noqa: E402
from omni_extract_bench import matching as OM         # noqa: E402

FAILS = []
N_CASES = 120


def check(name, cond, detail=""):
    if not cond:
        FAILS.append(f"{name} {detail}")
    return cond


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if not ok else ""))


# ── generators ───────────────────────────────────────────────────────────────────
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
    d = {}
    for i in range(rnd.randint(2, 5)):
        r = rnd.random()
        if r < 0.35 and depth < 2:
            d[f"arr{i}"] = [rand_row(rnd, 4) for _ in range(rnd.randint(1, 6))]
        elif r < 0.5 and depth < 2:
            d[f"obj{i}"] = rand_doc(rnd, depth + 1)
        else:
            d[f"s{i}"] = rand_scalar(rnd)
    return d


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


# ── P1 identity ──────────────────────────────────────────────────────────────────
ok = True
for s in range(N_CASES):
    rnd = random.Random(s)
    g = rand_doc(rnd)
    r = FG.fair_grade(copy.deepcopy(g), copy.deepcopy(g), {})
    if r["leaf_total"] and abs(r["leaf_accuracy"] - 100.0) > 1e-9:
        ok = check("P1 identity", False, f"seed {s} -> {r['leaf_accuracy']}")
        break
report("P1  identity: grade(G,G) == 100", ok)

# ── P2 permutation invariance ────────────────────────────────────────────────────
ok = True
for s in range(N_CASES):
    rnd = random.Random(1000 + s)
    g, p = rand_doc(rnd), None
    p = copy.deepcopy(g)
    for k, v in p.items():                      # perturb a few leaves so it isn't trivially 100
        if isinstance(v, list) and v:
            rnd.shuffle(v)
    base = FG.fair_grade(copy.deepcopy(g), copy.deepcopy(g), {})["leaf_accuracy"]
    perm = FG.fair_grade(p, copy.deepcopy(g), {})["leaf_accuracy"]
    if abs(base - perm) > 1e-9:
        ok = check("P2 permutation", False, f"seed {s}: {base} vs {perm}")
        break
report("P2  permutation invariance: array order is irrelevant", ok)

# ── P3 monotonicity (NEW) ────────────────────────────────────────────────────────
ok, tested = True, 0
for s in range(N_CASES * 2):
    rnd = random.Random(2000 + s)
    g = rand_doc(rnd)
    p = copy.deepcopy(g)
    paths = [x for x in leaves_of(g) if get_at(g, x) is not None]
    if len(paths) < 3:
        continue
    # break several leaves, then repair ONE and require the score not to fall
    broken = rnd.sample(paths, min(3, len(paths)))
    for b in broken:
        set_at(p, b, "___WRONG___")
    before = FG.fair_grade(copy.deepcopy(p), copy.deepcopy(g), {})["leaf_accuracy"]
    set_at(p, broken[0], get_at(g, broken[0]))          # repair one
    after = FG.fair_grade(copy.deepcopy(p), copy.deepcopy(g), {})["leaf_accuracy"]
    tested += 1
    if after < before - 1e-9:
        ok = check("P3 monotonicity", False, f"seed {s}: repairing a leaf dropped {before}->{after}")
        break
report(f"P3  monotonicity: fixing a leaf never lowers the score  ({tested} cases)", ok)

# ── P4 no double counting (NEW) ──────────────────────────────────────────────────
ok = True
for s in range(N_CASES):
    rnd = random.Random(3000 + s)
    g = rand_doc(rnd)
    r = FG.fair_grade(copy.deepcopy(g), copy.deepcopy(g), {})
    expected = len([x for x in leaves_of(g) if get_at(g, x) is not None])
    if r["leaf_total"] != expected:
        ok = check("P4 no double counting", False,
                   f"seed {s}: denominator {r['leaf_total']} vs {expected} non-null gold leaves")
        break
report("P4  no double counting: denominator == count of gold leaves", ok)

# ── P5 determinism ───────────────────────────────────────────────────────────────
ok = True
for s in range(60):
    rnd = random.Random(4000 + s)
    g, p = rand_doc(rnd), rand_doc(random.Random(4000 + s + 7))
    a = FG.fair_grade(copy.deepcopy(p), copy.deepcopy(g), {})
    b = FG.fair_grade(copy.deepcopy(p), copy.deepcopy(g), {})
    c = FG.fair_grade(copy.deepcopy(p), copy.deepcopy(g), {})
    if not (a == b == c):
        ok = check("P5 determinism", False, f"seed {s}")
        break
report("P5  determinism: repeated grading is identical", ok)

# ── P7 symmetry of canonicalisation ──────────────────────────────────────────────
ok = True
for s in range(400):
    rnd = random.Random(5000 + s)
    a, b = rand_scalar(rnd), rand_scalar(rnd)
    if a is None or b is None:
        continue
    if FG.cmp_leaf(a, b) != FG.cmp_leaf(b, a):
        ok = check("P7 symmetry", False, f"{a!r} vs {b!r}")
        break
report("P7  symmetry: cmp(p,g) == cmp(g,p)", ok)

# ── P8 optimality of row assignment (NEW) ────────────────────────────────────────
def brute_force_best(P, G, w):
    """Exhaustive maximum-weight matching, for small n only."""
    best = [0]
    def rec(i, used, tot):
        if i == len(P):
            best[0] = max(best[0], tot); return
        rec(i + 1, used, tot)                      # leave Pi unpaired
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

# ── P9 null neutrality ───────────────────────────────────────────────────────────
r = FG.fair_grade({"a": None, "b": None}, {"a": None, "b": None}, {})
report("P9  null neutrality: null==null contributes nothing", check(
    "P9", r["leaf_total"] == 0, str(r)))

# ── P10 sign fidelity ────────────────────────────────────────────────────────────
free = all(FG.cmp_leaf(a, b) >= 1.0 for a, b in
           [("(98.2)", "-98.2"), ("−98.2", "-98.2"), ("98.2-", "-98.2"), ("+98.2", "98.2")])
strict = all(FG.cmp_leaf(a, b) < 1.0 for a, b in
             [("-98.2", "98.2"), ("(98.2)", "98.2"), ("-1.0", "1.0")])
report("P10 sign fidelity: notation free, wrong sign never free",
       check("P10", free and strict, f"notation_ok={free} strict_ok={strict}"))


# ── P11 nesting invariance (REGRESSION: the 349-row corruption) ──────────────────
# A document that scored 100.0 while returning 44 of 349 rows. Omission was free for
# TOP-LEVEL arrays (only matched pairs entered the denominator) but charged when the same
# data sat one level down. Any A/B run through this metric was affected, always in the
# direction of flattering an extractor that omits data.
ok = True
for s in range(60):
    rnd = random.Random(9000 + s)
    n = rnd.randint(2, 40)
    keep = rnd.randint(0, n)
    gold = [{"id": i, "v": i * 2} for i in range(n)]
    got = [{"id": i, "v": i * 2} for i in range(keep)]
    flat = FG.fair_grade({"rows": got}, {"rows": gold},
                         {"properties": {"rows": {"type": "array"}}})["leaf_accuracy"]
    nested = FG.fair_grade({"w": {"rows": got}}, {"w": {"rows": gold}}, {})["leaf_accuracy"]
    if abs(flat - nested) > 0.05:
        ok = check("P11 nesting invariance", False,
                   f"seed {s}: {keep}/{n} rows -> top-level {flat:.1f} vs nested {nested:.1f}")
        break
report("P11 nesting invariance: depth must not change the score", ok)

# ── P12 omission is charged ─────────────────────────────────────────────────────
ok = True
for s in range(60):
    rnd = random.Random(9500 + s)
    n = rnd.randint(4, 40)
    gold = [{"id": i, "v": i} for i in range(n)]
    half = FG.fair_grade({"rows": gold[: n // 2]}, {"rows": gold},
                         {"properties": {"rows": {"type": "array"}}})["leaf_accuracy"]
    if half > 99.0:     # returning half the rows must NOT look perfect
        ok = check("P12 omission charged", False, f"seed {s}: half of {n} rows scored {half:.1f}")
        break
report("P12 omission is charged, not free", ok)

# ── P13 scalar arrays are scored at every depth ─────────────────────────────────
g = {"tags": ["a", "b", "c", "d"]}
p_ = {"tags": ["a", "WRONG"]}
flat = FG.fair_grade(p_, g, {"properties": {"tags": {"type": "array"}}})
nested = FG.fair_grade({"w": p_}, {"w": g}, {})
report("P13 top-level scalar arrays are scored",
       check("P13", flat["leaf_total"] > 0 and flat["leaf_total"] == nested["leaf_total"],
             f"top denom={flat['leaf_total']} nested denom={nested['leaf_total']}"))

# ── P14 pairing agrees with scoring (REGRESSION: format-penalty via mis-pairing) ──
# The assignment weight used bare canonical equality while scoring is format-tolerant. When
# the DISCRIMINATING field between rows was a date, a vendor emitting `02/20/2024` against
# gold `2024-02-20` scored 0 on it during matching, the rows tied on everything else, and the
# matcher paired them arbitrarily -- a fully correct extraction scored 50.0. Any equivalence
# cmp_leaf honours must also be honoured when CHOOSING the pairing, or the benchmark
# reintroduces exactly the format penalty it exists to remove.
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
    sc = FG.fair_grade({"rows": got}, {"rows": gold}, SCH_ROWS)["leaf_accuracy"]
    if abs(sc - 100.0) > 1e-6:
        ok = check("P14 pairing/scoring agreement", False, f"{nm} scored {sc:.1f}, want 100.0")
        break
report("P14 pairing honours the same equivalences as scoring", ok)

# P14b: the general invariant behind P14, fuzzed rather than enumerated.
#   cmp_leaf(a, b) == 1  =>  _match_canon(a) == _match_canon(b)
# If two values score as equal but hash to different pairing keys, rows distinguished by that
# field cannot pair and a correct extraction is scored as a miss. The first version of
# _match_canon violated exactly this for integers vs decimals (`5` vs `5.0`): _asfloat only
# parses values containing ".", so the two took different branches, and predictions whose
# numeric fields were the discriminator matched NO gold row at all -- one provider's recall
# went to 0.000 on every document in two subsets.
NUMERIC_FORMS = [5, 5.0, "5", "5.0", "5.00", 0.5, ".5", "0.50", 1000, "1,000", "1000.0",
                 -3, -3.0, "-3", "(3)", 33.3333333, 33.33333330001, 8303911426,
                 "2024-01-15", "01/15/2024", "2024-1-15", True, False, "x", "", None]
ok, pairs_checked = True, 0
for a in NUMERIC_FORMS:
    for b in NUMERIC_FORMS:
        if a is None or b is None:
            continue
        pairs_checked += 1
        if FG.cmp_leaf(a, b) >= 1.0 and FG.canon_key(a) != FG.canon_key(b):
            ok = check("P14b key/comparator agreement", False,
                       f"cmp({a!r},{b!r})=1 but keys differ: "
                       f"{FG.canon_key(a)!r} vs {FG.canon_key(b)!r}")
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
        if FG.cmp_leaf(a, b) >= 1.0 and FG.canon_key(a) != FG.canon_key(b):
            ok = check("P14b key/comparator agreement", False,
                       f"seed {s_}: cmp({a!r},{b!r})=1 but keys differ")
            break
report(f"P14b equal values always share a pairing key  ({pairs_checked} pairs)", ok)

# ── P15 greedy fallback is never silent ─────────────────────────────────────────
# Arrays too large to solve exactly fall back to greedy, which is suboptimal and under-scores
# providers that return big tables. The fallback is unavoidable (largest gold array is 6881
# rows, solver is O(n^3)); reporting a greedy score AS IF exact is not.
small = {"rows": [{"i": n} for n in range(20)]}
r_small = FG.fair_grade(small, small, SCH_ROWS)
huge = {"rows": [{"i": n, "v": n % 7} for n in range(OM.MAX_EXACT + 200)]}
r_huge = FG.fair_grade(huge, huge, SCH_ROWS)
# ── P16 one equality everywhere (REGRESSION: blocking used a third definition) ───
# Blocking asserts "these rows can never pair". If it uses a STRICTER notion of equality than
# the scorer, it silently forbids pairings the scorer would accept. It used bare str(), so rows
# differing only in the CASING of the blocking field could never match: leaf 0.0, recall 0.00,
# on content the comparator calls identical. Blocking is active on ~46% of benchmark arrays.
SCH_R = {"properties": {"rows": {"type": "array"}}}
gold_b = [{"segment_type": "Company", "v": 10}, {"segment_type": "Segment", "v": 20}]
pred_b = [{"segment_type": "company", "v": 10}, {"segment_type": "segment", "v": 20}]
r_case = FG.fair_grade({"rows": pred_b}, {"rows": gold_b}, SCH_R)
# ...but blocking must still SEPARATE genuinely different segments, or it would over-merge.
gold_d = [{"segment_type": "company", "v": 10}, {"segment_type": "segment", "v": 99}]
pred_d = [{"segment_type": "company", "v": 10}, {"segment_type": "segment", "v": 20}]
r_diff = FG.fair_grade({"rows": pred_d}, {"rows": gold_d}, SCH_R)
report("P16 blocking uses the scorer's equality, and still separates real differences",
       check("P16", abs(r_case["leaf_accuracy"] - 100.0) < 1e-6
             and abs(r_diff["leaf_accuracy"] - 75.0) < 1e-6,
             f"casing={r_case['leaf_accuracy']:.1f} (want 100) "
             f"distinct={r_diff['leaf_accuracy']:.1f} (want 75)"))

report("P15 greedy fallback is reported, not silent",
       check("P15", r_small["matching_exact"] is True and r_huge["matching_exact"] is False,
             f"small={r_small['matching_exact']} huge={r_huge['matching_exact']}"))


# ── P17 identity holds for documents containing rows that assert nothing ────────
# (REGRESSION) Rows whose payload is entirely null are ignored for fairness -- a provider
# should not be scored on a row that states nothing. That filter was applied to GROUND TRUTH
# ONLY, so a prediction mirroring ground truth exactly still carried the rows ground truth had
# just discarded; they counted as spurious and grade(gt, gt) came out at 95.09 on a real 10-Q.
# The most faithful possible extraction was penalised. P1 could not catch it because the
# generator never produces an all-null row -- so this builds one explicitly.
SCH_P17 = {"properties": {"rows": {"type": "array"}}}
doc_with_empty = {"rows": [
    {"id": 1, "value": 10},
    {"id": None, "value": None},          # asserts nothing
    {"id": 2, "value": 20},
    {"id": None, "value": None},          # asserts nothing
]}
r_id = FG.fair_grade(copy.deepcopy(doc_with_empty), copy.deepcopy(doc_with_empty), SCH_P17)
# and a partial answer must still be charged, i.e. the filter must not make omission free
partial = {"rows": [{"id": 1, "value": 10}]}
r_part = FG.fair_grade(partial, copy.deepcopy(doc_with_empty), SCH_P17)
report("P17 identity holds when a document contains all-null rows",
       check("P17", abs(r_id["leaf_accuracy"] - 100.0) < 1e-6 and r_part["leaf_accuracy"] < 99.0,
             f"identity={r_id['leaf_accuracy']:.2f} (want 100) partial={r_part['leaf_accuracy']:.2f} (want <100)"))


# ── P18 an object key is a value: casing must not decide the score ──────────────
# Object keys were compared LITERALLY while every other value went through canon_key -- a
# second definition of equality, in the one place a document (not the schema) supplies the
# string. Ground truth is not reliably verbatim about case, so grading it penalised the
# extractor that transcribed the document more faithfully than gold did.
SCH_P18 = {"type": "object", "properties": {
    "groups": {"type": "object", "additionalProperties": {"type": "array",
                                                          "items": {"type": "string"}}}}}
gold_p18 = {"groups": {"Core Competencies": ["Brand Strategy", "SEO"], "Other": ["Slack"]}}
same_but_caps = {"groups": {"CORE COMPETENCIES": ["Brand Strategy", "SEO"], "Other": ["Slack"]}}
r_caps = FG.fair_grade(same_but_caps, copy.deepcopy(gold_p18), SCH_P18)
report("P18 a key differing only in case scores identically",
       check("P18", abs(r_caps["leaf_accuracy"] - 100.0) < 1e-6,
             f"got {r_caps['leaf_accuracy']:.2f}, want 100"))

# and the relaxation must not make WRONG values free
wrong_p18 = {"groups": {"CORE COMPETENCIES": ["Nonsense", "SEO"], "Other": ["Slack"]}}
r_wrong = FG.fair_grade(wrong_p18, copy.deepcopy(gold_p18), SCH_P18)
report("P18b folding keys does not excuse wrong values",
       check("P18b", r_wrong["leaf_accuracy"] < 99.0,
             f"got {r_wrong['leaf_accuracy']:.2f}, want <100"))

# COLLISION: two keys of one object that canonicalise together must NOT be merged, because
# merging silently discards one side's value -- worse than the problem being fixed.
pairs = FG.pair_object_keys({"Total": 1, "TOTAL": 2}, {"Total": 1})
report("P18c colliding keys fall back to literal pairing",
       check("P18c", len(pairs) == 2 and ("TOTAL", None) in pairs,
             f"got {pairs}"))

print(f"\n{'ALL METRIC PROPERTIES HOLD' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
sys.exit(1 if FAILS else 0)
