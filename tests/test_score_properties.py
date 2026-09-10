#!/usr/bin/env python3
"""What must be true of the path-based scorer (`score`), stated as properties.

The equivalence harness (`test_paths_equivalence.py`) asks "does this agree with the scorer
it replaces, and where it disagrees, why". This file asks the independent question: is the
path scorer correct on its own terms? A rewrite that merely reproduces its predecessor
inherits its predecessor's bugs -- three denominator defects and an order-sensitivity bug
were found in that predecessor while this module was being written -- so agreement is
evidence, not proof.

Every check below names the property in words. Run: python3 tests/test_paths_properties.py
"""
import copy
import itertools
import os as _os
import random
import sys as _sys
import time
import tracemalloc

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench import matching as OM                           # noqa: E402
from omni_extract_bench.grading import cmp_leaf, fair_grade, canon_key            # noqa: E402
from omni_extract_bench.score import (                          # noqa: E402
    node_key, format_node, _find_arrays, KEY, INDEX,
    grade, flatten, align, explain)

FAILS = []
WORDS = ["alpha", "beta", "gamma", "Acme Ltd", "J. Smith", "2024-03-01", "Q2", "USD"]


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


def rand_scalar(rnd):
    return rnd.choice([rnd.randint(1, 9999), round(rnd.uniform(0, 5000), 2),
                       rnd.choice(WORDS), rnd.choice([True, False]), None])


def rand_row(rnd, nf=None):
    return {f"f{i}": rand_scalar(rnd) for i in range(nf or rnd.randint(2, 5))}


def rand_doc(rnd, depth=0):
    d = {}
    for i in range(rnd.randint(2, 5)):
        r = rnd.random()
        if r < 0.24 and depth < 2:
            d[f"arr{i}"] = [rand_row(rnd, 4) for _ in range(rnd.randint(1, 6))]
        elif r < 0.34:
            d[f"sarr{i}"] = [rand_scalar(rnd) for _ in range(rnd.randint(1, 5))]
        elif r < 0.42 and depth < 2:
            d[f"marr{i}"] = ([rand_row(rnd, 3) for _ in range(rnd.randint(1, 3))]
                             + [rand_scalar(rnd) for _ in range(rnd.randint(1, 3))])
        elif r < 0.50:
            d[f"aarr{i}"] = [[rand_scalar(rnd) for _ in range(rnd.randint(1, 3))]
                             for _ in range(rnd.randint(1, 4))]
        elif r < 0.62 and depth < 2:
            d[f"obj{i}"] = rand_doc(rnd, depth + 1)
        else:
            d[f"s{i}"] = rand_scalar(rnd)
    return d


ACC = lambda p, g: grade(copy.deepcopy(p), copy.deepcopy(g), {})   # noqa: E731

# ═══════════════════════════════════════════════════════════════════════════════
print("\nNO PAIRING OF ROWS SCORES HIGHER THAN THE ONE CHOSEN")
# The pairing weight was replaced with a masked multiset when this module was written, which
# is a different objective from the dict intersection it replaced. If masking merges two
# things it should not, the solver optimises the wrong quantity and is still "optimal" -- for
# the wrong problem. Brute force is the only way to know.
def brute_force_matched(pred_rows, gold_rows):
    """Most matched leaves achievable by ANY pairing of these flat rows."""
    P = [flatten(r) for r in pred_rows]
    G = [flatten(r) for r in gold_rows]
    best = 0
    for k in range(min(len(P), len(G)) + 1):
        for psel in itertools.combinations(range(len(P)), k):
            for gsel in itertools.permutations(range(len(G)), k):
                best = max(best, sum(
                    sum(1 for key, v in P[pi].items()
                        if key in G[gi] and canon_key(v) == canon_key(G[gi][key]))
                    for pi, gi in zip(psel, gsel)))
    return best

ok, worst = True, None
for s in range(150):
    rnd = random.Random(9000 + s)
    vals = lambda: {f"f{j}": rnd.choice(["A", "B", "C", 1, 2.5]) for j in range(3)}
    gold = [vals() for _ in range(rnd.randint(2, 4))]
    pred = [vals() for _ in range(rnd.randint(2, 4))]
    got = ACC({"rows": pred}, {"rows": gold})["matched"]
    best = brute_force_matched(pred, gold)
    if got != best:
        ok = False
        worst = f"seed {s}: chose {got} matched leaves, brute force finds {best}"
        break
report("the chosen pairing maximises matched leaves (checked against brute force)", ok, worst)

# ═══════════════════════════════════════════════════════════════════════════════
print("\nDEPTH DOES NOT CHANGE THE SCORE")
# If wrapping a document in extra levels can change its score, some scoring path depends on
# depth, and the whole class of depth-dependent bugs is live. Masking makes deeper array
# indices wildcards, so depth is exactly what this module touches.
ok, detail = True, None
for s in range(200):
    rnd = random.Random(9200 + s)
    gold = rand_doc(rnd)
    pred = rand_doc(random.Random(9200 + s))          # same shape, then corrupt
    pred = copy.deepcopy(gold)
    if rnd.random() < 0.7:
        k = sorted(pred)[0]
        pred[k] = rand_scalar(rnd)
    flat = ACC(pred, gold)
    for wrap in (1, 2):
        w_p, w_g = copy.deepcopy(pred), copy.deepcopy(gold)
        for _ in range(wrap):
            w_p, w_g = {"w": w_p}, {"w": w_g}
        deep = ACC(w_p, w_g)
        if (abs(flat["accuracy"] - deep["accuracy"]) > 1e-9
                or flat["total"] != deep["total"]):
            ok = False
            detail = (f"seed {s} wrap {wrap}: flat {flat['accuracy']:.4f}/"
                      f"{flat['total']} vs nested {deep['accuracy']:.4f}/"
                      f"{deep['total']}")
            break
    if not ok:
        break
report("wrapping a document in extra levels changes neither score nor denominator", ok, detail)

# ═══════════════════════════════════════════════════════════════════════════════
print("\nARRAY ORDER IS IRRELEVANT, AT EVERY DEPTH")
ok, detail = True, None
for s in range(200):
    rnd = random.Random(9400 + s)
    gold = rand_doc(rnd)
    if not flatten(gold):
        continue

    def shuffle(n):
        if isinstance(n, dict):
            return {k: shuffle(v) for k, v in n.items()}
        if isinstance(n, list):
            out = [shuffle(x) for x in n]
            rnd.shuffle(out)
            return out
        return n
    r = ACC(shuffle(copy.deepcopy(gold)), gold)
    if abs(r["accuracy"] - 100) > 1e-9:
        ok, detail = False, f"seed {s}: reordering a correct answer scored {r['accuracy']:.4f}"
        break
report("reordering every array in a correct answer still scores 100", ok, detail)

# rows whose only distinguishing content is NESTED -- the case the paired walker fails
bad_path = bad_tree = 0
for s in range(200):
    rnd = random.Random(9600 + s)
    rows = [{"tag": "same", "d": {f"d{i}": rand_scalar(rnd) for i in range(3)}}
            for _ in range(rnd.randint(2, 4))]
    gold, perm = {"rows": rows}, {"rows": random.Random(s + 1).sample(rows, len(rows))}
    bad_path += ACC(perm, gold)["accuracy"] < 99.99
    bad_tree += fair_grade(copy.deepcopy(perm), copy.deepcopy(gold), {})["leaf_accuracy"] < 99.99
report("rows distinguished only by nested content are still order-free", bad_path == 0,
       f"failed {bad_path}/200")
note(f"for comparison, grading.fair_grade fails this on {bad_tree}/200")

# ═══════════════════════════════════════════════════════════════════════════════
print("\nFIXING A WRONG VALUE NEVER LOWERS THE SCORE")
# Correcting a leaf can change which rows pair, and therefore which addresses exist. If the
# re-pairing ever costs more than the correction gains, the metric punishes improvement.
ok, detail = True, None
for s in range(300):
    rnd = random.Random(9800 + s)
    gold = [{f"f{j}": rnd.choice(["A", "B", "C", 1, 2.5]) for j in range(4)}
            for _ in range(rnd.randint(2, 4))]
    pred = copy.deepcopy(gold)
    wrong = []
    for i, row in enumerate(pred):
        for f in row:
            if rnd.random() < 0.4:
                row[f] = "CORRUPT"
                wrong.append((i, f))
    if not wrong:
        continue
    before = ACC({"rows": pred}, {"rows": gold})["accuracy"]
    for i, f in wrong:
        fixed = copy.deepcopy(pred)
        fixed[i][f] = gold[i][f]
        after = ACC({"rows": fixed}, {"rows": gold})["accuracy"]
        if after < before - 1e-9:
            ok = False
            detail = f"seed {s}: fixing row {i} field {f} took {before:.4f} -> {after:.4f}"
            break
    if not ok:
        break
report("correcting one wrong leaf never lowers the score", ok, detail)

# ═══════════════════════════════════════════════════════════════════════════════
print("\nEVERY GOLD LEAF COSTS EXACTLY ONE DENOMINATOR SLOT")
ok, detail = True, None
for s in range(300):
    rnd = random.Random(10000 + s)
    gold = rand_doc(rnd)

    def subset(n):
        """A prediction that only ever DROPS gold content, never adds."""
        if isinstance(n, dict):
            return {k: subset(v) for k, v in n.items() if rnd.random() > 0.3}
        if isinstance(n, list):
            return [subset(x) for x in n if rnd.random() > 0.3]
        return n
    pred = subset(copy.deepcopy(gold))
    r = ACC(pred, gold)
    if r["total"] != len(flatten(gold)):
        ok = False
        detail = (f"seed {s}: denominator {r['total']} != gold leaves "
                  f"{len(flatten(gold))} for a prediction that only omits")
        break
report("a prediction that only omits has denominator == gold's leaf count", ok, detail)

ok = True
for s in range(200):
    rnd = random.Random(10200 + s)
    n = rnd.randint(4, 20)
    gold = [{"id": i, "v": i * 1.5} for i in range(n)]
    if ACC({"rows": gold[: n // 2]}, {"rows": gold})["accuracy"] > 99.0:
        ok = False
        break
report("returning only some of the gold rows never scores 100", ok)

# ═══════════════════════════════════════════════════════════════════════════════
print("\nNULL ASSERTS NOTHING")
g_null = {"a": 1, "b": None, "rows": [{"x": 5, "y": None}, {"x": None, "y": None}]}
r_null = ACC(copy.deepcopy(g_null), g_null)
r_bare = ACC({"a": 1, "rows": [{"x": 5}]}, {"a": 1, "rows": [{"x": 5}]})
report("null on both sides adds nothing to numerator or denominator",
       r_null["total"] == r_bare["total"] and abs(r_null["accuracy"] - 100) < 1e-9,
       f"with nulls {r_null['total']} slots, without {r_bare['total']}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\nTHE SAME INPUT ALWAYS PRODUCES THE SAME OUTPUT")
ok, detail = True, None
for s in range(200):
    rnd = random.Random(10400 + s)
    gold, pred = rand_doc(rnd), rand_doc(random.Random(10400 + s + 1))
    runs = [ACC(pred, gold) for _ in range(3)]
    # and again with the prediction's keys inserted in a different order
    shuffled = dict(sorted(pred.items(), reverse=True))
    runs.append(ACC(shuffled, gold))
    if len({(r["total"], r["matched"]) for r in runs}) != 1:
        ok, detail = False, f"seed {s}: {[(r['total'], r['matched']) for r in runs]}"
        break
report("repeated grading, and re-ordered object keys, give identical results", ok, detail)

# ═══════════════════════════════════════════════════════════════════════════════
print("\nAN OBJECT KEY IS NEVER CONFUSED WITH AN ARRAY INDEX")
# A document may legally contain the key "0". Untagged paths would let ("m", 0) mean both
# `m["0"]` and `m[0]`, silently joining an object leaf to an array leaf.
g_key = {"m": {"0": "A", "1": "B"}}
p_arr = {"m": ["A", "B"]}
r = ACC(p_arr, g_key)
report("an object keyed \"0\" does not join an array's element 0",
       r["matched"] == 0 and r["total"] == 4,
       f"matched {r['matched']} (want 0), denominator {r['total']} (want 4: 2 gold + 2 spurious)")
report("a key literally named \"*\" does not collide with a masked index",
       abs(ACC({"m": {"*": [1, 2]}}, {"m": {"*": [1, 2]}})["accuracy"] - 100) < 1e-9)

# ═══════════════════════════════════════════════════════════════════════════════
print("\nA FABRICATED ROW GETS ITS OWN ADDRESS, NEVER A REAL ONE")
gold_rows = [{"a": "x", "b": 1.0}, {"a": "y", "b": 2.0}]
none_pair = [{"a": "q", "b": 9.0}, {"a": "r", "b": 8.0}, {"a": "s", "b": 7.0}]
r = ACC({"rows": none_pair}, {"rows": gold_rows})
report("when no predicted row pairs, gold and predicted leaves are all charged",
       r["matched"] == 0 and r["total"] == 10,
       f"matched {r['matched']} (want 0), denominator {r['total']} (want 4 gold + 6 spurious = 10)")
addrs = {k[1][1] for k, _, _, _ in explain({"rows": none_pair}, {"rows": gold_rows})}
report("fabricated rows occupy addresses disjoint from gold's indices",
       addrs == {0, 1, "p0", "p1", "p2"}, f"addresses {sorted(addrs, key=str)}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\nALIGNING AN ALREADY-ALIGNED PREDICTION CHANGES NOTHING")
ok, detail = True, None
for s in range(200):
    rnd = random.Random(10600 + s)
    gold = rand_doc(rnd)
    pred = rand_doc(random.Random(10600 + s + 1))
    gp = flatten(gold)
    once, _ = align(gp, flatten(pred))
    twice, _ = align(gp, dict(once))
    if once != twice:
        ok, detail = False, f"seed {s}: {len(set(once) ^ set(twice))} addresses moved on re-alignment"
        break
report("alignment is idempotent: addresses are stable once assigned", ok, detail)

# ═══════════════════════════════════════════════════════════════════════════════
print("\nDEGENERATE SHAPES TERMINATE AND SCORE SENSIBLY")
DEGENERATE = [
    ({}, {}), ({"a": []}, {"a": []}), ({"a": {}}, {"a": {}}),
    ({"a": [[]]}, {"a": [[]]}), ({"a": [{}]}, {"a": [{}]}),
    ({"a": [[], {}, None, 1]}, {"a": [[], {}, None, 1]}),
    ({"a": []}, {"a": [{"x": 1}]}), ({"a": [{"x": 1}]}, {"a": []}),
    ({}, {"a": [{"x": 1}]}), ({"a": [{"x": 1}]}, {}),
    ({"a": [[[[1]]]]}, {"a": [[[[1]]]]}),
    ({"a": {"b": {"c": []}}}, {"a": {"b": {"c": []}}}),
]
ok, detail = True, None
t0 = time.time()
for pred, gold in DEGENERATE:
    try:
        r = ACC(pred, gold)
        if not (0.0 <= r["accuracy"] <= 100.0) or r["total"] < 0:
            ok, detail = False, f"{pred} vs {gold} -> {r}"
            break
    except Exception as exc:                                    # noqa: BLE001
        ok, detail = False, f"{pred} vs {gold} raised {type(exc).__name__}: {exc}"
        break
report(f"empty and degenerate containers do not crash or hang ({time.time()-t0:.2f}s)", ok, detail)

# identity must still hold for every degenerate shape that has leaves at all
ok = all(abs(ACC(copy.deepcopy(g), copy.deepcopy(g))["accuracy"] - 100) < 1e-9
         for _, g in DEGENERATE if flatten(g))
report("identity holds for degenerate shapes that contain any leaf", ok)

# ═══════════════════════════════════════════════════════════════════════════════
print("\nNESTED ARRAYS ARE ORDER-FREE, AND THEIR WEIGHT IS NEVER OVER-COUNTED")
deep_gold = {"cube": [[[1.5, 2.5], [3.5, 4.5]], [[5.5, 6.5], [7.5, 8.5]]]}


def shuffle_all(n, rnd):
    if isinstance(n, list):
        out = [shuffle_all(x, rnd) for x in n]
        rnd.shuffle(out)
        return out
    return n


ok = all(abs(ACC({"cube": shuffle_all(copy.deepcopy(deep_gold["cube"]), random.Random(s_))},
                 deep_gold)["accuracy"] - 100) < 1e-9 for s_ in range(20))
report("a three-deep array reordered at every level still scores 100", ok)
bad = copy.deepcopy(deep_gold)
bad["cube"][0][0][0] = 99.9
r = ACC(bad, deep_gold)
report("one wrong cell in a three-deep array costs one cell, not a whole sub-array",
       r["total"] == 9 and r["matched"] == 7,
       f"denominator {r['total']} (want 8 gold + 1 spurious = 9), matched {r['matched']} (want 7)")

# The pairing weight must never claim more matches than a pairing could actually deliver.
# An over-estimate can make the solver prefer a worse pairing; a lower bound can only fail to
# break a tie. Checked by brute force over rows that contain nested arrays.
from omni_extract_bench.score import _extract_rows_at, _worth_if_paired  # noqa: E402


def true_max_matched(a, b):
    """Most leaves two elements could share, over every pairing of their nested elements."""
    fa, fb = flatten(a), flatten(b)
    direct = sum(1 for k, v in fa.items()
                 if not any(st[0] == "i" for st in k) and k in fb
                 and canon_key(v) == canon_key(fb[k]))
    best = direct
    for key in set(a) | set(b):
        xa, xb = a.get(key), b.get(key)
        if not (isinstance(xa, list) and isinstance(xb, list)):
            continue
        sub = 0
        for k in range(min(len(xa), len(xb)) + 1):
            for psel in itertools.combinations(range(len(xa)), k):
                for gsel in itertools.permutations(range(len(xb)), k):
                    sub = max(sub, sum(
                        sum(1 for kk, vv in flatten(xa[i]).items()
                            if kk in flatten(xb[j])
                            and canon_key(vv) == canon_key(flatten(xb[j])[kk]))
                        for i, j in zip(psel, gsel)))
        best += sub
    return best


ok, detail = True, None
for s_ in range(120):
    rnd = random.Random(11000 + s_)
    mk = lambda: {"tag": rnd.choice(["A", "B"]),
                  "xs": [{f"k{j}": rnd.choice([1, 2, 3]) for j in range(rnd.randint(1, 2))}
                         for _ in range(rnd.randint(1, 3))]}
    a, b = mk(), mk()
    got, _shared = _worth_if_paired(_extract_rows_at(flatten({"r": [a]}), ((KEY, "r"),), frozenset())[0],
                          _extract_rows_at(flatten({"r": [b]}), ((KEY, "r"),), frozenset())[0], 999)
    cap = true_max_matched(a, b)
    if got > cap:
        ok, detail = False, f"seed {s_}: weight {got} exceeds achievable {cap}\n          a={a}\n          b={b}"
        break
report("the pairing weight never claims more matches than a pairing can deliver", ok, detail)

print("\nA NESTED ARRAY'S ORDER CANNOT CHANGE THE OUTER PAIRING")
# The weight that ranks candidate OUTER pairings includes the contribution of arrays nested
# inside the elements. If that contribution depends on the order of the nested array, the
# outer pairing does too -- and array order must never reach a score. Greedy inner matching
# had exactly that flaw: gold [{a:1},{b:2}] against [{a:1,b:2},{a:1}] weighed (1,1) in one
# inner order and (2,2) in the other. It is solved exactly now, and a maximum has no such
# freedom.
from omni_extract_bench.score import _extract_rows_at, _worth_if_paired   # noqa: E402


def row_weight(row_p, row_g):
    return _worth_if_paired(_extract_rows_at(flatten({"r": [row_p]}), ((KEY, "r"),), frozenset())[0],
                  _extract_rows_at(flatten({"r": [row_g]}), ((KEY, "r"),), frozenset())[0], 999)


G_INNER = {"xs": [{"a": 1}, {"b": 2}]}
P_INNER = {"xs": [{"a": 1, "b": 2}, {"a": 1}]}
weights = {row_weight({"xs": [P_INNER["xs"][i] for i in perm]}, G_INNER)
           for perm in ((0, 1), (1, 0))}
report("the outer weight is the same whichever order a nested array arrives in",
       len(weights) == 1, f"got {sorted(weights)} across the two inner orders")
report("and it is the true optimum, not a greedy lower bound",
       weights == {(2, 2)}, f"got {sorted(weights)}, want {{(2, 2)}}")

# the same thing over generated documents, with SEVERAL gold rows so the weight actually
# decides the outer pairing rather than being the only option
moved = 0
for s_ in range(400):
    rnd = random.Random(11500 + s_)
    mk = lambda: {f"k{j}": rnd.choice([1, 2, 3]) for j in range(rnd.randint(1, 2))}
    gold = {"rows": [{"tag": t, "xs": [mk() for _ in range(rnd.randint(2, 4))]}
                     for t in ("T", "T", "U")]}
    prow = {"tag": "T", "xs": [mk() for _ in range(rnd.randint(2, 4))]}
    other = {"tag": "T", "xs": [mk() for _ in range(2)]}
    scores = set()
    for perm in itertools.permutations(range(len(prow["xs"]))):
        pp = {"tag": "T", "xs": [prow["xs"][i] for i in perm]}
        scores.add(round(ACC({"rows": [pp, other]}, gold)["accuracy"], 9))
    moved += len(scores) > 1
report("permuting a nested array never moves the score, across generated documents",
       moved == 0, f"{moved}/400 documents scored differently under some inner permutation")

print("\nTHE WORKED NESTED EXAMPLE, ASSERTED END TO END")
# The example from the walkthrough, pinned: rows reordered, a nested array reversed inside a
# row, one nested cell misread, one row fabricated. Every interesting behaviour of the nested
# path in one document, so a regression in any of them shows up here.
NEST_SCHEMA = {"type": "object", "properties": {
    "segments": {"type": "array", "items": {"type": "object", "properties": {
        "name": {"type": "string"},
        "quarters": {"type": "array", "items": {"type": "number"}}}}}}}
NEST_GOLD = {"segments": [{"name": "Cloud", "quarters": [10.5, 12.0, 14.25]},
                          {"name": "Devices", "quarters": [8.0, 7.5, 9.0]}]}
NEST_PRED = {"segments": [{"name": "Devices", "quarters": [9.0, 7.5, 8.0]},
                          {"name": "Cloud", "quarters": [10.5, 12.0, 99.9]},
                          {"name": "Other", "quarters": [1.0]}]}
rn = grade(copy.deepcopy(NEST_PRED), copy.deepcopy(NEST_GOLD), NEST_SCHEMA)
tn = fair_grade(copy.deepcopy(NEST_PRED), copy.deepcopy(NEST_GOLD), NEST_SCHEMA)
report("8 gold leaves + 3 spurious = denominator 11, with 7 matched",
       (rn["total"], rn["matched"]) == (11, 7),
       f"got {rn['matched']}/{rn['total']}")
report("scores 63.64, and the paired walker agrees",
       abs(rn["accuracy"] - 700 / 11) < 1e-9
       and abs(rn["accuracy"] - tn["leaf_accuracy"]) < 1e-9,
       f"path {rn['accuracy']:.4f}, tree {tn['leaf_accuracy']:.4f}")
report("every address it found, it read correctly: the loss is entirely structural",
       abs(rn["read_right"] - 1.0) < 1e-9
       and abs(rn["found"] * rn["read_right"] - rn["accuracy"] / 100) < 1e-9,
       f"structure {rn['found']:.4f} x value {rn['read_right']:.4f}")

verdicts = {"".join(f"[{st[1]!r}]" for st in k): v
            for k, _g, _p, v in explain(copy.deepcopy(NEST_PRED), copy.deepcopy(NEST_GOLD))}
expected = {
    # the reordered-but-correct row: every quarter lands on gold's address
    "['segments'][1]['name']": "match",
    "['segments'][1]['quarters'][0]": "match",
    "['segments'][1]['quarters'][1]": "match",
    "['segments'][1]['quarters'][2]": "match",
    # the misread quarter costs exactly two slots, not the row
    "['segments'][0]['name']": "match",
    "['segments'][0]['quarters'][0]": "match",
    "['segments'][0]['quarters'][1]": "match",
    "['segments'][0]['quarters'][2]": "missing",
    "['segments'][0]['quarters']['p2']": "spurious",
    # the fabricated row gets a fresh address at the OUTER level
    "['segments']['p2']['name']": "spurious",
    "['segments']['p2']['quarters'][0]": "spurious",
}
report("every address and verdict is exactly as documented", verdicts == expected,
       "differences: " + str({k: (expected.get(k), verdicts.get(k))
                              for k in set(expected) | set(verdicts)
                              if expected.get(k) != verdicts.get(k)}))
report("a fabricated CELL inside a real row is charged separately from a fabricated ROW",
       "['segments'][0]['quarters']['p2']" in verdicts and "['segments']['p2']['name']" in verdicts)

print("\nORDER IS FREE BY DEFAULT, AND CHARGED WHERE DECLARED")
# Row order in a document is a rendering artefact, so every array is order-free unless named.
# A recipe's steps are not: step 2 before step 1 is a different answer. An array is named the
# way `explain` prints it, and a key containing a dot or a bracket is quoted there, so it
# cannot be mistaken for structure.
RECIPE = {"steps": ["mix", "bake", "cool"]}
BACKWARDS = {"steps": ["cool", "bake", "mix"]}
STEPS = ["steps"]
strict = grade(BACKWARDS, RECIPE, None, STEPS)["accuracy"]
report("an array is order-free unless declared otherwise",
       abs(ACC(BACKWARDS, RECIPE)["accuracy"] - 100) < 1e-9)
report("declaring it ordered charges the reordering",
       abs(strict - 100 / 3) < 1e-6, f"got {strict:.4f}")
report("an ordered array still scores 100 when the order is right",
       abs(grade(copy.deepcopy(RECIPE), copy.deepcopy(RECIPE), None,
                       STEPS)["accuracy"] - 100) < 1e-9)

# Declaring one array ordered must leave every other array alone.
MIXED_G = {"steps": ["mix", "bake"], "tags": ["x", "y"]}
MIXED_P = {"steps": ["mix", "bake"], "tags": ["y", "x"]}
report("declaring one array ordered does not touch the others",
       abs(grade(MIXED_P, MIXED_G, None, STEPS)["accuracy"] - 100) < 1e-9
       and abs(grade(MIXED_P, MIXED_G, None,
                           ["steps", "tags"])["accuracy"] - 50) < 1e-9)

# The pairing weight must honour the same order rule the score does. It did not at first:
# rows distinguished only by an ordered inner array all tied at weight (2,2), the matcher
# picked arbitrarily, and a PERFECT extraction scored 0.00. `_prepare` now folds an ordered
# nested array into the direct leaves, so the weight compares it positionally as well and
# `_price_nested` never sees it.
NEST_G = {"books": [{"chapters": ["a", "b"]}, {"chapters": ["b", "a"]}]}
NEST_P = {"books": [{"chapters": ["b", "a"]}, {"chapters": ["a", "b"]}]}   # rows swapped
nested_score = grade(NEST_P, NEST_G, None, ["books[*].chapters"])["accuracy"]
report("rows distinguished only by an ordered inner array still pair correctly",
       abs(nested_score - 100) < 1e-9, f"got {nested_score:.4f}")

# Rows order-free while their inner arrays are ordered: the row swap stays free, the inner
# reordering is charged. One name covers every instance, which is why `[*]` blanks the index.
DEEP_G = {"books": [{"title": "A", "chapters": ["c1", "c2"]},
                    {"title": "B", "chapters": ["d1", "d2"]}]}
DEEP_P = {"books": [{"title": "B", "chapters": ["d1", "d2"]},      # row swap: free
                    {"title": "A", "chapters": ["c2", "c1"]}]}     # inner reorder: charged
deep = grade(DEEP_P, DEEP_G, None, ["books[*].chapters"])
report("outer rows stay order-free while their inner arrays are ordered",
       abs(deep["accuracy"] - 200 / 3) < 1e-6,
       f"got {deep['accuracy']:.4f}, want 66.67")
report("a name covers every instance of a nested array",
       format_node(((KEY, "books"), (INDEX, 0), (KEY, "chapters")))
       == format_node(((KEY, "books"), (INDEX, 7), (KEY, "chapters")))
       == "books[*].chapters")
report("declaring an array ordered leaves identity at 100 over generated documents",
       all(abs(grade(copy.deepcopy(d), copy.deepcopy(d), None,
                           [format_node(n) for n in _find_arrays(list(flatten(d)))]
                           )["accuracy"] - 100) < 1e-9
           for d in (rand_doc(random.Random(12000 + s_)) for s_ in range(120))
           if flatten(d)))

print("\nTHE PRICE PROMISED IS THE SCORE DELIVERED")
# The scorer runs in three phases: PRICE a candidate pair (pure, depth-first, address-free),
# DECIDE the best pairing of a whole array, COMMIT the chosen addresses (breadth-first). The
# split is only sound if the weight pricing promised equals the leaves committing delivers at
# that pair's addresses. If they can diverge, the matcher optimises one quantity while the
# score reports another -- which is exactly what happened before ordered arrays were folded
# into an element's direct leaves, where a perfect extraction scored 0.00.
#
# It holds by construction for two reasons, and is checked here anyway: both phases call the
# same `_best_pairing`, and pricing is EXACT rather than a bound -- for two elements the
# matched count is the direct matches plus the maximum over each nested array, and nested
# arrays under one element are independent, so summing their maxima IS the maximum. That last
# step is the tree property; with `$ref`-style sharing the sub-problems would couple and the
# promise would break.
from omni_extract_bench.score import _best_pairing, _extract_rows_at, align, INDEX   # noqa: E402


def price_vs_delivery(gold, pred):
    """(promised, delivered) matched-leaf counts for every pair the matcher chose."""
    gold_addr, raw = flatten(gold), flatten(pred)
    prefix = ((KEY, "rows"),)
    grows, prows = _extract_rows_at(gold_addr, prefix, frozenset()), _extract_rows_at(raw, prefix, frozenset())
    if not grows or not prows:
        return []
    scale = 1 + len(raw)
    promised = {(a, b): m for a, b, m, _sh in _best_pairing(prows, grows, scale)}
    committed, _inexact = align(gold_addr, raw)
    out = []
    for (_a, b), m in promised.items():
        under = [k for k in gold_addr
                 if len(k) > 1 and k[:1] == prefix and k[1] == (INDEX, b)]
        delivered = sum(1 for k in under
                        if k in committed and cmp_leaf(committed[k], gold_addr[k]) >= 1.0)
        out.append((m, delivered))
    return out


def nested_row(rnd, depth=0):
    row = {f"f{j}": rnd.choice(["A", "B", "C", 1, 2.5]) for j in range(rnd.randint(1, 3))}
    if depth < 2 and rnd.random() < 0.6:
        row["xs"] = [nested_row(rnd, depth + 1) for _ in range(rnd.randint(1, 3))]
    if depth < 2 and rnd.random() < 0.4:
        row["ns"] = [rnd.choice([1, 2, 3]) for _ in range(rnd.randint(1, 3))]
    return row


checked = mismatched = 0
for s_ in range(400):
    rnd = random.Random(40000 + s_)
    gold = {"rows": [nested_row(rnd) for _ in range(rnd.randint(1, 4))]}
    pred = {"rows": [nested_row(rnd) for _ in range(rnd.randint(1, 4))]}
    for promised, delivered in price_vs_delivery(gold, pred):
        checked += 1
        mismatched += promised != delivered
report("every price the matcher paid is a price the score honours",
       mismatched == 0 and checked > 500,
       f"{mismatched} of {checked} chosen pairs delivered something other than their price")
note(f"checked {checked} chosen pairs across 400 documents nested up to three levels")

print("\nROW COUNTS AGREE WITH THE SCORER THIS REPLACES")
# `fair_grade` finds top-level arrays via `normalize.arrays_of(schema)`, so its row counts are
# only defined when a schema is supplied. `grade` derives them from the data. Compared
# here on equal footing: the paired walker is handed a schema naming every array it should see.
def schema_for(*docs):
    """Declare every top-level array either side carries, so neither scorer is blindfolded."""
    props = {}
    for d in docs:
        for k, v in d.items():
            props.setdefault(k, {"type": "array"} if isinstance(v, list) else {})
            if isinstance(v, list):
                props[k] = {"type": "array"}
    return {"properties": props}


def drop_some(node, rnd):
    """A prediction derived from gold: drops rows and corrupts values, keeps the shape."""
    if isinstance(node, dict):
        return {k: drop_some(v, rnd) for k, v in node.items()}
    if isinstance(node, list):
        return [drop_some(x, rnd) for x in node if rnd.random() > 0.3]
    return rand_scalar(rnd) if rnd.random() < 0.3 else node


diffs = []
for s_ in range(400):
    rnd = random.Random(10800 + s_)
    gold = rand_doc(rnd)
    for pred in (copy.deepcopy(gold), drop_some(copy.deepcopy(gold), rnd), {}):
        sch = schema_for(gold, pred)
        a = fair_grade(copy.deepcopy(pred), copy.deepcopy(gold), sch)
        b = grade(copy.deepcopy(pred), copy.deepcopy(gold), sch)
        # The paired walker counts rows in TOP-LEVEL arrays only. This scorer counts them at
        # every depth, so it can only ever see more, never fewer, and must agree exactly
        # wherever the document has no array below the top.
        if (b["gt_rows"] < a["gt_rows"] or b["pred_rows"] < a["pred_rows"]
                or b["matched_rows"] < a["matched"]):
            diffs.append((s_, (a["gt_rows"], a["pred_rows"], a["matched"]),
                          (b["gt_rows"], b["pred_rows"], b["matched_rows"])))
report("row counts are never lower than the paired walker's top-level-only counts",
       not diffs, f"{len(diffs)} disagreements, first 3: {diffs[:3]}")

# Counting at every depth is the point: a table one level down is still a table. The paired
# walker counts the WRAPPER row and reports recall on that, which is why it says a document
# missing 7 of its 10 rows returned everything.
deep_g = {"outer": [{"rows": [{"a": i} for i in range(10)]}]}
deep_p = {"outer": [{"rows": [{"a": i} for i in range(3)]}]}
deep = grade(deep_p, deep_g)
report("rows nested below the top level are counted",
       deep["gt_rows"] == 11 and deep["matched_rows"] == 4,
       f"gt_rows={deep['gt_rows']} matched_rows={deep['matched_rows']}, want 11 and 4")
report("recall counts leaves, so it is the same flat or wrapped",
       abs(grade(deep_p, deep_g)["recall"]
           - grade({"rows": [{"a": i} for i in range(3)]},
                   {"rows": [{"a": i} for i in range(10)]})["recall"]) < 1e-9)
report("a wrong value is charged by precision and recall, as a wrong class would be",
       grade({"a": 1, "b": 99}, {"a": 1, "b": 2})["precision"] == 0.5
       and grade({"a": 1, "b": 99}, {"a": 1, "b": 2})["recall"] == 0.5)
report("invented fields are charged even with no array in the document",
       abs(grade({"a": 1, "b": 2, "z": 9}, {"a": 1, "b": 2})["precision"] - 2/3) < 1e-9)

# The paired walker counts rows ONLY for arrays the schema declares; the path scorer counts
# any array it finds. A vendor that returns rows under a key the schema never mentioned has
# still returned rows, and they are still charged in the leaf score by both scorers -- so
# leaving them out of `pred_rows` makes precision disagree with the number beside it.
und_g = {"rows": [{"x": 1}]}
und_p = {"rows": [{"x": 1}], "extra": [{"y": 2}, {"y": 3}]}
u_sch = {"properties": {"rows": {"type": "array"}}}
u_tree = fair_grade(copy.deepcopy(und_p), copy.deepcopy(und_g), u_sch)
u_path = grade(copy.deepcopy(und_p), copy.deepcopy(und_g), u_sch)
report("rows returned under an undeclared key are counted as predicted rows",
       u_path["pred_rows"] == 3 and u_tree["pred_rows"] == 1,
       f"path {u_path['pred_rows']} (want 3), tree {u_tree['pred_rows']} (want 1)")
note(f"both charge the undeclared leaves in the score (tree {u_tree['leaf_accuracy']:.2f}, "
     f"path {u_path['accuracy']:.2f}), but only the path scorer counts them as rows")

# The paired walker's row counts COLLAPSE TO ZERO without a schema; the path scorer's do not.
# This is not a difference of opinion, it is a footgun: `cli._score_one` passes `{}` whenever
# the schema file is absent, so `score-dir` and `leaderboard` run without `--schema-dir`
# report recall and precision of 0.00 for every document while still printing a leaf score.
g_rc, p_rc = {"rows": [{"x": 1}, {"x": 2}]}, {"rows": [{"x": 1}]}
no_sch = fair_grade(copy.deepcopy(p_rc), copy.deepcopy(g_rc), {})
report("the path scorer's row counts do not depend on being handed a schema",
       ACC(p_rc, g_rc)["recall"] == 0.5,
       f"recall {ACC(p_rc, g_rc)['recall']}")
note(f"paired walker with no schema reports gt_rows={no_sch['gt_rows']}, "
     f"recall={no_sch['recall']:.2f} for a document with 2 gold rows -- see cli._score_one")

# the one case they are EXPECTED to differ on: a row with no leaves at all
er = ACC({"rows": [{}, {"x": 1}]}, {"rows": [{}, {"x": 1}]})
et = fair_grade({"rows": [{}, {"x": 1}]}, {"rows": [{}, {"x": 1}]},
                {"properties": {"rows": {"type": "array"}}})
note(f"an empty row {{}} counts as a row to the paired walker ({et['gt_rows']}) "
     f"and not to the path scorer ({er['gt_rows']}) -- it has no leaves to address")

print("\nAN APPROXIMATE SCORE SAYS SO")
restore = OM.force_approximate()            # shrink the budget, whichever solver is active
# The rows must be INSEPARABLE: component decomposition now splits an array before the solver
# sees it, so rows that share nothing never reach the fallback at all. These all share a
# column, which is what a real un-splittable table looks like (one segment label on every row,
# or a column of repeated zeros).
big_gold = [{"seg": "same", "b": float(i)} for i in range(40)]
big_pred = list(reversed(big_gold))
r_greedy = ACC({"rows": big_pred}, {"rows": big_gold})
restore()
r_exact = ACC({"rows": big_pred}, {"rows": big_gold})
report("a grade that used the greedy fallback reports matching_exact=False",
       r_greedy["matching_exact"] is False and r_greedy["approximated"],
       f"matching_exact={r_greedy['matching_exact']} greedy_blocks={r_greedy['approximated']}")
report("the same grade with the exact solver reports matching_exact=True",
       r_exact["matching_exact"] is True and not r_exact["approximated"])

# An approximate INNER solve must be reported too. `_price_nested` used to carry its own greedy
# branch and take no inexact list, so an approximate nested pairing left matching_exact=True --
# a silent approximation, which is the one thing this metric does not do. The branch is gone
# (it was order-dependent AND slower than the exact solve) and the report threads through the
# recursion. The elements here share values so the candidate graph cannot shatter into
# trivially-exact components.
INNER_G = {"rows": [{"seg": "same",
                     "xs": [{"v": i % 3, "w": "same"} for i in range(30)]}]}
INNER_P = {"rows": [{"seg": "same",
                     "xs": [{"v": i % 3, "w": "same"} for i in reversed(range(30))]}]}
report("an inner array solved exactly reports matching_exact=True",
       ACC(INNER_P, INNER_G)["matching_exact"] is True)
restore = OM.force_approximate()
r_inner = ACC(INNER_P, INNER_G)
restore()
report("an approximate INNER solve is reported, not hidden under an exact outer one",
       r_inner["matching_exact"] is False and r_inner["approximated"],
       f"matching_exact={r_inner['matching_exact']} greedy_blocks={r_inner['approximated']}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\nHARNESS METADATA IS STRIPPED THE SAME WAY")
# `normalize.prep_*` strips the {value, citations} envelope a cited field arrives in. The
# outer {"result": ...} wrapper is the CLI's job (`cli._unwrap`), not the scorer's -- asserted
# here so the division of labour is recorded rather than assumed.
plain = {"a": 1, "rows": [{"x": 5}]}
cited = {"a": {"value": 1, "citations": ["p1"], "confidence": 0.9},
         "rows": [{"x": {"value": 5, "citations": ["p2"]}}]}
report("a {value, citations} envelope around a field is stripped",
       abs(ACC(cited, plain)["accuracy"] - 100) < 1e-9,
       f"got {ACC(cited, plain)['accuracy']:.2f}")
res = {"result": plain}
report("an outer {\"result\": ...} wrapper is NOT the scorer's job, in either scorer",
       abs(ACC(res, plain)["accuracy"]
           - fair_grade(copy.deepcopy(res), copy.deepcopy(plain), {})["leaf_accuracy"]) < 1e-9,
       f"path {ACC(res, plain)['accuracy']:.2f} vs "
       f"tree {fair_grade(copy.deepcopy(res), copy.deepcopy(plain), {})['leaf_accuracy']:.2f}")
side = {"a": 1, "a_citations": ["p1"], "a_meta": {"conf": 0.9}, "rows": [{"x": 5}]}
report("per-field _citations/_meta sidecars are not charged as spurious leaves",
       abs(ACC(side, plain)["accuracy"] - 100) < 1e-9,
       f"got {ACC(side, plain)['accuracy']:.2f}")
ph = {"metrics": [{"data_period": "FY25", "segment_type": "co", "value": None},
                  {"data_period": "FY25", "segment_type": "co", "value": 5.0}]}
report("gold rows whose payload is entirely null are dropped from both sides",
       abs(ACC(copy.deepcopy(ph), copy.deepcopy(ph))["accuracy"] - 100) < 1e-9
       and abs(ACC({"metrics": [ph["metrics"][1]]}, copy.deepcopy(ph))["accuracy"] - 100) < 1e-9,
       f"identity {ACC(copy.deepcopy(ph), copy.deepcopy(ph))['accuracy']:.2f}, "
       f"without the placeholder {ACC({'metrics': [ph['metrics'][1]]}, copy.deepcopy(ph))['accuracy']:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\nTHE SCHEMA ARGUMENT DOES NOT AFFECT THE SCORE")
# grade accepts a schema for call compatibility and ignores it: leaf comparison never
# depended on declared types, and array detection now comes from the data rather than from
# `arrays_of`, which only recognised arrays it could see through anyOf.
SCH = {"properties": {"rows": {"anyOf": [{"type": "array", "items": {"type": "object"}},
                                         {"type": "null"}]}}}
g_any = {"rows": [{"x": 1}, {"x": 2}]}
p_any = {"rows": [{"x": 2}, {"x": 9}]}
with_s, without_s = ACC(p_any, g_any), grade(p_any, g_any)
report("passing a schema, or none, gives the same score and the same row counts",
       with_s == without_s, f"{with_s} vs {without_s}")
t_any = fair_grade(copy.deepcopy(p_any), copy.deepcopy(g_any), SCH)
note(f"nullable (anyOf) array row counts: paired walker {t_any['gt_rows']}, "
     f"path scorer {with_s['gt_rows']}")

# ═══════════════════════════════════════════════════════════════════════════════
print("\nBEHAVIOUR WE HAVE DECIDED NOT TO CHANGE YET (pinned so it cannot drift)")
# Object keys are addresses, matched literally: a predicted key that is not exactly a gold
# key names a field the extractor invented, and is charged as one.
KEY_G, KEY_P = {"m": {"total": 1.0, "net": 2.0}}, {"m": {"total": 1.0, "Net": 2.0}}
r_key = ACC(KEY_P, KEY_G)
report("a predicted key that is not exactly gold's is charged as invented",
       (r_key["matched"], r_key["total"]) == (1, 3),
       f"got {r_key['matched']}/{r_key['total']}, want 1/3 "
       f"(total matches; net missing; Net spurious)")
report("the path scorer and the paired walker agree on literal key matching",
       abs(r_key["accuracy"]
           - fair_grade(copy.deepcopy(KEY_P), copy.deepcopy(KEY_G), {})["leaf_accuracy"]) < 1e-9)

# OPEN MAPS ARE NOT EVALUATED. `additionalProperties` asks the extractor to invent the keys by
# reading them off the page; the strict dialect drops the keyword before the schema reaches a
# vendor, so grading it would score a request the harness never delivered. The subtree is
# skipped on both sides -- neither numerator nor denominator -- and the skip is REPORTED, so an
# ungraded chunk of a document can never pass unnoticed.
OM_SCH = {"type": "object", "properties": {
    "invoice_no": {"type": "string"},
    "groups": {"type": "object", "additionalProperties": {"type": "array",
                                                          "items": {"type": "string"}}}}}
OM_G = {"invoice_no": "INV-1", "groups": {"Core Competencies": ["SEO", "Brand"]}}
for label, om_p in (("identical", copy.deepcopy(OM_G)),
                    ("heading in caps", {"invoice_no": "INV-1",
                                         "groups": {"CORE COMPETENCIES": ["SEO", "Brand"]}}),
                    ("map omitted entirely", {"invoice_no": "INV-1"}),
                    ("map fabricated", {"invoice_no": "INV-1", "groups": {"Nonsense": ["x"]}})):
    r_om = grade(om_p, copy.deepcopy(OM_G), OM_SCH)
    t_om = fair_grade(copy.deepcopy(om_p), copy.deepcopy(OM_G), OM_SCH)
    report(f"open map, {label}: only the declared field is scored",
           (r_om["total"], r_om["matched"]) == (1, 1)
           and abs(r_om["accuracy"] - t_om["leaf_accuracy"]) < 1e-9,
           f"path {r_om['matched']}/{r_om['total']}, tree {t_om['leaf_accuracy']:.2f}")
report("the skip is reported, not silent",
       grade(copy.deepcopy(OM_G), copy.deepcopy(OM_G), OM_SCH)["skipped_open_maps"] == ["groups"]
       and fair_grade(copy.deepcopy(OM_G), copy.deepcopy(OM_G), OM_SCH)["ignored_open_maps"] == 1,
       f"path {grade(copy.deepcopy(OM_G), copy.deepcopy(OM_G), OM_SCH)['skipped_open_maps']}, "
       f"tree {fair_grade(copy.deepcopy(OM_G), copy.deepcopy(OM_G), OM_SCH)['ignored_open_maps']}")
# The walk follows the DATA, so a node is only recorded when that side reaches it. Collecting
# from gold alone meant a prediction that returned an open map gold omitted had the subtree
# skipped and never reported -- an ungraded region going unmentioned, which is exactly what the
# report exists to prevent.
report("a skip is reported whichever side reaches the open map",
       grade({"invoice_no": "INV-1"}, copy.deepcopy(OM_G), OM_SCH)["skipped_open_maps"]
       == ["groups"]
       and grade(copy.deepcopy(OM_G), {"invoice_no": "INV-1"},
                       OM_SCH)["skipped_open_maps"] == ["groups"],
       f"gold-only {grade({'invoice_no': 'INV-1'}, copy.deepcopy(OM_G), OM_SCH)['skipped_open_maps']}, "
       f"pred-only {grade(copy.deepcopy(OM_G), {'invoice_no': 'INV-1'}, OM_SCH)['skipped_open_maps']}")
report("explain shows the skipped subtree instead of letting it vanish",
       [v.verdict for v in explain(copy.deepcopy(OM_G), copy.deepcopy(OM_G), OM_SCH)
        if v.verdict.startswith("skipped")] == ["skipped (open map)"])
report("with no schema nothing is skipped, because nothing can be identified",
       grade(copy.deepcopy(OM_G), copy.deepcopy(OM_G))["total"] == 3
       and grade(copy.deepcopy(OM_G), copy.deepcopy(OM_G))["skipped_open_maps"] == [])

BG = {"rows": [{"v": 1.0}, {"v": 2.0}]}
BP = {"rows": [{"v": 1.0, "segment_type": "co"}, {"v": 2.0, "segment_type": "co"}]}
bp_path = ACC(BP, BG)["accuracy"]
bp_tree = fair_grade(copy.deepcopy(BP), copy.deepcopy(BG), {"properties": {"rows": {"type": "array"}}})["leaf_accuracy"]
report("a predicted row carrying a dimension field gold lacks is still pairable",
       abs(bp_path - 50.0) < 1e-9,
       f"path {bp_path:.2f}, want 50.00 (2 gold leaves matched, 2 extra charged)")
note(f"the paired walker scores this {bp_tree:.2f}: it blocks unconditionally, so the rows "
     f"land in disjoint blocks and every correct value is charged twice")

DG = {"rows": [{"segment_type": "company", "a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}]}
DP = {"rows": [{"segment_type": "region", "a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}]}
dp_path = ACC(DP, DG)["accuracy"]
dp_tree = fair_grade(copy.deepcopy(DP), copy.deepcopy(DG), {"properties": {"rows": {"type": "array"}}})["leaf_accuracy"]
report("a row wrong only in its dimension field is charged for that field, not for the row",
       abs(dp_path - 80.0) < 1e-9, f"path {dp_path:.2f}, want 80.00 (4 of 5 fields right)")
note(f"the paired walker scores this {dp_tree:.2f}")

print("\nTHE LARGEST REAL ARRAY IS TRACTABLE")
rnd = random.Random(1)
N = 6900
big_g = {"rows": [{"a": f"n{i}", "b": round(i * 1.5, 2), "c": rnd.choice(WORDS)}
                  for i in range(N)]}
big_p = {"rows": rnd.sample(big_g["rows"], N // 2)}
tracemalloc.start()
t0 = time.time()
rb = ACC(big_p, big_g)
dt = time.time() - t0
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
tt0 = time.time()
rt = fair_grade(copy.deepcopy(big_p), copy.deepcopy(big_g), {})
tdt = time.time() - tt0
report(f"{N} rows x 3 fields scores in bounded time and agrees with the paired walker",
       abs(rb["accuracy"] - rt["leaf_accuracy"]) < 1e-9,
       f"path {rb['accuracy']:.4f} vs tree {rt['leaf_accuracy']:.4f}")
note(f"path {dt:.1f}s, peak {peak/1e6:.0f} MB, exact={rb['matching_exact']}  |  "
     f"tree {tdt:.1f}s, exact={rt['matching_exact']}")

print(f"\n{'PATH SCORER PROPERTIES HOLD' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
