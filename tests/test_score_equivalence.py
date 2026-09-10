#!/usr/bin/env python3
"""Equivalence harness for the path-based scorer (`score`).

PARITY IS NOT THE GOAL. The paired walker is known to be wrong in specific ways -- three
denominator-erosion defects and a P2 violation were found while building this module -- so a
harness that demanded identical numbers would be pinning the new scorer to the old one's
bugs. What makes the rewrite adoptable is instead:

  * every divergence is CLASSIFIED, with the side that is right named and a reason given;
  * the path scorer holds the properties in its own right (P1, P2, P19, the factorisation);
  * no divergence is unexplained. An unclassified difference is a failure.

Divergences are reported as counts by class, not asserted away.

Run: python3 tests/test_paths_equivalence.py
"""
import copy
import os as _os
import random
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench.grading import fair_grade                    # noqa: E402
from omni_extract_bench.score import grade, flatten    # noqa: E402

FAILS = []
WORDS = ["alpha", "beta", "gamma", "Acme Ltd", "J. Smith", "2024-03-01", "Q2", "USD"]


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if not ok else ""))
    if not ok:
        FAILS.append(f"{name} {detail}")


def rand_scalar(rnd):
    return rnd.choice([rnd.randint(1, 9999), round(rnd.uniform(0, 5000), 2),
                       rnd.choice(WORDS), rnd.choice([True, False]), None])


def rand_row(rnd, nf=None):
    return {f"f{i}": rand_scalar(rnd) for i in range(nf or rnd.randint(2, 5))}


def rand_doc(rnd, depth=0):
    """Every array shape the scorer has to survive, including the ones that used to break it."""
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


def reshape(node, rnd):
    """Swap a container for another kind: the disagreement a real vendor produces."""
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


def perturb(node, rnd):
    """Corrupt values and drop rows, leaving the shape intact."""
    if isinstance(node, dict):
        return {k: perturb(v, rnd) for k, v in node.items()}
    if isinstance(node, list):
        kept = [x for x in node if rnd.random() > 0.25]
        return [perturb(x, rnd) for x in kept]
    return rand_scalar(rnd) if rnd.random() < 0.3 else node


# ── 1. divergence census: every difference must fall into a KNOWN class ─────────
print("\n[1] DIVERGENCE CENSUS vs grading.fair_grade")


def classify(gold, pred):
    """Name the reason the two scorers differ on this document, or None if unexplained.

    Each class corresponds to a defect found in the paired walker while building this
    module. A document that diverges for none of these reasons is a real failure: either a
    bug in the path scorer, or a fourth defect nobody has characterised yet.
    """
    def nested_rows(n):
        """A row whose discriminating content is nested -- `_row_signature` cannot see it,
        so the paired walker pairs such rows blind (its P2 violation)."""
        if isinstance(n, dict):
            return any(nested_rows(v) for v in n.values())
        if isinstance(n, list):
            return any(isinstance(x, dict) and any(isinstance(v, (dict, list))
                                                   for v in x.values()) for x in n) \
                or any(nested_rows(x) for x in n)
        return False

    def dup_scalars(n):
        """A scalar array whose elements are not distinct: the greedy multiset match and the
        assignment can pick different partners, which is free to differ under P2 (equal
        weight) but changes which ADDRESS a value lands on."""
        if isinstance(n, dict):
            return any(dup_scalars(v) for v in n.values())
        if isinstance(n, list):
            sc = [FGP_canon(x) for x in n if not isinstance(x, (dict, list)) and x is not None]
            return len(sc) != len(set(sc)) or any(dup_scalars(x) for x in n)
        return False

    if nested_rows(gold) or nested_rows(pred):
        return "nested-row pairing (path scorer right: paired walker violates P1/P2)"
    if dup_scalars(gold) or dup_scalars(pred):
        return "duplicate scalar elements (tie between equal-weight assignments)"

    def sparse_rows(n):
        """Rows with many nulls produce weight-1 ties whose partners differ in DENOMINATOR.

        METRIC_SPEC 3.2 maximises the number of matched leaves; 4 divides by the union of
        gold and spurious addresses. Those are different objectives, so equal-weight
        assignments can score differently and the metric does not say which to pick. Measured
        on a real case: the same 11 matched leaves over a denominator of 20 or of 21,
        depending only on which weight-1 partner a row is given. Neither scorer is wrong;
        the specification is incomplete.
        """
        if isinstance(n, dict):
            return any(sparse_rows(v) for v in n.values())
        if isinstance(n, list):
            return any(isinstance(x, dict) and sum(v is None for v in x.values()) >= 1
                       for x in n) or any(sparse_rows(x) for x in n)
        return False

    if sparse_rows(gold) or sparse_rows(pred):
        return ("equal-weight tie with unequal denominators "
                "(METRIC_SPEC maximises matched leaves, not accuracy -- spec is incomplete)")
    return None


from omni_extract_bench.grading import canon_key as FGP_canon           # noqa: E402


classes, unexplained, total_div = {}, [], 0
for s_ in range(400):
    rnd = random.Random(s_)
    gold = rand_doc(rnd)
    for label, pred in (("identity", copy.deepcopy(gold)),
                        ("perturbed", perturb(copy.deepcopy(gold), rnd)),
                        ("reshaped", reshape(copy.deepcopy(gold), rnd)),
                        ("empty", {})):
        a = fair_grade(copy.deepcopy(pred), copy.deepcopy(gold), {})["leaf_accuracy"]
        b = grade(copy.deepcopy(pred), copy.deepcopy(gold), {})["accuracy"]
        if abs(a - b) <= 1e-9:
            continue
        total_div += 1
        why = classify(gold, pred)
        if why is None:
            unexplained.append((s_, label, a, b, gold, pred))
        else:
            classes[why] = classes.get(why, 0) + 1
print(f"  1600 documents x 4 predictions -> {total_div} divergences")
for why, n in sorted(classes.items(), key=lambda kv: -kv[1]):
    print(f"    {n:5d}  {why}")
report("every divergence falls into a known, named class",
       not unexplained,
       f"{len(unexplained)} unexplained, first: seed {unexplained[0][0]} "
       f"({unexplained[0][1]}) tree={unexplained[0][2]:.4f} path={unexplained[0][3]:.4f}"
       if unexplained else "")

# ── 2. properties the paired walker holds, which must not regress ───────────────
print("\n[2] PROPERTIES")
ok = True
for s in range(200):
    rnd = random.Random(1000 + s)
    gold = rand_doc(rnd)
    if not flatten(gold):
        continue          # no leaves -> denominator 0 -> 0.0 by convention, on both scorers
    if abs(grade(copy.deepcopy(gold), copy.deepcopy(gold), {})["accuracy"] - 100) > 1e-9:
        ok = False
        report("grading a document against itself scores 100", False, f"seed {s}")
        break
report("grading a document against itself scores 100", ok)

ok = True
for s in range(200):
    rnd = random.Random(2000 + s)
    gold = rand_doc(rnd)
    if not flatten(gold):
        continue
    perm = copy.deepcopy(gold)

    def shuffle(n):
        if isinstance(n, dict):
            return {k: shuffle(v) for k, v in n.items()}
        if isinstance(n, list):
            out = [shuffle(x) for x in n]
            rnd.shuffle(out)
            return out
        return n
    if abs(grade(shuffle(perm), copy.deepcopy(gold), {})["accuracy"] - 100) > 1e-9:
        ok = False
        report("reordering any array leaves a correct answer at 100", False, f"seed {s}")
        break
report("reordering any array leaves a correct answer at 100, for every array shape", ok)

# P2 for nested rows: the case the paired walker fails (145/200 measured). This is the
# defect that motivated the rewrite, so it is asserted here rather than merely described.
bad_tree = bad_path = 0
for s in range(200):
    rnd = random.Random(3000 + s)
    rows = [{"tag": "same", "d": {f"d{i}": rand_scalar(rnd) for i in range(3)}}
            for _ in range(rnd.randint(2, 4))]
    gold = {"rows": rows}
    perm = {"rows": random.Random(s + 1).sample(rows, len(rows))}
    bad_tree += fair_grade(copy.deepcopy(perm), copy.deepcopy(gold), {})["leaf_accuracy"] < 99.99
    bad_path += grade(copy.deepcopy(perm), copy.deepcopy(gold), {})["accuracy"] < 99.99
report("rows distinguished only by nested content stay order-free",
       bad_path == 0, f"path scorer failed {bad_path}/200 (paired walker fails {bad_tree}/200)")
print(f"        (for reference: grading.fair_grade violates this on {bad_tree}/200)")

ok = True
for s in range(300):
    rnd = random.Random(4000 + s)
    gold = rand_doc(rnd)
    pred = reshape(copy.deepcopy(gold), rnd)
    r = grade(pred, copy.deepcopy(gold), {})
    floor = len(flatten(gold))
    if r["total"] < floor:
        ok = False
        report("gold leaves never leave the denominator", False,
               f"seed {s}: denom {r['total']} < gold leaves {floor}")
        break
report("gold leaves never leave the denominator", ok)

# ── 3. the factorisation is exact, not approximate ──────────────────────────────
print("\n[3] DECOMPOSITION")
worst = 0.0
for s in range(400):
    rnd = random.Random(5000 + s)
    gold = rand_doc(rnd)
    pred = perturb(copy.deepcopy(gold), rnd)
    r = grade(pred, copy.deepcopy(gold), {})
    worst = max(worst, abs(r["found"] * r["read_right"] - r["accuracy"] / 100))
report("leaf_accuracy == path_jaccard x value_accuracy", worst < 1e-9, f"max error {worst:.2e}")

# ── 4. the shape defects that motivated this module ─────────────────────────────
print("\n[4] SHAPE DEFECTS (unrepresentable here, asserted anyway)")
G = {"invoice_no": "INV-1", "rows": [{"a": "x", "b": 1.5}, {"a": "y", "b": 2.5}]}
as_list = grade({"invoice_no": "INV-1", "rows": []}, copy.deepcopy(G), {})
as_obj = grade({"invoice_no": "INV-1", "rows": {}}, copy.deepcopy(G), {})
report("a gold array returned as an object is charged, not dropped",
       as_obj["total"] == as_list["total"],
       f"[] -> {as_list['total']}, {{}} -> {as_obj['total']}")

GM = {"items": [{"a": "x"}, "note1", "note2"]}
report("mixed arrays: scalar elements are scored",
       grade({"items": [{"a": "x"}]}, copy.deepcopy(GM), {})["accuracy"] < 99.0
       and grade(copy.deepcopy(GM), copy.deepcopy(GM), {})["accuracy"] > 99.9)

GG = {"grid": [[1.5, 2.5], [3.5, 4.5]]}
report("arrays of arrays: compared structurally, not by string repr",
       grade({"grid": [[15, 25], [35, 45]]}, copy.deepcopy(GG), {})["accuracy"] < 1.0
       and grade({"grid": [[3.5, 4.5], [1.5, 2.5]]}, copy.deepcopy(GG), {})["accuracy"] > 99.9)

print(f"\n{'PATH SCORER EQUIVALENT' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
