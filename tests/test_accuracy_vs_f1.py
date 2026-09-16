#!/usr/bin/env python3
"""How `accuracy` and `f1` relate, and exactly when they disagree.

They can rank two predictions oppositely, so the spec has to say which one decides. It does
(`accuracy`), and this file establishes the algebra behind that choice, because "these two
numbers sometimes disagree" is the sort of thing a reader is entitled to see proven.

    m = matched   w = misread   u = unfound   x = fabricated + invented_item + invented_field

    G = m + w + u        gold addresses
    P = m + w + x        asserted addresses
    T = m + w + u + x    the union

    accuracy = m / T          f1 = 2m / (G + P)

`f1` is the Dice coefficient over (address, value) pairs, and charges a misread twice --
`FP = w + x` and `FN = w + u` both contain it, as in object detection. `accuracy` is Jaccard
over the same pairs EXCEPT that a misread counts once instead of twice, which is P4 and which
is what makes "you recovered 60% of this document" literally true.

That single deviation is the whole reason the two can disagree about ranking. With no
misreads, accuracy is Jaccard and f1 is 2J/(1+J) -- monotone, so identical ordering.

Run: python3 tests/test_accuracy_vs_f1.py
"""
import copy
import os as _os
import random
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench.score import grade                                 # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


rnd = random.Random(11)


def scalar():
    return rnd.choice([1, 2.5, "a", "B", "2024-10-31", 0, "x"])


def doc(d=0):
    o = {}
    for i in range(rnd.randint(1, 4)):
        r = rnd.random()
        if d < 2 and r < .3:    o[f"f{i}"] = [doc(d + 1) for _ in range(rnd.randint(0, 3))]
        elif r < .45:           o[f"f{i}"] = [scalar() for _ in range(rnd.randint(0, 3))]
        elif d < 2 and r < .55: o[f"f{i}"] = doc(d + 1)
        else:                   o[f"f{i}"] = scalar()
    return o


def sch(x):
    if isinstance(x, dict):
        return {"type": "object", "properties": {k: sch(v) for k, v in x.items()}}
    if isinstance(x, list):
        merged = {}
        for e in x:
            if isinstance(e, dict):
                merged.update(e)
        return {"type": "array", "items": sch(merged) if merged else {}}
    return {}


def perturb(node, p_wrong, p_drop, p_add):
    node = copy.deepcopy(node)

    def walk(y):
        if isinstance(y, dict):
            for k in list(y):
                if rnd.random() < p_wrong:  y[k] = scalar()
                elif rnd.random() < p_drop: del y[k]
                else:                       walk(y[k])
            if rnd.random() < p_add:        y[f"z{rnd.randint(0, 9)}"] = scalar()
        elif isinstance(y, list):
            for e in y:
                walk(e)
    walk(node)
    return node


def buckets(r):
    m, w, u = r["matched"], r["misread"], r["unfound"]
    x = r["fabricated"] + r["invented_item"] + r["invented_field"]
    return m, w, u, x


# ═══════════════════════════════════════════════════════════════════════════════
print("\nTHE ALGEBRA, OVER GENERATED DOCUMENTS")
bad_t = bad_f1 = bad_acc = bad_dice = seen = 0
for _ in range(1000):
    g = doc()
    s = sch(g)
    if not s.get("properties"):
        continue
    for p in (copy.deepcopy(g), perturb(g, .3, .1, .1), perturb(g, .0, .3, .3), {}):
        r = grade(p, g, s)
        seen += 1
        m, w, u, x = buckets(r)
        G, P, T = m + w + u, m + w + x, m + w + u + x
        bad_t += T != r["total"]
        bad_acc += bool(T) and abs(r["accuracy"] / 100 - m / T) > 1e-12
        bad_f1 += bool(G and P) and abs(r["f1"] - 2 * m / (G + P)) > 1e-12
        if w == 0 and T and G and P:
            J = m / T
            bad_dice += abs(r["f1"] - 2 * J / (1 + J)) > 1e-12
report(f"total == m + w + u + x   ({seen} gradings)", bad_t == 0, f"{bad_t} violations")
report("accuracy == m / total", bad_acc == 0, f"{bad_acc} violations")
report("f1 == 2m / (gold + asserted), i.e. Dice over (address, value) pairs",
       bad_f1 == 0, f"{bad_f1} violations")
report("with no misread, accuracy is Jaccard and f1 == 2J/(1+J)",
       bad_dice == 0, f"{bad_dice} violations")
note("so f1 is the textbook measure; accuracy is Jaccard minus one misread charge (P4)")

print("\naccuracy READS THE ALIGNMENT; f1 AND JACCARD CANNOT")
# The claim METRIC_SPEC section 4 rests on. These two predictions are indistinguishable to
# any function of (matched, |gold|, |asserted|) -- so f1 and jaccard MUST score them alike.
# accuracy does not, because it knows the first pair shares an address.
AB_S = {"properties": {k: {"type": "string"} for k in "abc"}}
AB_G = {"a": "1", "b": "2"}
A = grade({"a": "1", "b": "99"}, AB_G, AB_S)      # found b, misread it
B = grade({"a": "1", "c": "99"}, AB_G, AB_S)      # missed b, invented c
def _j(r):
    m, w, u = r["matched"], r["misread"], r["unfound"]
    x = r["fabricated"] + r["invented_item"] + r["invented_field"]
    d = m + 2 * w + u + x
    return m / d if d else 0.0
report("the two are identical in claim-set terms",
       (A["matched"], A["matched"] + A["misread"] + A["unfound"],
        A["asserted"]) == (B["matched"],
                           B["matched"] + B["misread"] + B["unfound"], B["asserted"]),
       f"A m={A['matched']} |G|={A['matched']+A['misread']+A['unfound']} "
       f"|P|={A['asserted']}; B m={B['matched']} "
       f"|G|={B['matched']+B['misread']+B['unfound']} |P|={B['asserted']}")
report("so f1 scores them the same", abs(A["f1"] - B["f1"]) < 1e-12,
       f"{A['f1']:.4f} vs {B['f1']:.4f}")
report("and jaccard scores them the same", abs(_j(A) - _j(B)) < 1e-12,
       f"{_j(A):.4f} vs {_j(B):.4f}")
report("but accuracy tells them apart", abs(A["accuracy"] - B["accuracy"]) > 1e-9,
       f"{A['accuracy']:.2f} vs {B['accuracy']:.2f}")
note("hence accuracy is NOT a set-similarity measure -- it uses strictly more information,")
note("and f1 is a coarsening of it rather than a generalisation")

print("\nWHERE accuracy IS INDIFFERENT, AND WHERE IT IS NOT")
IN_S = {"properties": {"a": {"type": "string"}, "b": {"type": "string"},
                       "blank": {"type": ["string", "null"]}}}
filled_G = {"a": "1", "b": "2"}
silent_G = {"a": "1", "blank": None}
abst_f = grade({"a": "1"}, filled_G, IN_S)
gues_f = grade({"a": "1", "b": "WRONG"}, filled_G, IN_S)
abst_s = grade({"a": "1"}, silent_G, IN_S)
gues_s = grade({"a": "1", "blank": "WRONG"}, silent_G, IN_S)
report("at an address where gold HAS a value, a wrong value costs what a blank costs",
       abs(abst_f["accuracy"] - gues_f["accuracy"]) < 1e-9,
       f"blank {abst_f['accuracy']:.2f}, wrong {gues_f['accuracy']:.2f}")
report("...and precision is what separates them",
       gues_f["precision"] < abst_f["precision"] - 1e-9)
report("where gold is SILENT, asserting is charged -- it creates a new address",
       gues_s["accuracy"] < abst_s["accuracy"] - 1e-9,
       f"blank {abst_s['accuracy']:.2f}, asserted {gues_s['accuracy']:.2f}")
report("...and it is counted as `fabricated`", gues_s["fabricated"] == 1,
       f"fabricated {gues_s['fabricated']}")
note("so the hallucination case is charged in the headline, not only in a diagnostic")

print("\nP20 -- WITH NO MISREADS, THEIR RANKING IS IDENTICAL")
pairs = inv = 0
for _ in range(4000):
    g = doc()
    s = sch(g)
    if not s.get("properties"):
        continue
    a, b = perturb(g, .0, .25, .25), perturb(g, .0, .10, .40)
    ra, rb = grade(a, g, s), grade(b, g, s)
    if ra["misread"] or rb["misread"] or abs(ra["accuracy"] - rb["accuracy"]) < 1e-9:
        continue
    pairs += 1
    inv += (ra["accuracy"] > rb["accuracy"]) != (ra["f1"] > rb["f1"])
report(f"zero ranking disagreements over {pairs} misread-free pairs", inv == 0,
       f"{inv} disagreements, which would break P20")
note("2J/(1+J) is strictly increasing in J, so no disagreement is possible here")

print("\nAND WITH MISREADS, THEY CAN DISAGREE -- WHICH IS WHY accuracy DECIDES")
pairs2 = inv2 = 0
for _ in range(4000):
    g = doc()
    s = sch(g)
    if not s.get("properties"):
        continue
    a, b = perturb(g, .35, .02, .02), perturb(g, .02, .25, .25)
    ra, rb = grade(a, g, s), grade(b, g, s)
    if abs(ra["accuracy"] - rb["accuracy"]) < 1e-9:
        continue
    pairs2 += 1
    inv2 += (ra["accuracy"] > rb["accuracy"]) != (ra["f1"] > rb["f1"])
report("a model that misreads and one that omits/invents CAN be ranked oppositely",
       inv2 > 0, f"{inv2} of {pairs2} pairs -- if this hits zero the docs are stale")
note(f"{inv2} of {pairs2} pairs ({100 * inv2 / max(pairs2, 1):.1f}%) -- accuracy is kinder to")
note("a model that misreads, f1 to one that omits and invents. accuracy is the score.")

print("\nP21 -- accuracy CANNOT BE RAISED EXCEPT BY BEING RIGHT MORE OFTEN")
# Exhaustive over every strategy on a small document: omit each field, fill it correctly, or
# fill it wrongly. If any strategy ever scored above one that produced MORE correct values,
# accuracy would be gameable. None does.
import itertools                                                            # noqa: E402

G_S = {"properties": {f"k{i}": {"type": "string"} for i in range(6)}}
G_G = {f"k{i}": f"v{i}" for i in range(4)}
by_correct = {}
for bits in itertools.product([0, 1, 2], repeat=6):
    pred = {}
    for i, b in enumerate(bits):
        if b == 1:   pred[f"k{i}"] = G_G.get(f"k{i}", "WRONG")
        elif b == 2: pred[f"k{i}"] = "WRONG"
    r = grade(pred, G_G, G_S)
    by_correct.setdefault(r["matched"], []).append(r["accuracy"])
ceilings = [(m, max(v)) for m, v in sorted(by_correct.items())]
report(f"the best reachable accuracy is monotone in correct values ({3**6} strategies)",
       all(a <= b for (_m, a), (_n, b) in zip(ceilings, ceilings[1:])), str(ceilings))
report("no strategy beats one that produced more correct values",
       all(max(by_correct[m]) <= min(max(by_correct[n]) for n in by_correct if n > m)
           for m in by_correct if any(n > m for n in by_correct)),
       str(ceilings))
note("so guessing well raises accuracy because it produces correct values -- not an exploit")

print("\nWHAT accuracy IS INDIFFERENT TO, AND WHAT precision SAYS ABOUT IT")
IND_S = {"properties": {"a": {"type": "string"}, "b": {"type": "string"}}}
IND_G = {"a": "x", "b": "y"}
blank = grade({"a": "x"}, IND_G, IND_S)
wrong = grade({"a": "x", "b": "WRONG"}, IND_G, IND_S)
report("a wrong value and a blank score the same accuracy",
       abs(blank["accuracy"] - wrong["accuracy"]) < 1e-9,
       f"blank {blank['accuracy']}, wrong {wrong['accuracy']}")
report("...and precision tells them apart",
       wrong["precision"] < blank["precision"] - 1e-9,
       f"blank {blank['precision']:.3f}, wrong {wrong['precision']:.3f}")
note("nothing is gained by the wrong value, but a consumer would act on it -- hence both")

print("\nCORPUS VARIANCE PRICES A BASE-RATE GUESS; IT DOES NOT DEFEAT IT")
BR_S = {"properties": {"lines": {"type": "array", "items": {"properties":
        {"sku": {"type": "string"}, "region": {"type": "string"}}}}}}


def _corpus(seed, us_rate):
    rr = random.Random(seed)
    return {"lines": [{"sku": f"S{i}",
                       "region": "US" if rr.random() < us_rate else rr.choice(["EU", "APAC"])}
                      for i in range(60)]}


gains = []
for rates in ([0.9] * 4, [0.9, 0.5, 0.3, 0.1], [0.1] * 4):
    a = g_ = 0.0
    for i, rate in enumerate(rates):
        d = _corpus(i, rate)
        a += grade({"lines": [{"sku": x["sku"]} for x in d["lines"]]}, d, BR_S)["accuracy"]
        g_ += grade({"lines": [{"sku": x["sku"], "region": "US"} for x in d["lines"]]},
                    d, BR_S)["accuracy"]
    gains.append(round((g_ - a) / len(rates), 2))
report("guessing the modal value gains on every corpus, most where the prior is strongest",
       all(g > 0 for g in gains) and gains[0] > gains[1] > gains[2], str(gains))
note("so variance is not a defence -- it prices the guess at the average hit rate, correctly")

print("\nAN OMISSION ALWAYS COSTS AT LEAST AS MUCH AS AN INVENTION")
# (m-1)/T against m/(T+1): the first is smaller for every m <= T, by 1/(T(T+1)).
worse = []
for N in (2, 3, 5, 10, 50):
    keys = [f"k{i}" for i in range(N)]
    S = {"properties": {k: {"type": "number"} for k in keys + ["extra"]}}
    gold = {k: i for i, k in enumerate(keys)}
    a_omit = grade({k: i for i, k in enumerate(keys) if k != keys[-1]}, gold, S)["accuracy"]
    a_inv = grade(dict(gold, extra=999), gold, S)["accuracy"]
    worse.append((N, round(a_omit, 2), round(a_inv, 2), a_omit <= a_inv + 1e-9))
report("losing a true fact never scores above adding a false one",
       all(ok for *_r, ok in worse), str(worse))
note("gap is 1/(T(T+1)): 16.67 points at 2 values, 0.04 at 50 -- P12's lean, but faint")

print(f"\n{'accuracy AND f1 RELATE AS DOCUMENTED' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
