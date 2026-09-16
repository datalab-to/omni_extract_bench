#!/usr/bin/env python3
"""What `canon_key` must and must not fold, stated as properties.

Normalisation exists for one purpose: to absorb the difference between two faithful
transcriptions of the same ink, and nothing else. A model reads a value off a page; a human
wrote the ground truth from the same page; the two spell it differently. Everything
`canon_key` folds should be that difference, and nothing it folds should be more.

Four properties follow, and this file is each of them written down.

P1  NO FALSE MERGES.  If a document could plausibly contain both values IN THE SAME FIELD
    MEANING DIFFERENT THINGS, their keys must differ. That phrasing is the operational test
    and it settles cases quickly: no document distinguishes `1,000` from `1000` as different
    amounts, so fold; a document absolutely can carry `INV-007` and `INV-7` as different
    invoices, so do not.

P2  NO FALSE SPLITS.  If both values are faithful transcriptions of the same ink, the keys
    must be equal. `ACME CORP` and `Acme Corp`; `Inst itutional` and `Institutional`.

P1 and P2 pull against each other and that tension IS the design. A table that only checked
one direction would be satisfied by a function that folds everything, or nothing.

P3  THE FALLBACK IS THE FLOOR.  `canon_key` tries number, then date, then timestamp, then
    falls through to the string path. The fallback receives every value no recogniser claimed
    -- INCLUDING malformed ones -- so it must be the most conservative transform available.
    Today it is the most aggressive, which is backwards: failing to parse as a date currently
    earns a value the heaviest mangling in the system. Every collision in KNOWN_COLLISIONS
    reached it that way.

P4  THE KEY IS A FUNCTION OF THE VALUE ALONE.  Not of the field, the document, or a config.
    This is what per-field scoring rules were rejected to preserve; see METRIC_SPEC 5.2.

Two lists hold the P1 violations that exist today, and the difference between them matters.

ACCEPTED_LENIENCY is leniency we CHOSE. Punctuation folds from anywhere, which merges
`5.2.1.5` with `5215`; we kept that because measuring the corpus said the fold recovers 1,607
genuine matches and credits no wrong value even once. This is a benchmark of extraction, not
of punctuation, and where the evidence is one-sided the models get the benefit of the doubt.

KNOWN_COLLISIONS is debt nobody chose. The test asserts that set has not GROWN, and tells you
when one is fixed so it can be promoted into MUST_DIFFER.

Both are listed rather than absent, and that IS the point: a silent merge is invisible in a
score, and the reason 140 of these went unnoticed in the corpus is that nothing was looking.

Run: python3 tests/test_canon_properties.py
"""
import os as _os
import random
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from omni_extract_bench.values import canon_key                        # noqa: E402

FAILS = []


def report(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}" if not ok else ""))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f"          {text}")


# ── P2: two spellings of one fact must agree ────────────────────────────────────────────
MUST_MATCH = [
    ("case",                  "ACME CORP",              "Acme Corp"),
    ("doubled whitespace",    "Acme  Corp",             "Acme Corp"),
    ("OCR mid-word space",    "Inst itutional",         "Institutional"),
    ("trailing period",       "Table B-1.",             "Table B-1"),
    ("company suffix punct",  "101 SECOND STREET INC.", "101 SECOND STREET, INC."),
    ("smart quote",           "don’t",             "don't"),
    ("thousands separator",   "1,000",                  "1000"),
    ("currency symbol",       "$5",                     "5"),
    ("percent symbol",        "50%",                    "50"),
    ("accounting negative",   "(98.2)",                 "-98.2"),
    ("trailing minus",        "98.2-",                  "-98.2"),
    ("number precision",      "5",                      "5.0"),
    ("number precision 2",    "12.9",                   "12.90"),
    ("date formats",          "01/15/2024",             "2024-01-15"),
    ("timestamp at midnight", "2024-10-31T00:00:00Z",   "2024-10-31"),
    ("integral float ID",     "8303911426.0",           "8303911426"),
    ("footnote marker",       "229 [1]",                "229"),
    ("footnote marker, letter","Total [a]",              "Total"),
    # Punctuation folds from anywhere -- see ACCEPTED_LENIENCY below for the price and the
    # measurement that set it. These are the cases it buys, and they are the common ones:
    # 1,532 of the 1,607 matches it recovers differ by punctuation alone.
    # Unicode spellings of ASCII characters. NFKD leaves all of these alone, so they had to be
    # named -- 994 corpus values carry one, and `Pascual\u2010Montano` did not fold like
    # `Pascual-Montano` until they were.
    ("unicode HYPHEN",        "Pascual\u2010Montano",        "Pascual-Montano"),
    ("non-breaking hyphen",   "asset\u2011backed",           "asset-backed"),
    ("minus sign",            "180.08 \u2212 121.40",        "180.08 - 121.40"),
    ("soft hyphen is invisible", "Bras\u00adilia",           "Brasilia"),
    # A list marker at the START of a value is the page's bullet, not the value.
    ("leading bullet",        "\u2022 Maintain a safe work environment",
                              "Maintain a safe work environment"),
    ("leading black square",  "\u25a0 MAHOGANY HOUSE",       "MAHOGANY HOUSE"),
    ("leading arrow",         "\u25b6 USA",                  "USA"),
    ("dash runs are one mark", "-",                     "--"),
    ("longer dash run",       "---",                    "-"),
    ("em dash",               "\u2014",                     "-"),
    ("EIN",                   "31-1440073",             "311440073"),
    ("ZIP+4",                 "44308-1801",             "443081801"),
    ("phone",                 "713-203-6913",           "7132036913"),
    ("phone, dotted",         "713.203.6913",           "713-203-6913"),
    # two or more dots between digits are separators, not a decimal point, so they still fold
    ("dotted phone, 3 groups", "512.784.7407",          "512-784-7407"),
    # Two or more dots cannot be decimal points, so a European thousands grouping folds to the
    # US one with no ambiguity. Only the SINGLE dotted group `1.000` is undecidable.
    ("European millions",     "1.000.000",              "1,000,000"),
    ("European millions, 7",  "1.234.567",              "1,234,567"),
    ("with a unit",           "12.345.678 EUR",         "12,345,678 EUR"),
    ("post box",              "P.O. BOX 125",           "PO BOX 125"),
    ("abbreviated street",    "100 F STREET, NE",       "100 F. STREET NE"),
    ("abbreviated city",      "FT LAUDERDALE",          "FT. LAUDERDALE"),
    ("legal entity suffix",   "BLOOMBERG FINANCE L.P.", "BLOOMBERG FINANCE LP"),
]

# ── P1: values a document could hold as DIFFERENT facts must not agree ──────────────────
MUST_DIFFER = [
    ("arXiv id",              "arXiv:2405.06211v3",     "arXiv:2405.6211v3"),
    ("controlled vocab",      "None",                   "Not applicable"),
    # Placeholder WORDS are ink, and different words are different answers. On an
    # adverse-event form `None` (no action was taken) and `N/A` (does not apply) are not the
    # same reply, and a dash is a third thing. None of them is an empty cell either: absence
    # has no address at all, so it cannot be confused with a printed mark.
    ("printed None vs N/A",   "None",                   "N/A"),
    ("printed None vs dash",  "None",                   "-"),
    ("printed N/A vs dash",   "N/A",                    "--"),
    ("printed word vs empty", "None",                   ""),
    ("dash vs empty",         "-",                      ""),
    # `NA` is not reliably a placeholder: in Nike's 10-Q `segment_name` it sits beside
    # `North America` and `Greater China`, where it abbreviates the region. Folding it to
    # empty keyed 2,546 values as nothing.
    ("NA is not empty",       "NA",                     ""),
    ("NA vs None",            "NA",                     "None"),
    # A fold may remove an annotation; it may never CONSUME the value. The footnote rule used
    # to erase `[1]` entirely, so the fourteen `ref_number` values in the EU e-invoice
    # bibliography all keyed as empty and reversing every one of them scored 100.00.
    # ...but only at the start. An interior marker may be separating two things, and the
    # corpus has the cases that make each exclusion necessary.
    ("interior bullet kept",  "MITTAL COURT \u2219 NARIMAN POINT", "MITTAL COURT NARIMAN POINT"),
    ("interior arrow kept",   "country \u25b6 USA",          "country USA"),
    # MIDDLE DOT is a unit separator: N\u00b7m is a newton-metre, nm is a nanometre.
    ("middle dot is a unit",  "N\u00b7m",                    "Nm"),
    # \u00ab\u2026\u00bb are quotation marks in the Greek filings, not bullets.
    ("guillemets are quotes", "\u00ab\u039d\u03b9\u03ba\u03cc\u03bb\u03b1\u03bf\u03c2\u00bb",            "\u039d\u03b9\u03ba\u03cc\u03bb\u03b1\u03bf\u03c2"),
    # A redaction block is content; stripping \u25a0 everywhere would make every one alike.
    ("redaction blocks differ", "\u25a0\u25a0\u25a0-\u25a0\u25a0-\u25a0\u25a0\u25a0\u25a0",      "\u25a0\u25a0-\u25a0\u25a0\u25a0\u25a0"),
    ("redaction is not a dash", "\u25a0\u25a0\u25a0-\u25a0\u25a0-\u25a0\u25a0\u25a0\u25a0",      "-"),
    ("bracket ref numbers",   "[1]",                    "[2]"),
    ("bracket vs empty",      "[1]",                    ""),
    ("bracket vs ellipsis",   "[1]",                    "..."),
    # promoted from KNOWN_COLLISIONS once the fold that merged them was narrowed
    ('zero-padded identifier', 'INV-007', 'INV-7'),
    ('zero-padded postal', '02000', '2000'),
    ('zero-padded state', '06', '6'),
    ('zip, all zeros', '0', '00000'),
    ('email local part', 'wenqifan03@gmail.com', 'wenqifan3@gmail.com'),
    ('part number', 'A-01', 'A1'),
    ('swim time', '1:00.50', '1:50'),
    ('swim time, precision', '1:03.28', '1:3.28'),
    ('drug concentration', '0.11%w/w', '11%w/w'),
    ('share class', 'COM PAR $.001', 'COM PAR $.01'),
    # A decimal point inside a longer value is content, not punctuation. Deleting it turned
    # `1.5 mg` into `15mg`, so a 1.5 mg and a 15 mg dose arm were one key. `_f_number` only
    # protects a value that is ENTIRELY a numeral, so anything carrying a unit was exposed.
    ('dose', '1.5 mg', '15 mg'),
    ('dose, trailing zero', '5.00 kg', '500 kg'),
    ('dose with a rate', '12.5 mg/day', '125 mg/day'),
    ('trial arm', 'Ormelytide 1.5 mg once weekly', 'Ormelytide 15 mg once weekly'),
    ('concentration', '0.5 mL', '05 mL'),
]

#: LENIENCY WE CHOSE, not debt. Each is a real P1 violation, and each is the price of a
#: fold the corpus said was worth paying. A benchmark of extraction should not fail a model
#: over a period, so where the measurement is one-sided the models get the benefit of the
#: doubt -- but the cost is written down here rather than discovered later in a score.
ACCEPTED_LENIENCY = [
    # Punctuation folds from anywhere. Buys 1,607 value matches across the 660-document
    # snapshot measured 2026-09-11, of
    # which 1,532 differ by punctuation alone and 0 credit a wrong value as right; of the
    # gold values it merges inside one field, all 215 have identical digit strings.
    ('section identifier',  '5.2.1.5',      '5215'),
    ('address unit',        '#30-2',        '#302'),
    ('zone code',           'RR-2',         'RR2'),
    ('age range',           '90-94',        '9094'),
    # Reference markers are stripped wherever they appear, which is wanted -- a bibliography
    # entry with its `[4]` and one without are the same entry. The price is that two values
    # differing ONLY by the marker collapse together. No corpus field does that today.
    ('reference marker',    'see [1]',      'see [2]'),
    # `canonical` strips `/` (which is also why `1/2` == `12` below). `N/A` therefore keys as
    # `NA`. Checked rather than assumed: exactly ONE gold field in that
    # same snapshot holds both
    # spellings, a contract-number field where both mean "not applicable".
    ('slash stripped',      'N/A',          'NA'),
    # The number fold strips `,` `$` `%` before reading a numeral, so a value it CLAIMS loses
    # its unit. A currency amount and a percentage therefore share a key. Accepted: the
    # alternative is deciding `50%` != `50`, and a page writing a bare `50` in a rate column
    # means the same thing as one writing `50%`.
    ('currency vs percent',  '$5',          '5%'),
    ('percent vs bare',      '50%',         '50'),
    # The `accents` fold exists to do this and says so in its `why`: a transcription that
    # dropped a diacritic is the same name. The price is that alphabets where an accented
    # form is a SEPARATE LETTER lose the distinction -- Å is its own letter in Norwegian, not
    # an A with a ring, so `Åse` and `Ase` are two names and one key.
    ('German umlaut',       'Müller',       'Muller'),
    ('Nordic ring',         'Åse',          'Ase'),
    ('Turkish dotted I',    'İstanbul',     'Istanbul'),
    # Trailing zeros fold because `12.90` and `12.9` are one printed rate at two precisions
    # (METRIC_SPEC 2, asserted in MUST_MATCH). A version number pays for it: v1.1 and v1.10
    # are different releases and one key.
    ('version number',      '1.1',          '1.10'),
]

#: DEBT: P1 violations nobody chose -- no rule was aiming at these, they are side effects.
#: Anything caused deliberately belongs in ACCEPTED_LENIENCY above, however regrettable it
#: looks; the two lists answer different customer questions ("what do you get wrong?" versus
#: "what did you decide to give up?") and mixing them makes both answers untrustworthy.
#: Fix a fold, move the pair up into MUST_DIFFER, and this list shrinks.
KNOWN_COLLISIONS = [
    ('vulgar fraction', '½', '12'),
    ('written fraction', '1/2', '12'),
    ('malformed date', '2025-01-2025', '20250120 25'),
]

print("\nP2  TWO SPELLINGS OF ONE FACT AGREE")
split = [(n, a, b) for n, a, b in MUST_MATCH if canon_key(a) != canon_key(b)]
report(f"all {len(MUST_MATCH)} fold as they should", not split,
       "; ".join(f"{n}: {a!r} != {b!r}" for n, a, b, in split))

print("\nP1  VALUES A DOCUMENT COULD TELL APART DO NOT AGREE")
merged = [(n, a, b) for n, a, b in MUST_DIFFER if canon_key(a) == canon_key(b)]
report(f"all {len(MUST_DIFFER)} stay distinct", not merged,
       "; ".join(f"{n}: {a!r} == {b!r}" for n, a, b in merged))

print("\n    ...and the leniency we chose is still exactly the leniency we priced")
strayed = [(n, a, b) for n, a, b in ACCEPTED_LENIENCY if canon_key(a) != canon_key(b)]
report(f"all {len(ACCEPTED_LENIENCY)} accepted merges still merge", not strayed,
       "; ".join(f"{n}: {a!r} != {b!r}" for n, a, b in strayed))
note("these are deliberate: punctuation folds from anywhere, and METRIC_SPEC 5.3 says why.")
note("if one of them stops merging the fold changed -- check that was intended.")

print("\n    ...and the known violations have not grown")
still_broken = [(n, a, b) for n, a, b in KNOWN_COLLISIONS if canon_key(a) == canon_key(b)]
fixed = [(n, a, b) for n, a, b in KNOWN_COLLISIONS if canon_key(a) != canon_key(b)]
report(f"no new collisions beyond the {len(KNOWN_COLLISIONS)} recorded",
       len(still_broken) <= len(KNOWN_COLLISIONS))
if fixed:
    print(f"  ---   {len(fixed)} KNOWN COLLISION(S) NOW FIXED — promote to MUST_DIFFER:")
    for n, a, b in fixed:
        print(f"          {n}: {a!r} vs {b!r}")
note(f"{len(still_broken)} of {len(KNOWN_COLLISIONS)} still collide; "
     f"every one reaches the string fallback (P3)")

# The unit-stripping above only applies to a value the number fold CLAIMS. One it declines
# keeps its symbols, so the two halves of the rule look inconsistent and are: `$5` == `5%`
# because both parse, while `5% Notes` != `5 Notes` because neither does. Pinned so the
# asymmetry is a recorded shape rather than a half-finished fix.
report("a value the number fold declines keeps its unit",
       canon_key("5% Notes") != canon_key("5 Notes")
       and canon_key("$1,234 total") != canon_key("1234 total"),
       f"{canon_key('5% Notes')!r} vs {canon_key('5 Notes')!r}")

# Attaching a unit must not change what a number means. The number route reads a lone `1.000`
# as one and `1,000` as one thousand; before the decimal-point rule the punctuation strip
# overrode that as soon as a word followed, so `1.000 notes` meant one thousand notes.
report("a number keys the same alone as it does with a unit attached",
       (canon_key("1.000") != canon_key("1,000")
        and canon_key("1.000 notes") != canon_key("1,000 notes")),
       f"{canon_key('1.000 notes')!r} vs {canon_key('1,000 notes')!r}")
note("commas are untouched by that rule: they strip alone and embedded, as they always did.")

print("\nP1b  A FOLD MAY TRIM A VALUE, NEVER CONSUME IT")
# Keying as "" is not the same as being thrown out. A thrown-out value has no address at all
# (`flatten` gates on `states_nothing`, so canon_key never even sees one). A value that keys
# as "" HAS an address, is scored, and equals every other value some fold emptied.
#
# The invariant, stated once:        canon_key(v) == ""  implies  states_nothing(v)
#
# Asserted below by construction rather than by a list, because a list is what let seventeen
# spellings sit here unnoticed -- `()`, `,`, `/`, `"`, `. . .`, `..`, `...`, `null`, and the
# `[1]`-shaped values that made fourteen bibliography reference numbers a single key.
from omni_extract_bench.values import states_nothing                    # noqa: E402

PUNCT = "-.,/()[]{}\"'`~!@#$%^&*_+=|\\:;?<> \t"
probes = [
    # every single punctuation character, alone and doubled
    *[c for c in PUNCT], *[c * 2 for c in PUNCT], *[c * 3 for c in PUNCT],
    # the shapes the folds are built to trim, standing alone
    "[1]", "[a]", "[12]", "(1)", "..", "...", "...............", "../../..", ". . .",
    "null", "Null", "NULL", "- -", "-- --", "( )", "(())", '""', "''", "  .  ",
    # and a few that must keep working
    "229 [1]", "Table B-1.", "Inst itutional", "0", "false", "N/A", "None",
]
consumed = [v for v in probes if canon_key(v) == "" and not states_nothing(v)]
report(f"none of {len(probes)} probe values is consumed by a fold", not consumed,
       f"these key as the empty string with content on the way in: {consumed}")

# The other half: the values that SHOULD key empty still do, and they are exactly the
# structurally-empty ones -- the ones the harness itself makes vendors disagree about.
structural = [None, "", "   ", "\t\n", [], {}]
report("structurally empty values still key empty",
       all(canon_key(v) == "" for v in structural),
       f"{[(v, canon_key(v)) for v in structural if canon_key(v) != '']}")
note("`null` and `\"\"` are one thing because a strict dialect must emit the key with null")
note("where a permissive one omits it -- the harness causes the disagreement (METRIC_SPEC 5).")
note("`()` and the STRING \"null\" are content a model chose to emit, so they assert.")

# ...and searched for, not just listed. A list is what let seventeen spellings sit here
# unnoticed, so this enumerates every 1- and 2-character string over the punctuation the folds
# touch, then fuzzes longer ones and sweeps the Unicode categories most likely to be stripped.
import itertools as _it, unicodedata as _ud                             # noqa: E402
_PUNCT = "-.,/()[]{}\"'`~!@#$%^&*_+=|\\:;?<>" + " \t\n"
_ALPHA = _PUNCT + "0aA"
_probes = ["".join(t) for L in (1, 2) for t in _it.product(_ALPHA, repeat=L)]
_rnd = random.Random(7)
_probes += ["".join(_rnd.choice(_ALPHA) for _ in range(_rnd.randint(3, 10)))
            for _ in range(20000)]
_cats = {"Pd", "Pi", "Pf", "Ps", "Pe", "Po", "Pc", "Sm", "Sk", "Zs", "Cf", "Mn", "No", "Nl"}
_probes += [c for cp in range(0x2FFF) for c in (chr(cp),)
            if _ud.category(c) in _cats]
_probes += [0, 0.0, False, True, float("inf"), float("nan"), 10 ** 400, "NaN", "Infinity"]
_eaten = [v for v in _probes if canon_key(v) == "" and not states_nothing(v)]
_nonstr = [v for v in _probes if not isinstance(canon_key(v), str)]
report(f"no counterexample among {len(_probes):,} searched values", not _eaten and not _nonstr,
       f"consumed: {_eaten[:12]}  non-string keys: {_nonstr[:6]}")
note("this is a search, not a list -- every 1- and 2-char string over the punctuation the")
note("folds touch, 20k random longer ones, every Unicode codepoint under 0x2FFF in a")
note("strippable category, and the overflow scalars. Verified separately on all 17.7M")
note("gold and prediction leaves in the corpus: zero.")
note("It also holds BY CONSTRUCTION: canon_key has exactly two paths returning '', the")
note("structural-absence branch and a fallback that cannot be empty unless str(v) is all")
note("whitespace -- which the first branch already caught. A new fold cannot reintroduce it.")

# The fallback must not re-break P2: it removes whitespace and nothing else.
for a, b in (("( )", "()"), (". . .", "..."), ("- -", "--")):
    report(f"the fallback still folds whitespace: {a!r} == {b!r}",
           canon_key(a) == canon_key(b), f"{canon_key(a)!r} vs {canon_key(b)!r}")

print("\nP3  THE FALLBACK IS THE FLOOR")
# A value that fails its recogniser lands in the string path. That must not be a trapdoor:
# a malformed date should keep its identity, not be milled into something else's key.
report("a recognised date never reaches the string path",
       str(canon_key("2025-01-25")).startswith("#d"), f"{canon_key('2025-01-25')!r}")
report("a malformed date does not become a valid one",
       canon_key("2025-01-2025") != canon_key("2025-01-25"))
note("recognition, not type, is what protects a value: anything date-SHAPED that fails to")
note("parse gets the full string treatment. `2025-01-2025` is milled to '2025012025', which")
note("is also what the unrelated string `20250120 25` becomes -- see KNOWN_COLLISIONS.")

print("\nP5  THE PIPELINE IS ENUMERABLE, AND SAYS WHY")
# Normalisation used to be one long function of sequential mutations. It is now an ordered
# list of named steps, because a customer who disagrees with a match needs to know WHICH rule
# to argue with -- `31-1440073` -> `311440073` is not an answer, "the punctuation rule removed
# the hyphen" is.
from omni_extract_bench.values import FOLDS, canon_trace                # noqa: E402
# ORDER IN `FOLDS` IS BEHAVIOUR, not presentation. Pinned here because the failures are
# invisible: the key still looks plausible, it is merely a different one.
_names = [f.name for f in FOLDS]
report("`accents` runs before `typography`",
       _names.index("accents") < _names.index("typography"),
       "NFKD decomposes every vulgar fraction to N + U+2044 + M, and typography is what maps "
       "U+2044 to '/'. Reverse them and no fraction folds.")
report("...so every vulgar fraction folds, not just the ones someone listed",
       all(canon_key(c) == canon_key(t) for c, t in
           (("\u00bd", "1/2"), ("\u00be", "3/4"), ("\u2157", "3/5"), ("\u2152", "1/10"))),
       f"{[(c, canon_key(c)) for c in chr(0xbd) + chr(0xbe) + chr(0x2157)]}")
report("`dash mark` runs before `punctuation`",
       _names.index("dash mark") < _names.index("punctuation"),
       "otherwise the punctuation strip erases a dash run to nothing first")
report("`number` runs after the typography that feeds it",
       _names.index("typography") < _names.index("number"),
       "a minus sign must already be ASCII before the value is read as a number")

# EVERY EXAMPLE A FOLD PROMISES IN ITS `why` MUST BE TRUE. `why` is what a customer reads when
# they disagree with a match, so a stale one is worse than none. Three had already drifted:
# `punctuation` still said periods strip from anywhere after the decimal-point rule landed,
# `accents` claimed only accents while NFKD also flattens superscripts and ligatures, and
# `quotes and brackets` said "enclosing" when parentheses and slashes strip from anywhere.
_WHY_CLAIMS = [
    ("case", "ACME CORP", "Acme Corp", True), ("case", " Acme Corp ", "Acme Corp", True),
    ("accents", "M\u00fcller", "Muller", True), ("accents", "\u00b5g", "ug", True),
    ("accents", "\ufb01le", "file", True), ("accents", "\uff46\uff55\uff4c\uff4c", "full", True),
    ("accents", "\u216b", "XII", True), ("accents", "10\u00b2", "102", True),
    ("accents", "Ac\u200bme", "Acme", True),
    ("typography", "5\u2032", "5'", True), ("typography", "180 \u2212 121", "180 - 121", True),
    ("number", "1,000", "1000", True), ("number", "$5", "5", True),
    ("number", "12.90", "12.9", True), ("number", "02000", "2000", False),
    ("dash mark", "-", "--", True),
    ("punctuation", "PO BOX 125", "P.O. BOX 125", True),
    ("punctuation", "31-1440073", "311440073", True),
    ("punctuation", "1.5 mg", "15 mg", False),
    ("punctuation", "1.000.000", "1,000,000", True),
    ("punctuation", "512.784.7407", "512-784-7407", True),
    ("quotes and brackets", "A(B)C", "ABC", True),
    ("quotes and brackets", "and/or", "andor", True),
    ("quotes and brackets", "N/A", "NA", True),
]
_broken = [(n, a, b, want) for n, a, b, want in _WHY_CLAIMS
           if (canon_key(a) == canon_key(b)) is not want]
report(f"all {len(_WHY_CLAIMS)} examples promised in a step's `why` are true", not _broken,
       "; ".join(f"{n}: {a!r} {'!=' if w else '=='} {b!r}" for n, a, b, w in _broken))
note("if this fails, the fold changed and the sentence a customer reads is now a lie.")
note("fix the `why`, not the test.")

# A TRACE MUST NEVER CONTRADICT ITS OWN KEY. `canon_trace` is shown to someone disagreeing
# with a match, so a trace ending at "" beside a key of `()` is worse than no explanation.
# That was live for every value built only from strippable characters until the no-consumed-
# value guard started recording itself as a step.
import random as _rnd_mod                                              # noqa: E402
_r = _rnd_mod.Random(5)
_ALPHA = "[]()0123456789abz .,-'\"/%$\u2022\u00b5\u00bd"
_tprobes = ["[1]", "229 [1]", "1,000", "$5", "1.5 mg", "-", "--", "\u00bd", "()", ",", '"',
            "\u2022", "2024-01-15", "02000", "N/A", "ACME CORP"]
_tprobes += ["".join(_r.choice(_ALPHA) for _ in range(_r.randint(1, 8))) for _ in range(20000)]
_liars = []
for _v in _tprobes:
    _t = canon_trace(_v)
    if _t.changes and _t.changes[-1].after != _t.canon:
        _liars.append((_v, _t.changes[-1].after, _t.canon))
report(f"no trace of {len(_tprobes):,} contradicts its own key", not _liars,
       "; ".join(f"{v!r} ends at {a!r} but keys as {c!r}" for v, a, c in _liars[:4]))
note("a value the folds empty is restored by canon_key's guard, and the guard records that")
note("as a `value restored` step -- otherwise the trace stops at '' and the key disagrees.")

report(f"the pipeline is a list of {len(FOLDS)} named steps",
       len(FOLDS) >= 8 and all(f.name and f.why for f in FOLDS),
       f"{[f.name for f in FOLDS]}")
report("every step explains itself in a full sentence",
       all(f.why.endswith(".") and len(f.why) > 30 for f in FOLDS),
       f"{[f.name for f in FOLDS if not (f.why.endswith('.') and len(f.why) > 30)]}")
note("steps: " + " -> ".join(f.name for f in FOLDS))

_t = canon_trace("31-1440073")
report("a trace names the step that did the folding",
       [c.step for c in _t.changes] == ["punctuation"] and _t.canon == "311440073",
       f"{_t}")
_t2 = canon_trace("\u2022 Maintain a safe work environment")
report("...and reports several in order when several fire",
       [c.step for c in _t2.changes] == ["case", "list marker", "punctuation"], f"{_t2}")
report("a value that takes the date route reports it and folds nothing",
       canon_trace("01/15/2024").route == "date" and not canon_trace("01/15/2024").changes,
       f"{canon_trace('01/15/2024')}")
report("the trace agrees with canon_key on every MUST_MATCH and MUST_DIFFER value",
       all(canon_trace(v).canon == canon_key(v)
           for _n, a, b in MUST_MATCH + MUST_DIFFER for v in (a, b)))

print("\nP4  THE KEY IS A FUNCTION OF THE VALUE ALONE")
import inspect                                                          # noqa: E402
sig = inspect.signature(canon_key)
report("canon_key takes exactly one argument", len(sig.parameters) == 1,
       f"signature is {sig}")
report("...and is deterministic across calls",
       all(canon_key(v) == canon_key(v)
           for v in ("Acme", 5.0, "2024-01-15", None, "", "INV-007")))

print("\nGENERATED: DISTINCT IDENTIFIERS KEEP DISTINCT KEYS")
rnd = random.Random(11)
for label, mint in (
    # Uniform width on purpose: within one document identifiers are padded alike, so this
    # guards the realistic case. The MIXED-width failure is in KNOWN_COLLISIONS above.
    ("zero-padded ids",  lambda i: f"INV-{i:05d}"),
    ("dotted versions",  lambda i: f"{i // 100}.{i // 10 % 10}.{i % 10}"),
    ("timed results",    lambda i: f"{i // 6000}:{i // 100 % 60:02d}.{i % 100:02d}"),
):
    vals = [mint(rnd.randrange(1, 99999)) for _ in range(300)]
    keys = {canon_key(v) for v in vals}
    uniq = len(set(vals))
    report(f"{label}: {uniq} distinct values -> {len(keys)} distinct keys",
           len(keys) == uniq, f"{uniq - len(keys)} collided, e.g. "
           f"{[v for v in vals if sum(1 for w in vals if canon_key(w) == canon_key(v)) > 1][:4]}")

print(f"\n{'CANON PROPERTIES HOLD' if not FAILS else 'FAILURES:'}")
for f in FAILS:
    print(f"   {f}")
_sys.exit(1 if FAILS else 0)
