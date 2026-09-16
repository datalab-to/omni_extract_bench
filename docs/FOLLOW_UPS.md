# Follow-ups

Things measured and deliberately not built, so the next person starts from the numbers rather
than from the argument.

## Compose batches by weight, not by row count

`run` fills each batch with a fixed number of rows. On a corpus with a heavy tail that is the
wrong unit, because one row can be the whole run.

**Measured** on the 620-document corpus (datalab predictions, one machine, 8 workers,
`seconds` read from the scores table this run now records):

    736s of work      median document 4.8ms      slowest 189s      spread 315,066x
    the five slowest documents are 73% of all the work

Simulating schedules against those real times, 8 workers pulling batches dynamically:

| batching | makespan |
|---|---|
| by count, 19/batch (today's default here) | 346s — 1.83x the floor |
| by count, 1/batch | 223s |
| **by bytes of ground truth, 4 MB budget** | **219s — 1.16x** |
| oracle, packed by measured time | 189s |

The floor is 189s: one document takes that alone and its arrays cannot be split, so no
schedule beats it. Perfect balance would be 92s and is not reachable.

**What predicts cost** (Spearman against measured seconds, all 620):

| proxy | ρ | |
|---|---|---|
| addresses from a previous run | +0.964 | needs a previous run |
| ground-truth file size | +0.956 | one stat, or a `gt_bytes` column |
| longest array in the gold | +0.882 | needs a parse |
| prediction file size | +0.841 | one stat |
| schema size | **+0.159** | free, and useless |

The schema is free because it is already inline in the manifest, and it does not work: it
describes a document's *width* and the cost is driven by its *height*.

**The shape to build.** Fill a batch until it holds ~4 MB of ground truth, with the row count
as a cap so a corpus of tiny documents still splits across workers. It needs no global sort,
no up-front pass and no total, so it still streams. The budget does not want tuning: 0.5 MB
through 8 MB all land at 1.09–1.17x the floor.

**One trap.** Sorting big-first and then batching by count is the *worst* option measured,
577s — the first batch gets every monster. Sorting only helps if the batches are composed by
weight afterwards.

Until then, `--batch-size 1` is the blunt version and already gets 346s to 223s.

## Memoise `canon_key`

Profiling one 56 KB document: 0.257s total, of which `grade` is 0.243s and **canonicalising
values is 0.200s** across 11,146 calls — the assignment problem is 0.070s. Across ten
documents, **69% of leaf values are repeats** (33,818 values, 10,423 distinct), because tables
repeat dates, codes and totals.

A memo looks like roughly a 2x win on the metric. Not done: the cache key has to handle
unhashable leaves, and the win should be measured on a real corpus rather than inferred from a
dedup ratio.
