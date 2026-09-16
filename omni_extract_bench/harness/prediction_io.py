"""What counts as a USABLE prediction. One definition, shared by every scoring path.

There were two. `report.pred_for` treated an empty `{}` as "produced nothing" (so the document
counted as not attempted), while the finalize path handed `{}` to the grader, which scored it
0.0 and counted it as ATTEMPTED. Same output, two different meanings, depending only on which
provider produced it.

The headline is unaffected -- an unusable prediction scores 0 either way -- but the coverage
column and the score-on-attempted view are exactly the numbers that separate "processes every
document badly" from "silently fails on hard documents", and they were wrong. gemini showed
100% coverage on longarray while 37 of its 45 outputs were `{"result": {}}`.
"""


def usable(result):
    """True when `result` is a prediction that can be scored on its merits."""
    return bool(isinstance(result, dict) and result and "__error__" not in result)
