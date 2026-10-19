"""Field matching: a prediction and its gold extraction compared value by value, each value addressed by its
key path, with rows paired by content.

It works in stages, one module each:

    documents.py       what is compared: vendor wrappers and sidecar fields stripped, the schema read, and each
                       document flattened into addresses, one per value
    pairing.py         which predicted row goes with which gold row (with `matching.py`, the assignment solver)
    compare.py         one document compared with its gold: a verdict per address, and the document's numbers
    field_matching.py  a dataset: each document's gold read and checked, every document's numbers summarized,
                       and `FieldMatching`, the scorer as omni-bench-core grades a run

`values.py` says whether two values are equal, and `recognise.py` whether a text is a date or a number.

Every module but `field_matching.py` works on its own, with nothing from omni-bench-core or the rest of oeb but
the `UsageError` a malformed gold raises. The public functions are imported here, so
`field_matching.score(pred, gold, schema)` scores one prediction.
"""
from . import compare, documents, matching, pairing, recognise, values
from .compare import Verdict, score
from .documents import INDEX, KEY, format_node, is_open_map, node_key, resolve_refs, show, unwrap_schema
from .field_matching import FieldMatching, Gold, GoldError, check, gold, summarize
from .pairing import align

__all__ = ["INDEX", "KEY", "FieldMatching", "Gold", "GoldError", "Verdict", "align", "check", "compare",
           "documents", "format_node", "gold", "is_open_map", "matching", "node_key", "pairing", "recognise",
           "resolve_refs", "score", "show", "summarize", "unwrap_schema", "values"]
