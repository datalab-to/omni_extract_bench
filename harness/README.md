# Harness: generating predictions

The scorer in `omni_extract_bench/` grades predictions. This package produces them, which is
the half a reader cannot otherwise check: a scorer can only be fair about what it is handed, so
the rules deciding what each vendor is handed are here, in one code path, for every provider.

```bash
pip install -e ".[harness]"

# one provider over the whole dataset
python -m harness.run_provider datalab --data-root data/ --out-root predictions/

# one subset, five documents, to try it out
python -m harness.run_provider gemini --data-root data/ --out-root predictions/ \
    --subsets internal --max-docs 5

# then score what it produced
omni-extract-bench leaderboard --pred-root predictions/ --data-root data/ --workers 8
```

`--data-root` is the dataset layout published on HuggingFace:
`<root>/<subset>/<doc>/{document.pdf,schema.json,ground_truth.json}`. Predictions are written to
`<out-root>/<provider>/<subset>/<doc>.json`, which is what the scorer's `--pred-root` expects.

## What is uniform

Every rule below is applied by the same code to every provider. They exist because each one was,
at some point, the thing deciding a vendor's score.

| rule | where |
| --- | --- |
| one timeout per document, identical for all (`--timeout`, default 1800s) | `run_provider.DEFAULT_TIMEOUT` |
| each provider at its maximum tier, and each model at its published output ceiling | `PROVIDER_TIER`, `MODEL_MAX_OUTPUT` |
| the same schema, with benchmark-only annotations removed | `strip_bench_keys` |
| conventions the gold follows are written into field descriptions for everyone | `schema_overlay.apply_overlay` |
| dialect translation allowed (inline `$ref`, collapse nullable unions, alias reserved keys); content changes are not | `deref`, per-adapter |
| transient failures (429, 5xx, empty 200) retry; real answers (400s) do not | `is_transient`, `with_transient_retry` |
| the full HTTP exchange, job ids, the schema sent and vendor usage are written before parsing | `run_one` |
| an account-level failure (402) stops the run rather than storing zeros | `is_account_failure` |

Disable the conventions overlay with `--no-overlay` if you want the schemas exactly as the
upstream corpora ship them; the published run used it, so leaving it on reproduces those numbers.

## Providers

| provider | adapter | credentials |
| --- | --- | --- |
| `datalab`, `datalab-accurate` | `providers/datalab.py` | `DATALAB_API_KEY` |
| `reducto` | `providers/reducto.py` (MIT, micro1) | `REDUCTO_API_KEY` |
| `llamaextract` | `providers/llamaextract.py` (MIT, micro1) | `LLAMA_CLOUD_API_KEY` |
| `extend` | `providers/extend.py` | `EXTEND_API_KEY`, `EXTEND_WORKSPACE_ID` |
| `mistral` | `providers/mistral.py` | `MISTRAL_API_KEY` |
| `azure-cu` | `providers/azure_cu.py` | `AZURE_CU_ENDPOINT`, `AZURE_CU_KEY` |
| `claude`, `gpt`, `gpt-pro`, `gemini` | `providers/llm_single_shot.py` | `OPENROUTER_API_KEY` |

Each adapter also runs standalone on one document, which is the quickest way to check a key:

```bash
python -m harness.providers.datalab --pdf doc.pdf --schema s.json --out out.json --mode balanced
```

Two provider notes that belong beside any score:

* **azure-cu**: the `completion` deployment (`--completion-model`) is a deployment choice, not a
  product tier. `gpt-4.1-mini` and `gpt-4.1` are different systems behind one API. Its field
  types are also a smaller set than JSON Schema's, so the schema it receives is a lossier
  statement of the task than other vendors get. Both facts should be published, not assumed.
* **LLM legs**: billing comes from OpenRouter's `usage.cost`, which is what was actually charged,
  including retries. Other vendors bill out of band and the `usd` column stays null rather than
  holding an estimate — a guessed cost beside a measured one, in the same column, is how a cost
  comparison becomes fiction.

## What each run records

`<out-root>/<provider>/<subset>/<doc>.json` holds `{result, _secs}`. Beside it,
`_raw/<subset>/<doc>.json` holds the evidence:

* `http` — every request and response (URL with query stripped, status, headers, body, timing),
  captured at the transport, including from adapters that run as subprocesses
* `job_ids` — vendor job identifiers, which make a timed-out run recoverable instead of re-payable
* `schema_sent` — the exact schema the vendor received, after overlay and dialect translation
* `cost` — vendor-reported billing where it exists, `billed_out_of_band: true` where it does not
* `llm_calls` — every completion pre-parse, so an unparseable answer can be salvaged from the
  record rather than re-paid for
* `run_manifest` — timeout, tier, model, output ceiling, whether the overlay applied, capture time

## Resuming, and timeouts

Re-running the same command skips documents that already hold a usable prediction and retries
the ones holding a transient failure. A stored vendor answer (a context-length rejection, a
schema 400) is never retried: it is a real result.

A document that hit the shared deadline while the vendor was still working is marked
`__timeout__`. If the job later completes on the vendor's side, it can be collected without
paying again — and the rule applies to every provider that has a captured job id, not only the
one whose number you care about:

```bash
python -m harness.recover_timeouts --out-root predictions/          # dry run
python -m harness.recover_timeouts --out-root predictions/ --apply
```

Recovered documents are marked `recovered_after_timeout`. Latency is a real product property, so
publish the recovery count beside the score rather than quietly folding it in.

## Licence

`providers/reducto.py` and `providers/llamaextract.py` derive from micro1's `longextract_bench`
(MIT) — see `providers/LICENSE-micro1`. Everything else is Apache-2.0 with the rest of the repo.
