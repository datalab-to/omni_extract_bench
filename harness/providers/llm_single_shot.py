"""Single-shot LLM extraction: the PDF and the schema go to the model in one call.

Used for the raw-model legs (Claude, GPT, Gemini) through any OpenAI-compatible endpoint;
the published runs used OpenRouter, which is also the only endpoint that returns real billing
(`usage.cost`) rather than a token count we would have to price ourselves.

Three behaviours here exist because getting them wrong decides a vendor's score:

  * `finish_reason == "length"` is NOT retried. The answer did not fit in `max_tokens`, and a
    retry at the same cap cannot fit a shorter one. Retrying truncation three times cost $7.35
    on a single document in an early run and produced nothing.
  * `choices` can be None when the UPSTREAM provider errored, with the real message in a
    top-level `error` field. Indexing `choices[0]` then raises an opaque TypeError that stands
    in for the vendor's own error, which is what makes a failure attributable.
  * a JSON parse failure is retried at a slightly higher temperature, but the model's text is
    returned to the caller either way, so a fenced or otherwise unparseable answer can be
    salvaged from the record rather than re-paid for.

Auth: OPENROUTER_API_KEY (or OEB_LLM_API_KEY with --base-url for another endpoint).

    python -m harness.providers.llm_single_shot --pdf doc.pdf --schema s.json --out out.json \\
        --model anthropic/claude-opus-5
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from pathlib import Path

from .envelope import write_output

SYSTEM_PROMPT = """\
You are a document data extraction system. You will be given a PDF document \
and a JSON schema describing the fields to extract.

Instructions:

- Read the field descriptions carefully — they specify what to extract and how.
- Review the entire document before extracting. Information for a single field \
may span multiple pages.
- When a value appears directly in the document, copy it exactly as written. \
Do not reformat dates, normalize capitalization, or paraphrase.
- When a value must be derived or inferred (e.g. counting items, computing \
totals, evaluating a condition), reason carefully and return the derived value.
- For array fields, extract every matching item. Do not truncate or summarize.
- For numeric fields, return numbers not strings. For booleans, return \
true/false.
- Use null when the document does not contain the information needed for a \
field. Use an empty array [] only when the category applies but has zero items.

Return a single JSON object conforming to the schema."""

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MAX_OUTPUT_TOKENS = 64000


class LLMSingleShot:
    def __init__(self, *, api_key: str, model: str, base_url: str = DEFAULT_BASE_URL,
                 max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS, timeout: float = 1800,
                 max_retries: int = 3):
        import openai
        self._openai = openai
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.calls: list[dict] = []          # every completion, pre-parse

    def __call__(self, pdf: Path, schema: dict):
        b64 = base64.b64encode(pdf.read_bytes()).decode("ascii")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "file", "file": {"filename": pdf.name,
                                          "file_data": f"data:application/pdf;base64,{b64}"}},
                {"type": "text", "text": ("Extract the data from this document according to the "
                                          "following JSON schema:\n\n"
                                          f"```json\n{json.dumps(schema, indent=2)}\n```")},
            ]},
        ]
        temp, last = 0.0, None
        for attempt in range(self.max_retries):
            if attempt:
                time.sleep((2 ** attempt) + random.uniform(0, 1))
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "extraction", "schema": schema,
                                                     "strict": False}},
                    max_tokens=self.max_output_tokens, temperature=temp, timeout=self.timeout,
                    extra_body={"usage": {"include": True}},   # real billing, not a token count
                )
            except (self._openai.RateLimitError, self._openai.APIStatusError) as e:
                last = f"{type(e).__name__}: {e}"[:300]; temp = min(temp + 0.1, 1.0); continue
            except Exception as e:  # noqa: BLE001
                return {"__error__": f"{type(e).__name__}: {e}"[:300]}

            choices = getattr(resp, "choices", None)
            usage = resp.usage.model_dump() if getattr(resp, "usage", None) else None
            if not choices:
                err = getattr(resp, "error", None) or {}
                msg = err.get("message") if isinstance(err, dict) else str(err)
                last = str(msg or "provider returned no choices")[:300]
                self.calls.append({"error": last, "usage": usage})
                temp = min(temp + 0.1, 1.0); continue

            ch = choices[0]
            text = getattr(getattr(ch, "message", None), "content", None)
            self.calls.append({"finish_reason": getattr(ch, "finish_reason", None),
                               "content": text, "usage": usage,
                               "cost_usd": (usage or {}).get("cost"),
                               "model": getattr(resp, "model", None)})
            if ch.finish_reason == "length":
                return {"__error__": (f"truncated at max_tokens={self.max_output_tokens}; not "
                                      f"retried (a retry cannot fit a shorter answer)")}
            if not text:
                return {"__error__": "empty completion"}
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                last = f"JSON parse error: {e}"[:200]; temp = min(temp + 0.1, 1.0); continue
            if isinstance(parsed, dict) and parsed:
                return parsed
            last = f"model returned {type(parsed).__name__}, not an object"
        return {"__error__": f"all {self.max_retries} attempts failed: {last}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=os.environ.get("OEB_LLM_BASE_URL", DEFAULT_BASE_URL))
    ap.add_argument("--max-output-tokens", type=int,
                    default=int(os.environ.get("OEB_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)))
    ap.add_argument("--timeout", type=float, default=float(os.environ.get("OEB_TIMEOUT", 1800)))
    a = ap.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OEB_LLM_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY (or OEB_LLM_API_KEY) must be set")
    p = LLMSingleShot(api_key=key, model=a.model, base_url=a.base_url,
                      max_output_tokens=a.max_output_tokens, timeout=a.timeout)
    t0 = time.time()
    result = p(a.pdf, json.loads(a.schema.read_text()))
    write_output(a.out, provider=a.model, result=result, latency_s=time.time() - t0,
                 usage={"model": a.model, "max_output_tokens": a.max_output_tokens,
                        "calls": p.calls})
    if isinstance(result, dict) and "__error__" in result:
        print(result["__error__"], file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
