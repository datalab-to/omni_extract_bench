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

Auth: OPENROUTER_API_KEY, or OPENAI_API_KEY. Another OpenAI-compatible endpoint is selected
with `base_url=` (`--base-url`), which is an argument and not an environment variable, so the
record can say which endpoint answered.

    python -m omni_extract_bench.harness.providers.llm_single_shot --pdf doc.pdf --schema s.json --out out.json \\
        --model anthropic/claude-opus-5
"""
from __future__ import annotations

import base64
import dataclasses
import json
import os
import random
import time
from pathlib import Path

import openai

from ..dialects import parse_model_json

from ..extraction import (Budget, Cost, Extraction, MissingCredential, VendorError)
from ._cli import run_cli

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
DEFAULT_MAX_OUTPUT_TOKENS = 64000        # floor for models with no published ceiling

#: Published output ceiling per model. A shared floor would silently truncate the models with
#: bigger ones, so parity here is "as much as each model will give" rather than one number for
#: everyone. Lives beside the adapter that sends it, so a hand-run and the benchmark cannot
#: disagree about how much a model was allowed to say.
MODEL_MAX_OUTPUT = {
    "anthropic/claude-opus-5": 128000,
    "openai/gpt-5.6-sol": 128000,
    "openai/gpt-5.6-sol-pro": 128000,
    "google/gemini-3.7-flash": 65536,
}


@dataclasses.dataclass(frozen=True)
class Config:
    """What a raw-model leg can be asked. `model` has no default: it IS the provider name."""

    model: str = dataclasses.field(
        metadata={"help": "an OpenRouter org/model id"})
    max_output_tokens: int | None = dataclasses.field(
        default=None, metadata={"help": "default: the model's published ceiling"})
    base_url: str = DEFAULT_BASE_URL
    attempts: int = 3


def max_output_for(model: str) -> int:
    return MODEL_MAX_OUTPUT.get(model, DEFAULT_MAX_OUTPUT_TOKENS)


def extract(pdf: Path, schema: dict, *, timeout: float = 1800.0,
            config: Config) -> Extraction:
    """One completion with the schema as `response_format`, retried only for a bad ANSWER.

    The loop here raises the temperature after a reply that would not parse, was empty of
    choices, or was not an object -- it is trying to get a usable answer out of a model that
    gave a malformed one. Transport failures are NOT retried here: a rate limit or a 5xx
    becomes a `VendorError` carrying its status, and the harness's own retry decides, so the
    two loops cannot multiply into 12 attempts on one document.
    """
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise MissingCredential("OPENROUTER_API_KEY (or OPENAI_API_KEY) must be set")

    budget = Budget(timeout)
    model = config.model
    max_output_tokens = config.max_output_tokens or max_output_for(model)
    client = openai.OpenAI(base_url=config.base_url, api_key=key)
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

    temperature, last, calls = 0.0, None, []
    for attempt in range(config.attempts):
        if attempt:
            time.sleep(min((2 ** attempt) + random.uniform(0, 1), budget.remaining()))
        # Each attempt gets what is LEFT of the document's budget, not a fresh copy of it.
        # With the full timeout each, three attempts could spend three times what every other
        # vendor was given -- and nothing noticed while a subprocess kill capped it from
        # outside.
        budget.check(f"after {attempt} attempt(s) that produced no usable answer")
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "extraction", "schema": schema,
                                                 "strict": False}},
                max_tokens=max_output_tokens, temperature=temperature,
                timeout=budget.remaining(),
                extra_body={"usage": {"include": True}},   # real billing, not a token count
            )
        except openai.APIStatusError as exc:
            raise VendorError(f"{type(exc).__name__}: {exc}"[:300],
                              status=getattr(exc, "status_code", None)) from None
        except openai.APIError as exc:
            raise VendorError(f"{type(exc).__name__}: {exc}"[:300]) from None

        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
        calls.append({"usage": usage, "model": getattr(resp, "model", None)})
        choices = getattr(resp, "choices", None)
        if not choices:
            err = getattr(resp, "error", None) or {}
            last = str((err.get("message") if isinstance(err, dict) else err)
                       or "provider returned no choices")[:300]
            temperature = min(temperature + 0.1, 1.0)
            continue

        choice = choices[0]
        text = getattr(getattr(choice, "message", None), "content", None)
        if choice.finish_reason == "length":
            # Not retried, and not a transport problem: the answer does not FIT. A second
            # attempt cannot produce a shorter one, and would cost the same again.
            raise VendorError(f"truncated at max_tokens={max_output_tokens} "
                              f"({model}'s published ceiling)", status=200)
        if not text:
            raise VendorError("empty completion", status=200)
        # `parse_model_json`, not `json.loads`: some models wrap their answer in a ```json
        # fence, and a parser that accepts only bare JSON scores those at zero. One run
        # discarded 12 of 24 documents from the most accurate provider in the field over a
        # backtick. The helper existed for this and no adapter was calling it.
        parsed = parse_model_json(text)
        if parsed is None:
            last = f"not JSON, even allowing a code fence: {text[:120]!r}"
            calls[-1]["text"] = text          # keep it: a parser fix re-reads instead of re-paying
            temperature = min(temperature + 0.1, 1.0)
            continue
        if isinstance(parsed, dict) and parsed:
            return Extraction(
                result=parsed,
                # The ceiling is recorded because a truncated answer cannot be read without
                # it: "finish_reason: length" means nothing unless you know what the limit was.
                raw={"calls": calls, "finish_reason": choice.finish_reason,
                     "max_output_tokens": max_output_tokens, "model": model},
                cost=Cost.reported(usage.get("cost"), "usage.cost",
                                   tokens_in=usage.get("prompt_tokens"),
                                   tokens_out=usage.get("completion_tokens")))
        last = f"model returned {type(parsed).__name__}, not an object"

    # The body carries every completion, text included, so an answer that no parser could read
    # is still on disk in `record["raw"]` -- which is what this module's docstring promises and
    # what makes a parser bug a re-read rather than a re-payment.
    raise VendorError(f"all {config.attempts} attempts returned an unusable answer: {last}",
                      status=200, body=json.dumps({"calls": calls, "model": model})[:4000])


def main() -> None:
    run_cli(extract, Config, "llm-single-shot")


if __name__ == "__main__":
    main()
