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

    oeb predict --provider anthropic/claude-opus-5 --doc doc.pdf --schema schema.json

    The model IS the provider name -- `--options {"model": ...}` is refused, so a run
    cannot be stored under a model it did not use.
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

from ..responses import parse_model_json

from ..budget import Budget

from ..contract import Cost, Extraction

from ..errors import MissingCredential, VendorError

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

    def __post_init__(self):
        """Resolve `None` to the model's published ceiling HERE, not in `extract`.

        `extract` used to do it, so the record wrote down `max_output_tokens: null` while the
        model was handed 128000 -- a setting stated by neither the Config nor the manifest,
        which is the whole thing `settings` exists to rule out. The parity rule is per-model
        ("as much as each will give"), so what that came to has to be on the document."""
        if self.max_output_tokens is None:
            object.__setattr__(self, "max_output_tokens", max_output_for(self.model))


def max_output_for(model: str) -> int:
    return MODEL_MAX_OUTPUT.get(model, DEFAULT_MAX_OUTPUT_TOKENS)


def prepare_schema(schema: dict) -> dict:
    """This vendor takes a JSON Schema as written; nothing to reshape."""
    return schema


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
    max_output_tokens = config.max_output_tokens
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
        budget.check(f"after {attempt} attempt(s) that produced no usable answer")
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                response_format={"type": "json_schema",
                                 "json_schema": {"name": "extraction", "schema": schema,
                                                 "strict": False}},
                max_tokens=max_output_tokens, temperature=temperature,
                timeout=budget.remaining(),
                extra_body={"usage": {"include": True}},
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
            raise VendorError(f"truncated at max_tokens={max_output_tokens} "
                              f"({model}'s published ceiling)", status=200)
        if not text:
            raise VendorError("empty completion", status=200)
        parsed = parse_model_json(text)
        if parsed is None:
            last = f"not JSON, even allowing a code fence: {text[:120]!r}"
            calls[-1]["text"] = text
            temperature = min(temperature + 0.1, 1.0)
            continue
        if isinstance(parsed, dict) and parsed:
            return Extraction(
                result=parsed,
                raw={"calls": calls, "finish_reason": choice.finish_reason,
                     "max_output_tokens": max_output_tokens, "model": model},
                cost=Cost.reported(usage.get("cost"), "usage.cost",
                                   tokens_in=usage.get("prompt_tokens"),
                                   tokens_out=usage.get("completion_tokens")))
        last = f"model returned {type(parsed).__name__}, not an object"

    raise VendorError(f"all {config.attempts} attempts returned an unusable answer: {last}",
                      status=200, body=json.dumps({"calls": calls, "model": model})[:4000])

