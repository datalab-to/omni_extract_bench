"""Single-shot LLM extraction: the PDF and the schema go to the model in one call, through any
OpenAI-compatible endpoint (OpenRouter by default, the only one that reports real billing, `usage.cost`).

Auth: OPENROUTER_API_KEY, or OPENAI_API_KEY. Another endpoint is chosen with the `base_url` option.

    oeb run --system anthropic/claude-opus-5

The model is the provider name on the command line; in code it is the `model` option,
`LLMSingleShot(model="anthropic/claude-opus-5")`, and it names the run folder (`label`).
"""
from __future__ import annotations

import base64
import dataclasses
import json
import os
import random
import time
from pathlib import Path
from typing import ClassVar

import openai
from omni_bench_core import adapter, errors

from .. import contract, responses

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


def max_output_for(model: str) -> int:
    return MODEL_MAX_OUTPUT.get(model, DEFAULT_MAX_OUTPUT_TOKENS)


@dataclasses.dataclass(frozen=True)
class LLMSingleShot(contract.Extractor):
    """A raw model, sent the PDF and the schema in one completion, through any OpenAI-compatible endpoint.

    One class stands for every model, so the model names the run folder (`label`). `max_output_tokens`
    of None is resolved to the model's published ceiling in `__post_init__`, so the record states the
    number the model was handed, not `null`."""

    name: ClassVar[str] = "llm_single_shot"
    keys: ClassVar[tuple[str, ...]] = ("OPENROUTER_API_KEY",)
    workers: ClassVar[int] = 5
    model: str = dataclasses.field(metadata={"help": "an OpenRouter org/model id"})
    max_output_tokens: int | None = dataclasses.field(
        default=None, metadata={"help": "default: the model's published ceiling"})
    base_url: str = DEFAULT_BASE_URL
    attempts: int = 3

    def __post_init__(self):
        if self.max_output_tokens is None:
            object.__setattr__(self, "max_output_tokens", max_output_for(self.model))

    def missing_keys(self) -> list[str]:
        """OPENAI_API_KEY stands in for OPENROUTER_API_KEY, for a `base_url` that is OpenAI's own."""
        return [] if os.environ.get("OPENAI_API_KEY") else super().missing_keys()

    @property
    def label(self) -> str:
        return self.model.replace("/", "__")

    def extract(self, pdf: Path, schema: dict, *, timeout: float) -> adapter.Call:
        """One completion with the schema as `response_format`, retried only for a bad answer.

        The loop here raises the temperature after a reply that would not parse, had no choices, or was not an
        object: it tries to get a usable answer out of a model that gave a malformed one. Transport failures
        are not retried here: a rate limit or a 5xx becomes a `SystemFailed` with its status, and omni-bench-core's
        retry decides, so the two loops can't multiply into 12 attempts on one document.

        Three cases are handled on purpose, because getting them wrong decides a vendor's score:

        - `finish_reason == "length"` is not retried: the answer didn't fit in `max_tokens`, and a retry at
          the same limit can't fit it either. Retrying it three times once cost $7.35 on one document;
        - `choices` can be None when the provider behind the endpoint failed, with the real message in a
          top-level `error`. That message is raised, so the failure says what went wrong; and
        - an answer that isn't JSON is retried at a slightly higher temperature, but the model's text is kept
          in the record either way, so a fenced or broken answer can be read again rather than paid for again.
        """
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ["OPENAI_API_KEY"]

        deadline = adapter.Budget(timeout)
        model = self.model
        max_output_tokens = self.max_output_tokens
        client = openai.OpenAI(base_url=self.base_url, api_key=key)
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

        temperature, last, calls, costs = 0.0, None, [], []
        for attempt in range(self.attempts):
            if attempt:
                time.sleep(min((2 ** attempt) + random.uniform(0, 1), deadline.remaining()))
            deadline.check(f"after {attempt} attempt(s) that produced no usable answer")
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages,
                    response_format={"type": "json_schema",
                                     "json_schema": {"name": "extraction", "schema": schema,
                                                     "strict": False}},
                    max_tokens=max_output_tokens, temperature=temperature,
                    timeout=deadline.remaining(),
                    extra_body={"usage": {"include": True}},
                )
            except openai.APIStatusError as exc:
                raise errors.SystemFailed(f"{type(exc).__name__}: {exc}",
                                          status=getattr(exc, "status_code", None)) from exc
            except openai.APITimeoutError as exc:
                raise errors.SystemTimeout(f"{type(exc).__name__}: {exc}") from exc
            except openai.APIConnectionError as exc:
                raise errors.SystemUnreachable(f"{type(exc).__name__}: {exc}") from exc
            except openai.APIError as exc:
                raise errors.SystemFailed(f"{type(exc).__name__}: {exc}") from exc

            with adapter.reading("the completion's usage"):
                usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
                calls.append({"usage": usage, "model": getattr(resp, "model", None)})
                # Note: every completion is billed, a malformed one too, so the cost is every completion's.
                costs.append(adapter.Cost.reported(usage.get("cost"), "usage.cost",
                                                   tokens_in=usage.get("prompt_tokens"),
                                                   tokens_out=usage.get("completion_tokens")))
            choices = getattr(resp, "choices", None)
            if not choices:
                err = getattr(resp, "error", None) or {}
                last = str((err.get("message") if isinstance(err, dict) else err)
                           or "provider returned no choices")
                temperature = min(temperature + 0.1, 1.0)
                continue

            with adapter.reading("the completion's text"):
                choice = choices[0]
                text = getattr(getattr(choice, "message", None), "content", None)
                finish = getattr(choice, "finish_reason", None)
            if finish == "length":
                raise errors.SystemFailed(f"truncated at max_tokens={max_output_tokens} "
                                          f"({model}'s published ceiling)", status=200, cost=adapter.Cost.total(costs))
            if not text:
                raise errors.SystemFailed("empty completion", status=200, cost=adapter.Cost.total(costs))
            parsed = responses.parse_model_json(text)
            if parsed is None:
                last = f"not JSON, even allowing a code fence: {text!r}"
                calls[-1]["text"] = text
                temperature = min(temperature + 0.1, 1.0)
                continue
            if isinstance(parsed, dict) and parsed:
                return adapter.Call(
                    raw={"extraction": parsed, "calls": calls, "finish_reason": finish,
                         "max_output_tokens": max_output_tokens, "model": model},
                    cost=adapter.Cost.total(costs))
            last = f"model returned {type(parsed).__name__}, not an object"

        raise errors.SystemFailed(f"all {self.attempts} attempts returned an unusable answer: {last}",
                                  status=200, body=json.dumps({"calls": calls, "model": model}),
                                  cost=adapter.Cost.total(costs))
