"""Per-request context carriers for the Agent App generation pipeline.

Fork extension: allow the embedded webapp to pass a whitelisted subset of the
chat ``inputs`` payload (e.g. an auth token) into the Agent App soul prompt,
using agent-v1 style ``{{key}}`` placeholder substitution.

How it works
------------
``app_generator.generate`` calls :func:`snapshot_runtime_inputs` with the raw
request inputs BEFORE ``contextvars.copy_context()``; the generation worker
thread re-applies that snapshot (see ``libs.flask_utils.preserve_flask_contexts``),
and ``runtime_request_builder`` calls :func:`render_runtime_inputs` when
composing the soul prompt.

Whitelist
---------
Keys are whitelisted via the ``AGENT_APP_PROMPT_INPUT_KEYS`` environment
variable (comma-separated). Unset/empty (default) captures nothing, so
:func:`render_runtime_inputs` returns the prompt unchanged. Values are taken
from the raw inputs, i.e. before the ``user_input_form`` filtering in
``_prepare_user_inputs``.

Substitution
------------
Only whitelisted keys are substituted, matching the exact ``{{key}}`` form
(same double-brace syntax as legacy agent v1 prompts; no whitespace inside the
braces). ``{{#...#}}`` workflow markers never match. Placeholders whose key is
not whitelisted are left untouched so misconfiguration stays visible.

Referer fallback
----------------
The stock webapp only forwards URL inputs that are declared as form variables,
so an undeclared key (e.g. auth_code) never reaches the request body. To avoid
shipping a custom web build, whitelisted keys missing from ``inputs`` are
recovered from the ``Referer`` header, which preserves the embedded iframe URL
(e.g. ``http://host/agent/<token>?auth_code=xxx``). The Referer is only trusted
when its host matches the request ``Host`` header (case-insensitive, default
ports normalized); otherwise it is ignored.

Note: ask_human resume turns (``resume_after_form_submission``) run in a
background task with empty inputs, so their placeholders are not substituted.
"""

import json
import os
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any
from urllib.parse import parse_qsl, urlsplit

AGENT_APP_PROMPT_INPUT_KEYS_ENV = "AGENT_APP_PROMPT_INPUT_KEYS"

agent_app_inputs_var: ContextVar[Mapping[str, Any]] = ContextVar("agent_app_inputs", default={})


def _whitelisted_keys() -> frozenset[str]:
    raw = os.environ.get(AGENT_APP_PROMPT_INPUT_KEYS_ENV, "")
    return frozenset(key.strip() for key in raw.split(",") if key.strip())


def _normalize_host(host: str) -> str:
    host = host.strip().lower().rstrip(".")
    if host.endswith(":80") or host.endswith(":443"):
        return host.rsplit(":", 1)[0]
    return host


def _inputs_from_referer() -> Mapping[str, Any]:
    """Recover inputs from the request ``Referer`` header (iframe URL).

    The Referer is only trusted when its host matches the current request's
    ``Host`` header (see :func:`_normalize_host`); a mismatched or absent
    Referer yields an empty mapping.
    """
    try:
        from flask import request

        referer = request.headers.get("Referer")
        if not referer:
            return {}
        parsed = urlsplit(referer)
        if not parsed.query:
            return {}
        if _normalize_host(parsed.netloc) != _normalize_host(request.host):
            return {}
        return dict(parse_qsl(parsed.query, keep_blank_values=True))
    except RuntimeError:
        # Outside a Flask request context (background/test calls): no fallback.
        return {}


def snapshot_runtime_inputs(inputs: Mapping[str, Any]) -> None:
    """Capture whitelisted raw inputs into the ContextVar for this run.

    Must be called before ``contextvars.copy_context()`` so the generation
    worker thread sees the values. Whitelisted keys missing from ``inputs``
    fall back to the host-validated request Referer. With an empty whitelist
    (default) the var is reset to an empty mapping and nothing reaches the
    soul prompt.
    """
    keys = _whitelisted_keys()
    if not keys:
        agent_app_inputs_var.set({})
        return
    captured = {key: inputs[key] for key in sorted(keys) if key in inputs}
    if len(captured) < len(keys):
        # Fork: fall back to the host-validated request Referer (the embedded
        # iframe URL) for whitelisted keys the request body does not carry.
        referer_inputs = _inputs_from_referer()
        for key in sorted(keys):
            if key not in captured and key in referer_inputs:
                captured[key] = referer_inputs[key]
    agent_app_inputs_var.set(captured)


def _value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def render_runtime_inputs(prompt: str) -> str:
    """Substitute ``{{key}}`` placeholders in the soul prompt with the
    whitelisted inputs captured by :func:`snapshot_runtime_inputs`.

    Non-whitelisted / unmatched placeholders are left untouched; with no
    captured inputs the prompt is returned as-is.
    """
    runtime_inputs = agent_app_inputs_var.get()
    if not runtime_inputs:
        return prompt
    for key, value in runtime_inputs.items():
        prompt = prompt.replace("{{" + key + "}}", _value_to_text(value))
    return prompt
