# Structured decoding (velum-labs fork)

This fork extends the upstream `mlx_lm.server` with constrained/structured
decoding, built on [outlines-core](https://github.com/dottxt-ai/outlines-core).
The output is constrained by masking logits with a compiled FSM so the model
can only emit tokens that keep the output valid.

The delta over the upstream tag is intentionally small and self-contained:

- `mlx_lm/structured/` — the constraint machinery: request parsing, JSON
  schema/regex/choice compilation with per-(model, constraint) caching, and
  the per-request logits processor.
- `mlx_lm/server.py` — a few dozen lines of optional hooks: parse the
  structured request parameters (HTTP 400 on malformed/uncompilable ones),
  carry the constraint on `LogitsProcessorArguments` (picklable, so it
  reaches every distributed rank), and append the constraint processor in
  `_make_logits_processors`.
- `setup.py` — the `structured` extra (`outlines-core`, exact pin) and the
  new subpackage.

Without the `structured` extra installed, the server behaves exactly like
upstream (the parameters are ignored and a warning is logged), which keeps
the fork trivially rebaseable.

## Usage

```sh
pip install "mlx-lm[structured] @ git+https://github.com/velum-labs/mlx-lm@structured-0.31.3"
mlx_lm.server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

Request parameters on `/v1/chat/completions` and `/v1/completions`:

| Field | Form | Meaning |
| --- | --- | --- |
| `response_format` | `{"type": "json_schema", "json_schema": {"schema": {...}}}` | Output is valid JSON matching the schema (OpenAI structured outputs) |
| `response_format` | `{"type": "json_object"}` | Output is a valid JSON object |
| `guided_json` | schema dict or JSON string | vLLM-style alias for a JSON schema constraint |
| `guided_regex` | regex string | Output matches the regex |
| `guided_choice` | list of strings | Output is exactly one of the choices |

At most one constraint per request; malformed or uncompilable constraints
are rejected with HTTP 400 at request time. Constrained requests work with
streaming, with continuous batching, and with speculative decoding (the
processor rolls its FSM back when draft tokens are rejected).

## Caveats

- The constraint applies from the first generated token. For reasoning
  ("thinking") models, disable thinking for constrained requests (e.g.
  `"chat_template_kwargs": {"enable_thinking": false}`).
- The first request for a new schema compiles a regex/FSM index (~0.1 s for
  small schemas); compiled indexes are cached per (model, constraint) and
  the tokenizer vocabulary once per model.
- Regex features outside the FSM subset (lookarounds, backreferences) and
  unsupported JSON schema constructs are rejected with HTTP 400.

## Maintaining the fork

Branches are a small patch series per adopted upstream tag
(`structured-<tag>`). To adopt a newer upstream mlx-lm: branch from the new
tag, cherry-pick the structured commits, resolve any drift in
`mlx_lm/server.py` (request parsing, `LogitsProcessorArguments`,
`_make_logits_processors`), and run the tests:

```sh
python -m unittest tests/test_structured.py -v
```

The spec-parsing tests run anywhere; FSM/processor/server-hook tests skip
unless `outlines-core` (the `structured` extra) plus `numpy` and `numba`
(test-only, for the CPU kernel) are installed.
