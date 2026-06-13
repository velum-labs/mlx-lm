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
- `mlx_lm/generate.py` — one upstream bugfix in `GenerationBatch.filter`:
  stale per-sequence sampler/logits-processor slots were left behind when a
  batch with none set drained, misaligning the processors of sequences
  inserted later. On the server this silently disabled the constraint of a
  structured request that followed a plain request on the same
  `BatchGenerator`.
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

## OpenAI-compatible server additions

This fork keeps the stock launch contract intact:

```sh
python -m mlx_lm server --model mlx-community/Qwen2.5-0.5B-Instruct-4bit --host 127.0.0.1 --port 8080
```

The additions are opt-in and additive.

### Embeddings

Start the chat server with a separate embedding model:

```sh
python -m mlx_lm server \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --embedding-model mlx-community/Qwen2.5-0.5B-Instruct-4bit
```

`POST /v1/embeddings` accepts OpenAI's string or string-array input shape:

```sh
curl -s http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen2.5-0.5B-Instruct-4bit","input":["hello","world"]}'
```

The response is:

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.0]}
  ],
  "model": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
  "usage": {"prompt_tokens": 2, "total_tokens": 2}
}
```

The embedding model is lazy-loaded on the first embeddings request and is held
separately from the chat model. If `--embedding-model` is omitted,
`/v1/embeddings` returns HTTP 400 with a clear error instead of affecting chat
routes. `GET /v1/models` includes the configured embedding model id.

The embedding implementation uses the configured MLX model's inner transformer
when available and mean-pools the final token hidden states.

### Tool Calling

`POST /v1/chat/completions` accepts OpenAI `tools` and `tool_choice`:

- `tool_choice: "none"` suppresses tools for the request.
- `tool_choice: "auto"` preserves the model/template-native tool-calling path.
- `tool_choice: "required"` or a forced function choice enables structured
  tool-call mode. With `mlx-lm[structured]` installed, generation is constrained
  to a JSON object whose `arguments` match the selected tool's JSON Schema.

Non-streaming tool-call responses use OpenAI's shape:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_...",
        "type": "function",
        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

Streaming responses emit SSE lines with `delta.tool_calls[].index`, a stable
`id`, `function.name`, and `function.arguments`, then finish with
`finish_reason: "tool_calls"` and `data: [DONE]`.

Smoke test:

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"default_model",
    "stream":true,
    "stream_options":{"include_usage":true},
    "messages":[{"role":"user","content":"Weather in Paris?"}],
    "tools":[{
      "type":"function",
      "function":{
        "name":"get_weather",
        "description":"Get weather",
        "parameters":{
          "type":"object",
          "properties":{"city":{"type":"string"}},
          "required":["city"],
          "additionalProperties":false
        }
      }
    }],
    "tool_choice":{"type":"function","function":{"name":"get_weather"}}
  }'
```

### Streaming Usage

For streamed chat completions, pass:

```json
{"stream": true, "stream_options": {"include_usage": true}}
```

The server emits a final chunk before `[DONE]` with:

```json
{"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
```

This matches the fields consumed by OpenAI-compatible clients such as the AI SDK
OpenAI-compatible provider.
