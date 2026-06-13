# HTTP Model Server

You use `mlx-lm` to make an HTTP API for generating text with any supported
model. The HTTP API is intended to be similar to the [OpenAI chat
API](https://platform.openai.com/docs/api-reference).

> [!NOTE]  
> The MLX LM server is not recommended for production as it only implements
> basic security checks.

Start the server with: 

```shell
mlx_lm.server --model <path_to_model_or_hf_repo>
```

For example:

```shell
mlx_lm.server --model mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

This will start a text generation server on port `8080` of the `localhost`
using Mistral 7B instruct. The model will be downloaded from the provided
Hugging Face repo if it is not already in the local cache.

To see a full list of options run:

```shell
mlx_lm.server --help
```

You can make a request to the model by running:

```shell
curl localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

### Request Fields

- `messages`: An array of message objects representing the conversation
  history. Each message object should have a role (e.g. user, assistant) and
  content (the message text).

- `role_mapping`: (Optional) A dictionary to customize the role prefixes in
  the generated prompt. If not provided, the default mappings are used.

- `stop`: (Optional) An array of strings or a single string. These are
  sequences of tokens on which the generation should stop.

- `max_tokens`: (Optional) An integer specifying the maximum number of tokens
  to generate. Defaults to `512`.

- `stream`: (Optional) A boolean indicating if the response should be
  streamed. If true, responses are sent as they are generated. Defaults to
  false.

- `temperature`: (Optional) A float specifying the sampling temperature.
  Defaults to `0.0`.

- `top_p`: (Optional) A float specifying the nucleus sampling parameter.
  Defaults to `1.0`.

- `top_k`: (Optional) An integer specifying the top-k sampling parameter.
  Defaults to `0` (disabled).

- `min_p`: (Optional) A float specifying the min-p sampling parameter.
  Defaults to `0.0` (disabled).

- `repetition_penalty`: (Optional) Applies a multiplicative penalty to repeated
  tokens. Defaults to `0.0` (disabled).

- `repetition_context_size`: (Optional) The size of the context window for
  applying repetition penalty. Defaults to `20`.

- `presence_penalty`: (Optional) Applies an additive penalty to tokens
  that appeared before. Defaults to `0.0` (disabled).

- `presence_context_size`: (Optional) The size of the context window for
  applying presence penalty. Defaults to `20`.

- `frequency_penalty`: (Optional) Applies an additive penalty proportional to
  how many times a token appeared previously. Defaults to `0.0` (disabled).

- `frequency_context_size`: (Optional) The size of the context window for
  applying frequency penalty. Defaults to `20`.

- `logit_bias`: (Optional) A dictionary mapping token IDs to their bias
  values. Defaults to `None`.

- `logprobs`: (Optional) An integer specifying the number of top tokens and
  corresponding log probabilities to return for each output in the generated
  sequence. If set, this can be any value between 1 and 10, inclusive.

- `model`: (Optional) A string path to a local model or Hugging Face repo id.
  If the path is local is must be relative to the directory the server was
  started in.

- `adapters`: (Optional) A string path to low-rank adapters. The path must be
  relative to the directory the server was started in.

- `draft_model`: (Optional) Specifies a smaller model to use for speculative
  decoding. Set to `null` to unload.

- `num_draft_tokens`: (Optional) The number of draft tokens the draft model
  should predict at once. Defaults to `3`.

### Response Fields

- `id`: A unique identifier for the chat.

- `system_fingerprint`: A unique identifier for the system.

- `object`: Any of "chat.completion", "chat.completion.chunk" (for
  streaming), or "text.completion".

- `model`: The model repo or path (e.g. `"mlx-community/Llama-3.2-3B-Instruct-4bit"`).

- `created`: A time-stamp for when the request was processed.

- `choices`: A list of outputs. Each output is a dictionary containing the fields:
    - `index`: The index in the list.
    - `logprobs`: A dictionary containing the fields:
        - `token_logprobs`: A list of the log probabilities for the generated
          tokens.
        - `tokens`: A list of the generated token ids.
        - `top_logprobs`: A list of lists. Each list contains the `logprobs`
          top tokens (if requested) with their corresponding probabilities.
    - `finish_reason`: The reason the completion ended. This can be either of
      `"stop"` or `"length"`.
    - `message`: The text response from the model.

- `usage`: A dictionary containing the fields:
    - `prompt_tokens`: The number of prompt tokens processed.
    - `completion_tokens`: The number of tokens generated.
    - `total_tokens`: The total number of tokens, i.e. the sum of the above two fields.

### OpenAI Chat Compatibility

The server accepts OpenAI-style `tools` and `tool_choice` on
`/v1/chat/completions`:

- `tools`: An optional array of function tools shaped as
  `{"type": "function", "function": {...}}`.
- `tool_choice: "none"`: Do not use tools for this request.
- `tool_choice: "auto"` or omitted: Preserve the model and chat-template native
  behavior.
- `tool_choice: "required"`: Force the model to return a tool call using one of
  the provided tools.
- `tool_choice: {"type": "function", "function": {"name": "..."}}`: Force a
  specific function tool.

Forced tool choice requires installing `mlx-lm[structured]`. In forced mode the
server constrains generation to a JSON tool-call object whose `arguments` match
the selected function's JSON Schema. Invalid generated tool JSON returns HTTP
400 before a streaming response starts.

Non-streaming forced tool responses set `choices[0].message.content` to `null`,
populate `choices[0].message.tool_calls`, and use
`finish_reason: "tool_calls"`. Streaming forced tool responses emit
`choices[0].delta.tool_calls` with a stable `index`, `id`,
`function.name`, and JSON-string `function.arguments`, then finish with
`finish_reason: "tool_calls"`.

For streamed chat completions, `stream_options: {"include_usage": true}` emits
one final chunk before `[DONE]` with an empty `choices` array and a populated
`usage` object.

### Embeddings

Start the server with a separate embedding model to enable
`POST /v1/embeddings`:

```shell
mlx_lm.server \
  --model mlx-community/Mistral-7B-Instruct-v0.3-4bit \
  --embedding-model mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ
```

The embedding model is lazy-loaded on the first embeddings request and is held
separately from the chat model. Embedding inference runs under a dedicated lock,
so concurrent HTTP requests cannot race during lazy loading or reuse the same
MLX model instance unsafely.

Supported request fields:

- `model`: Required. Must be the configured `--embedding-model` value or
  `default_model`.
- `input`: Required. A string or an array of strings.
- `encoding_format`: Optional. Only `"float"` is supported.

Explicitly rejected request shapes:

- Token-array inputs such as `[1, 2, 3]` or nested token arrays.
- Empty input arrays.
- `encoding_format` values other than `"float"`, including `"base64"`.
- `dimensions`, because this server does not truncate or project embeddings.
- Any other embedding request field not listed above.

The response uses the OpenAI embeddings shape:

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.0]}
  ],
  "model": "<configured-embedding-model>",
  "usage": {"prompt_tokens": 1, "total_tokens": 1}
}
```

Embeddings require an embedding-suitable model that is also supported by
`mlx_lm`'s model registry. For example, Qwen embedding models work through the
same MLX loader, while BERT-style sentence-transformer repos are not supported
unless their model type is implemented in `mlx_lm`. The provider accepts either
pooled embedding output or token hidden states. Token hidden states are pooled
with an attention-mask mean and L2-normalized before being returned.

### List Models

Use the `v1/models` endpoint to list available models:

```shell
curl localhost:8080/v1/models -H "Content-Type: application/json"
```

This will return a list of locally available models where each model in the
list contains the following fields:

- `id`: The Hugging Face repo id.
- `created`: A time-stamp representing the model creation time.

When `--embedding-model` is configured, `v1/models` also lists that model id.
