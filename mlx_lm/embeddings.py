"""Embedding request parsing and lazy MLX embedding model execution."""

from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx


class EmbeddingError(ValueError):
    """Base class for embedding request and model errors."""


class EmbeddingNotConfiguredError(EmbeddingError):
    """The server was not started with an embedding model."""


class EmbeddingModelUnavailableError(EmbeddingError):
    """The request selected an embedding model that is not served."""


class EmbeddingModelError(RuntimeError):
    """The configured embedding model could not produce embeddings."""


@dataclass(frozen=True)
class EmbeddingRequest:
    model: str
    inputs: List[str]


def parse_embedding_request(
    body: Any, configured_model: Optional[str]
) -> EmbeddingRequest:
    if not isinstance(body, dict):
        raise EmbeddingError("embeddings request body must be a JSON object")

    unsupported = sorted(set(body) - {"encoding_format", "input", "model"})
    if unsupported:
        fields = ", ".join(repr(field) for field in unsupported)
        raise EmbeddingError(f"unsupported embeddings request fields: {fields}")

    encoding_format = body.get("encoding_format", "float")
    if encoding_format != "float":
        raise EmbeddingError("'encoding_format' must be 'float'")

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise EmbeddingError("'model' must be a non-empty string")
    if configured_model is not None and model not in (configured_model, "default_model"):
        raise EmbeddingModelUnavailableError(
            f"Embedding model {model!r} is not available; "
            f"configured embedding model is {configured_model!r}"
        )

    inputs = body.get("input")
    if isinstance(inputs, str):
        input_list = [inputs]
    elif isinstance(inputs, list) and all(isinstance(item, str) for item in inputs):
        if not inputs:
            raise EmbeddingError("'input' must not be empty")
        input_list = inputs
    else:
        raise EmbeddingError("'input' must be a string or an array of strings")

    return EmbeddingRequest(model=configured_model or model, inputs=input_list)


def _l2_normalize(values: Any) -> Any:
    norm = mx.sqrt(mx.sum(values * values, axis=-1, keepdims=True))
    return values / mx.maximum(norm, mx.array(1e-12, dtype=values.dtype))


def masked_mean_pool(hidden: Any, mask: Any, normalize: bool = True) -> Any:
    hidden = hidden.astype(mx.float32)
    mask = mask.astype(mx.float32)
    mask = mx.expand_dims(mask, -1)
    pooled = mx.sum(hidden * mask, axis=1) / mx.maximum(
        mx.sum(mask, axis=1), mx.array(1e-12, dtype=mx.float32)
    )
    return _l2_normalize(pooled) if normalize else pooled


def _hidden_states(output: Any) -> Any:
    if hasattr(output, "text_embeds"):
        return output.text_embeds
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, tuple):
        return output[0]
    return output


class EmbeddingProvider:
    def __init__(
        self,
        model_id: Optional[str],
        tokenizer_config: Dict[str, Any],
        trust_remote_code: bool,
        is_distributed: bool,
        load_fn: Callable,
    ):
        self.model_id = model_id
        self._tokenizer_config = tokenizer_config
        self._trust_remote_code = trust_remote_code
        self._is_distributed = is_distributed
        self._load_fn = load_fn
        self._lock = Lock()
        self._model_key = None
        self._model = None
        self._tokenizer = None

    def _load(self):
        if not self.model_id:
            raise EmbeddingNotConfiguredError(
                "No embedding model configured; start the server with "
                "--embedding-model <hf-id-or-path> to enable /v1/embeddings"
            )
        if self._is_distributed:
            raise EmbeddingModelError("/v1/embeddings is not supported in distributed mode")
        if self._model_key != self.model_id:
            try:
                model, tokenizer = self._load_fn(
                    self.model_id,
                    tokenizer_config=self._tokenizer_config,
                    trust_remote_code=self._trust_remote_code,
                )
            except Exception as e:
                raise EmbeddingModelError(str(e)) from e
            self._model_key = self.model_id
            self._model = model
            self._tokenizer = tokenizer
        return self._model, self._tokenizer

    def embed(self, inputs: List[str]) -> Tuple[List[List[float]], int]:
        with self._lock:
            model, tokenizer = self._load()
            embeddings = []
            total_tokens = 0

            for text in inputs:
                tokens = tokenizer.encode(text, add_special_tokens=True)
                if not tokens:
                    if tokenizer.eos_token_id is None:
                        raise EmbeddingModelError("embedding tokenizer produced no tokens")
                    tokens = [tokenizer.eos_token_id]
                total_tokens += len(tokens)

                input_ids = mx.array([tokens])
                attention_mask = mx.ones(input_ids.shape, dtype=mx.float32)
                encoder = getattr(model, "model", model)
                hidden = _hidden_states(encoder(input_ids))
                if len(hidden.shape) == 2:
                    pooled = _l2_normalize(hidden.astype(mx.float32))[0]
                elif len(hidden.shape) == 3:
                    pooled = masked_mean_pool(hidden, attention_mask)[0]
                else:
                    raise EmbeddingModelError(
                        "embedding model must return pooled embeddings or token "
                        "hidden states"
                    )
                embeddings.append(pooled.tolist())

            return embeddings, total_tokens
