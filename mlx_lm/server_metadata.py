"""Import-safe metadata helpers for the HTTP server."""

import datetime
from typing import Any, Dict, List, Optional

from ._version import __version__
from .model_fusion_contracts import (
    MODEL_FUSION_SCHEMA_BUNDLE_HASH,
    validate_model_endpoint_fixture,
)


def utc_now() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def resolve_model_id(cli_args: Any) -> str:
    return getattr(cli_args, "model", None) or "default_model"


def loaded_model_status(response_generator: Any) -> str:
    model_provider = getattr(response_generator, "model_provider", None)
    if model_provider is None:
        return "unknown"
    if getattr(model_provider, "model", None) is None:
        return "not_loaded"
    return "loaded"


def make_health_response(
    *,
    model: str,
    mlx_version: str,
    platform: str,
    loaded_model_status: str,
    commit: str = "unknown",
    version: str = __version__,
) -> Dict[str, Any]:
    return {
        "status": "ok",
        "provider": "mlx-lm",
        "version": version,
        "model": model,
        "mlx_version": mlx_version,
        "platform": platform,
        "commit": commit,
        "loaded_model_status": loaded_model_status,
    }


def make_server_capabilities(
    *,
    structured_output_available: bool,
    embedding_model: Optional[str],
) -> Dict[str, str]:
    return {
        "chat_completions": "supported",
        "text_completions": "supported",
        "streaming": "supported",
        "tool_calls": "supported" if structured_output_available else "degraded",
        "structured_output": (
            "supported" if structured_output_available else "unsupported"
        ),
        "embeddings": "supported" if embedding_model is not None else "unsupported",
    }


def make_endpoint_support(capabilities: Dict[str, str]) -> Dict[str, str]:
    return {
        "/v1/chat/completions": "supported",
        "/chat/completions": "supported",
        "/v1/completions": "supported",
        "/v1/embeddings": capabilities["embeddings"],
        "/v1/models": "supported",
        "/v1/health": "supported",
        "/v1/capabilities": "supported",
    }


def make_model_endpoint_fixture(
    endpoint_id: str,
    model: str,
    *,
    base_url: Optional[str] = None,
    owner: str = "mlx-lm",
    provider: str = "mlx-lm",
    api_compatibility: str = "mlx-lm-server",
    capabilities: Optional[Dict[str, str]] = None,
    status: str = "succeeded",
    created_at: Optional[str] = None,
    producer_git_sha: str = "0" * 40,
    max_context_tokens: Optional[int] = None,
    estimated_memory_gb: Optional[float] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build and validate a local provider contract endpoint fixture."""
    record: Dict[str, Any] = {
        "schema": "model_endpoint.v1",
        "schema_version": "v1",
        "schema_bundle_hash": MODEL_FUSION_SCHEMA_BUNDLE_HASH,
        "producer": "mlx-lm",
        "producer_version": __version__,
        "producer_git_sha": producer_git_sha,
        "created_at": created_at or utc_now(),
        "endpoint_id": endpoint_id,
        "owner": owner,
        "provider": provider,
        "model": model,
        "api_compatibility": api_compatibility,
        "capabilities": (
            capabilities if capabilities is not None else {"chat_completions": "supported"}
        ),
        "status": status,
    }
    if base_url is not None:
        record["base_url"] = base_url
    if max_context_tokens is not None:
        record["max_context_tokens"] = max_context_tokens
    if estimated_memory_gb is not None:
        record["estimated_memory_gb"] = estimated_memory_gb
    if tags is not None:
        record["tags"] = tags

    validate_model_endpoint_fixture(record)
    return record


def make_capabilities_response(
    *,
    model: str,
    base_url: Optional[str],
    structured_output_available: bool,
    embedding_model: Optional[str],
    max_output_tokens: Optional[int],
    version: str = __version__,
) -> Dict[str, Any]:
    capabilities = make_server_capabilities(
        structured_output_available=structured_output_available,
        embedding_model=embedding_model,
    )
    endpoint = make_model_endpoint_fixture(
        endpoint_id=f"mlx-lm:{model}",
        model=model,
        base_url=base_url,
        api_compatibility="mlx-lm-server",
        capabilities=capabilities,
    )
    return {
        "object": "capabilities",
        "provider": "mlx-lm",
        "version": version,
        "model": model,
        "status": "succeeded",
        "capabilities": capabilities,
        "endpoints": make_endpoint_support(capabilities),
        "limits": {
            "max_context_tokens": None,
            "max_output_tokens": max_output_tokens,
        },
        "model_info": {
            "estimated_memory_gb": None,
            "quantization": "unknown",
        },
        "endpoint": endpoint,
        "schema": endpoint["schema"],
    }
