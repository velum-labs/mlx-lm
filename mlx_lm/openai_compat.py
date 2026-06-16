"""OpenAI compatibility helpers for the HTTP server."""

import datetime
import json
import logging
import re
import uuid
import urllib.parse
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .model_fusion_protocol import (
    MODEL_FUSION_PERSISTED_RECORDS,
    MODEL_FUSION_SCHEMA_BUNDLE_HASH,
)

_SHA256_PATTERN = re.compile(r"^sha256:[a-f0-9]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[a-f0-9]{40}$")

_CONTRACT_METADATA_FIELDS = (
    "schema",
    "schema_version",
    "schema_bundle_hash",
    "producer",
    "producer_version",
    "producer_git_sha",
    "created_at",
)
_STATUS_VALUES = {
    "pending",
    "running",
    "succeeded",
    "failed",
    "canceled",
    "requires_action",
    "skipped",
    "unsupported",
}
_OWNER_VALUES = {"fusionkit", "handoffkit", "cursorkit", "mlx-lm", "benchmark", "external"}
_API_COMPATIBILITY_VALUES = {
    "openai-chat-completions",
    "openai-responses",
    "mlx-lm-server",
    "custom",
}
_CAPABILITY_STATUS_VALUES = {"supported", "unsupported", "degraded", "unknown"}
_CHAT_ROLE_VALUES = {"system", "user", "assistant", "tool"}
_SIDE_EFFECT_VALUES = {
    "none",
    "read_only",
    "writes_workspace",
    "network",
    "tool_execution",
    "unknown",
}
_ERROR_KIND_VALUES = {
    "none",
    "provider_error",
    "validation_error",
    "timeout",
    "rate_limited",
    "tool_denied",
    "secret_denied",
    "capability_missing",
    "internal_error",
}

_MODEL_ENDPOINT_FIELDS = _CONTRACT_METADATA_FIELDS + (
    "endpoint_id",
    "owner",
    "provider",
    "model",
    "base_url",
    "api_compatibility",
    "capabilities",
    "max_context_tokens",
    "estimated_memory_gb",
    "tags",
    "status",
)
_MODEL_CALL_RECORD_FIELDS = _CONTRACT_METADATA_FIELDS + (
    "call_id",
    "endpoint_id",
    "provider_request_id",
    "model",
    "request_hash",
    "response_hash",
    "messages",
    "status",
    "side_effects",
    "started_at",
    "finished_at",
    "latency_ms",
    "usage",
    "output_text",
    "error",
    "metadata",
)


def validate_model_fusion_contract_fixture(
    value: Dict[str, Any], expected_schema: Optional[str] = None
) -> Dict[str, Any]:
    """Validate local model-fusion provider fixtures without cross-repo imports."""
    record = _require_object(value, expected_schema or "model-fusion contract")
    schema_name = expected_schema or record.get("schema")
    if schema_name not in MODEL_FUSION_PERSISTED_RECORDS:
        raise ValueError(f"Unsupported model-fusion fixture schema: {schema_name!r}")
    if schema_name == "model_endpoint.v1":
        return validate_model_endpoint_fixture(record)
    if schema_name == "model-call-record.v1":
        return validate_model_call_record_fixture(record)
    raise ValueError(f"Unsupported model-fusion fixture schema: {schema_name!r}")


def validate_model_endpoint_fixture(value: Dict[str, Any]) -> Dict[str, Any]:
    record = _require_object(value, "model_endpoint.v1")
    _validate_allowed_keys(record, _MODEL_ENDPOINT_FIELDS, "model_endpoint.v1")
    _validate_metadata(record, "model_endpoint.v1")
    _require_non_empty_string(record, "endpoint_id")
    _require_enum(record, "owner", _OWNER_VALUES)
    _require_non_empty_string(record, "provider")
    _require_non_empty_string(record, "model")
    _require_enum(record, "api_compatibility", _API_COMPATIBILITY_VALUES)
    _validate_capabilities(record.get("capabilities"))
    _require_enum(record, "status", _STATUS_VALUES)
    _validate_optional_uri(record, "base_url")
    _validate_optional_int(record, "max_context_tokens", minimum=1)
    _validate_optional_number(record, "estimated_memory_gb", minimum=0)
    _validate_optional_string_list(record, "tags")
    return record


def validate_model_call_record_fixture(value: Dict[str, Any]) -> Dict[str, Any]:
    record = _require_object(value, "model-call-record.v1")
    _validate_allowed_keys(record, _MODEL_CALL_RECORD_FIELDS, "model-call-record.v1")
    _validate_metadata(record, "model-call-record.v1")
    _require_non_empty_string(record, "call_id")
    _require_non_empty_string(record, "endpoint_id")
    _validate_optional_string(record, "provider_request_id")
    _require_non_empty_string(record, "model")
    _validate_hash(record, "request_hash")
    _validate_optional_hash(record, "response_hash")
    _validate_chat_messages(record.get("messages"))
    _require_enum(record, "status", _STATUS_VALUES)
    _require_enum(record, "side_effects", _SIDE_EFFECT_VALUES)
    _validate_datetime(record, "started_at")
    _validate_optional_datetime(record, "finished_at")
    _validate_optional_number(record, "latency_ms", minimum=0)
    _validate_optional_usage(record, "usage")
    _validate_optional_string(record, "output_text")
    _validate_optional_error(record, "error")
    if "metadata" in record and not isinstance(record["metadata"], dict):
        raise ValueError("metadata must be an object")
    return record


def _require_object(value: Any, context: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be a JSON object")
    return value


def _validate_allowed_keys(
    value: Dict[str, Any], allowed_keys: Sequence[str], context: str
) -> None:
    allowed = set(allowed_keys)
    extra = sorted(set(value) - allowed)
    if extra:
        raise ValueError(f"{context} contains unsupported fields: {', '.join(extra)}")


def _validate_metadata(value: Dict[str, Any], expected_schema: str) -> None:
    _require_keys(value, _CONTRACT_METADATA_FIELDS, expected_schema)
    if value["schema"] != expected_schema:
        raise ValueError(f"schema must be {expected_schema}")
    if value["schema_version"] != "v1":
        raise ValueError("schema_version must be v1")
    _validate_hash(value, "schema_bundle_hash")
    if value["schema_bundle_hash"] != MODEL_FUSION_SCHEMA_BUNDLE_HASH:
        raise ValueError(
            "schema_bundle_hash does not match the bundled model-fusion contract"
        )
    _require_non_empty_string(value, "producer")
    _require_non_empty_string(value, "producer_version")
    if not isinstance(value["producer_git_sha"], str) or not _GIT_SHA_PATTERN.match(
        value["producer_git_sha"]
    ):
        raise ValueError("producer_git_sha must be a 40-character lowercase git SHA")
    _validate_datetime(value, "created_at")


def _require_keys(value: Dict[str, Any], keys: Iterable[str], context: str) -> None:
    missing = [key for key in keys if key not in value]
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")


def _require_non_empty_string(value: Dict[str, Any], key: str) -> None:
    if not isinstance(value.get(key), str) or not value[key]:
        raise ValueError(f"{key} must be a non-empty string")


def _validate_optional_string(value: Dict[str, Any], key: str) -> None:
    if key in value and not isinstance(value[key], str):
        raise ValueError(f"{key} must be a string")


def _require_enum(value: Dict[str, Any], key: str, allowed: Iterable[str]) -> None:
    if value.get(key) not in allowed:
        raise ValueError(f"{key} has unsupported value: {value.get(key)!r}")


def _validate_hash(value: Dict[str, Any], key: str) -> None:
    if not isinstance(value.get(key), str) or not _SHA256_PATTERN.match(value[key]):
        raise ValueError(f"{key} must be sha256:<64 lowercase hex chars>")


def _validate_optional_hash(value: Dict[str, Any], key: str) -> None:
    if key in value:
        _validate_hash(value, key)


def _validate_datetime(value: Dict[str, Any], key: str) -> None:
    raw = value.get(key)
    if not isinstance(raw, str):
        raise ValueError(f"{key} must be an RFC 3339 date-time string")
    try:
        datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as e:
        raise ValueError(f"{key} must be an RFC 3339 date-time string") from e


def _validate_optional_datetime(value: Dict[str, Any], key: str) -> None:
    if key in value:
        _validate_datetime(value, key)


def _validate_optional_uri(value: Dict[str, Any], key: str) -> None:
    if key not in value:
        return
    raw = value[key]
    if not isinstance(raw, str):
        raise ValueError(f"{key} must be a URI string")
    parsed = urllib.parse.urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"{key} must be an absolute URI")


def _validate_optional_number(
    value: Dict[str, Any], key: str, minimum: Optional[float] = None
) -> None:
    if key not in value:
        return
    raw = value[key]
    if isinstance(raw, bool) or not isinstance(raw, (float, int)):
        raise ValueError(f"{key} must be a number")
    if minimum is not None and raw < minimum:
        raise ValueError(f"{key} must be at least {minimum}")


def _validate_optional_int(
    value: Dict[str, Any], key: str, minimum: Optional[int] = None
) -> None:
    if key not in value:
        return
    raw = value[key]
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{key} must be an integer")
    if minimum is not None and raw < minimum:
        raise ValueError(f"{key} must be at least {minimum}")


def _validate_optional_string_list(value: Dict[str, Any], key: str) -> None:
    if key not in value:
        return
    raw = value[key]
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ValueError(f"{key} must be an array of strings")


def _validate_capabilities(value: Any) -> None:
    if not isinstance(value, dict):
        raise ValueError("capabilities must be an object")
    for capability, status in value.items():
        if not isinstance(capability, str):
            raise ValueError("capability names must be strings")
        if status not in _CAPABILITY_STATUS_VALUES:
            raise ValueError(f"capability {capability!r} has unsupported status")


def _validate_chat_messages(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a non-empty array")
    for index, message in enumerate(value):
        context = f"messages[{index}]"
        _require_object(message, context)
        _validate_allowed_keys(message, ("role", "content"), context)
        if message.get("role") not in _CHAT_ROLE_VALUES:
            raise ValueError(f"{context}.role has unsupported value")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"{context}.content must be a string")


def _validate_optional_usage(value: Dict[str, Any], key: str) -> None:
    if key not in value:
        return
    usage = _require_object(value[key], key)
    _validate_allowed_keys(
        usage, ("prompt_tokens", "completion_tokens", "total_tokens"), key
    )
    for usage_key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        _validate_optional_int(usage, usage_key, minimum=0)


def _validate_optional_error(value: Dict[str, Any], key: str) -> None:
    if key not in value:
        return
    error = _require_object(value[key], key)
    _validate_allowed_keys(error, ("kind", "message", "retryable"), key)
    _require_keys(error, ("kind",), key)
    if error.get("kind") not in _ERROR_KIND_VALUES:
        raise ValueError("error.kind has unsupported value")
    _validate_optional_string(error, "message")
    if "retryable" in error and not isinstance(error["retryable"], bool):
        raise ValueError("error.retryable must be a boolean")


class ToolCallFormatter:
    def __init__(
        self, tool_parser: Callable, tools: Optional[List[Any]], streaming=False
    ):
        self._idx = 0
        self._tool_parser = tool_parser
        self._tools = tools
        self._streaming = streaming

    def _format(self, tc: Dict[str, Any], index: Optional[int] = None):
        tc_id = tc.get("id") or f"call_{uuid.uuid4().hex}"
        arguments = tc.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        out = {
            "function": {
                "name": tc["name"],
                "arguments": arguments,
            },
            "type": "function",
            "id": tc_id,
        }
        if self._streaming:
            out["index"] = self._idx if index is None else index
            if index is None:
                self._idx += 1
        return out

    def format_parsed(self, tool_calls: List[Dict[str, Any]]):
        return [self._format(tc, index=i) for i, tc in enumerate(tool_calls)]

    def __call__(self, tool_calls: List[str]):
        if not tool_calls:
            return []

        result = []
        for tool_text in tool_calls:
            try:
                parsed = self._tool_parser(tool_text, self._tools)
            except (ValueError, json.JSONDecodeError) as e:
                logging.warning(
                    f"Failed to parse tool call ({type(e).__name__}: {e}) - "
                    f"tool text was likely truncated mid-generation."
                )
                continue
            if not isinstance(parsed, list):
                parsed = [parsed]
            result.extend(self._format(tc) for tc in parsed)
        return result


def tool_function(tool: Dict[str, Any]):
    if not isinstance(tool, dict) or tool.get("type") != "function":
        raise ValueError("tools entries must be {'type': 'function', 'function': {...}}")
    function = tool.get("function")
    if not isinstance(function, dict):
        raise ValueError("function tool entries must include a function object")
    name = function.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("function tools require a non-empty function.name")
    parameters = function.get("parameters") or {"type": "object"}
    if not isinstance(parameters, dict):
        raise ValueError("function.parameters must be a JSON Schema object")
    return function


def validate_tools(tools: Any) -> Optional[List[Dict[str, Any]]]:
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise ValueError("'tools' must be an array")
    for tool in tools:
        tool_function(tool)
    return tools


def tool_schema(tool: Dict[str, Any]):
    function = tool_function(tool)
    return {
        "type": "object",
        "properties": {
            "name": {"enum": [function["name"]]},
            "arguments": function.get("parameters") or {"type": "object"},
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }


def tool_call_schema(tools: List[Dict[str, Any]]):
    schemas = [tool_schema(tool) for tool in tools]
    return schemas[0] if len(schemas) == 1 else {"oneOf": schemas}


def resolve_tool_choice(
    tool_choice: Any, tools: Optional[List[Dict[str, Any]]]
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    if tool_choice is None:
        tool_choice = "auto"
    if tool_choice in ("auto", "none"):
        return tool_choice, None
    if not tools:
        raise ValueError("tool_choice requires a non-empty 'tools' array")
    if tool_choice == "required":
        return tool_choice, tools
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function":
            raise ValueError("tool_choice object must have type 'function'")
        function = tool_choice.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise ValueError("tool_choice.function.name must be a string")
        name = function["name"]
        selected = [tool for tool in tools if tool_function(tool)["name"] == name]
        if not selected:
            raise ValueError(f"tool_choice selected unknown function: {name!r}")
        return "function", selected
    raise ValueError(
        "tool_choice must be 'auto', 'none', 'required', or a function choice object"
    )


def parse_structured_tool_calls(text: str, allowed_tools: List[Dict[str, Any]]):
    names = {tool_function(tool)["name"] for tool in allowed_tools}
    try:
        parsed = json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"model did not return a valid JSON tool call: {e}") from e

    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("model tool-call JSON must be an object or non-empty list")

    tool_calls = []
    for item in parsed:
        if not isinstance(item, dict):
            raise ValueError("each tool-call JSON item must be an object")
        name = item.get("name")
        arguments = item.get("arguments", {})
        if not isinstance(name, str) or name not in names:
            raise ValueError(f"model selected unknown function: {name!r}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"function.arguments for {name!r} is not valid JSON: {e}"
                ) from e
        if not isinstance(arguments, dict):
            raise ValueError(f"function.arguments for {name!r} must be an object")
        tool_calls.append(
            {
                "id": f"call_{uuid.uuid4().hex}",
                "name": name,
                "arguments": arguments,
            }
        )
    return tool_calls


def forced_tool_instruction(tools: List[Dict[str, Any]]) -> str:
    tool_names = ", ".join(tool_function(tool)["name"] for tool in tools)
    return (
        "You must call a function. Respond only with a JSON object matching "
        '{"name": <function name>, "arguments": <JSON object>}. '
        f"Choose one of: {tool_names}."
    )
