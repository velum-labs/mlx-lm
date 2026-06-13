"""OpenAI compatibility helpers for the HTTP server."""

import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple


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
