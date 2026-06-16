# Copyright © 2025 Apple Inc.

import json
from typing import Any, Iterator, Optional, Tuple

import re

# Matches <|"|>...<|"|> string literals (Gemma 4's string delimiter).
_GEMMA4_STR = r'<\|"\|>(?:(?!<\|"\|>)[\s\S])*?<\|"\|>'
_GEMMA4_STR_DELIMITER = '<|"|>'


def _find_tool_calls(text: str) -> Iterator[Tuple[str, str]]:
    pos = 0
    while True:
        call_start = text.find("call:", pos)
        if call_start < 0:
            return
        name_start = call_start + len("call:")
        name_end = name_start
        while name_end < len(text) and _is_name_char(text[name_end]):
            name_end += 1
        if name_end == name_start or name_end >= len(text) or text[name_end] != "{":
            pos = name_start
            continue
        args_start = name_end
        args_end = _find_balanced_args_end(text, args_start)
        if args_end is None:
            return
        yield text[name_start:name_end], text[args_start : args_end + 1]
        pos = args_end + 1


def _is_name_char(char: str) -> bool:
    return char.isalnum() or char in {"_", "-"}


def _find_balanced_args_end(text: str, start: int) -> Optional[int]:
    depth = 0
    pos = start
    while pos < len(text):
        if text.startswith(_GEMMA4_STR_DELIMITER, pos):
            string_end = text.find(
                _GEMMA4_STR_DELIMITER,
                pos + len(_GEMMA4_STR_DELIMITER),
            )
            if string_end < 0:
                return None
            pos = string_end + len(_GEMMA4_STR_DELIMITER)
            continue
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return pos
        pos += 1
    return None


def _gemma4_args_to_json(text: str) -> str:
    """Convert Gemma 4 tool call args to valid JSON.

    Gemma 4 uses unquoted keys and <|"|> as string delimiters
    instead of standard double quotes.
    """
    strings = []

    def _capture(m):
        strings.append(m.group(1))
        return f"\x00{len(strings) - 1}\x00"

    # Extract <|"|>-delimited strings and replace with placeholders
    text = re.sub(r'<\|"\|>(.*?)<\|"\|>', _capture, text, flags=re.DOTALL)
    # Quote bare keys
    text = re.sub(r"(?<=[{,])(\w+):", r'"\1":', text)
    # Restore captured strings as properly escaped JSON strings
    for i, s in enumerate(strings):
        text = text.replace(f"\x00{i}\x00", json.dumps(s))

    return text


def _parse_single(func_name: str, args_str: str) -> dict:
    """Parse a single call:name{args} match into a tool call dict."""
    json_str = _gemma4_args_to_json(args_str)
    arguments = json.loads(json_str)
    return dict(name=func_name, arguments=arguments)


def parse_tool_call(text: str, _: Optional[Any] = None):
    matches = list(_find_tool_calls(text))
    if not matches:
        raise ValueError("No function provided.")
    if len(matches) == 1:
        return _parse_single(*matches[0])
    return [_parse_single(*m) for m in matches]


tool_call_start = "<|tool_call>"
tool_call_end = "<tool_call|>"
