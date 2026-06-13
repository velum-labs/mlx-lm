import json
import unittest

from mlx_lm.openai_compat import (
    ToolCallFormatter,
    forced_tool_instruction,
    parse_structured_tool_calls,
    resolve_tool_choice,
    tool_call_schema,
    validate_tools,
)


TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


class TestOpenAICompatTools(unittest.TestCase):
    def test_validate_tools(self):
        self.assertEqual(validate_tools([TOOL]), [TOOL])
        with self.assertRaises(ValueError):
            validate_tools({"type": "function"})
        with self.assertRaises(ValueError):
            validate_tools([{"type": "function", "function": {"parameters": {}}}])

    def test_resolve_tool_choice_modes(self):
        self.assertEqual(resolve_tool_choice(None, [TOOL]), ("auto", None))
        self.assertEqual(resolve_tool_choice("none", [TOOL]), ("none", None))
        self.assertEqual(resolve_tool_choice("required", [TOOL]), ("required", [TOOL]))
        self.assertEqual(
            resolve_tool_choice(
                {"type": "function", "function": {"name": "search"}},
                [TOOL],
            ),
            ("function", [TOOL]),
        )
        with self.assertRaises(ValueError):
            resolve_tool_choice("required", None)
        with self.assertRaises(ValueError):
            resolve_tool_choice(
                {"type": "function", "function": {"name": "missing"}},
                [TOOL],
            )

    def test_tool_call_schema_multiple_tools(self):
        other = {
            "type": "function",
            "function": {
                "name": "read_file",
                "parameters": {"type": "object"},
            },
        }
        schema = tool_call_schema([TOOL, other])

        self.assertIn("oneOf", schema)
        names = [
            item["properties"]["name"]["enum"][0]
            for item in schema["oneOf"]
        ]
        self.assertEqual(names, ["search", "read_file"])

    def test_parse_structured_tool_calls(self):
        parsed = parse_structured_tool_calls(
            '{"name":"search","arguments":{"query":"mlx"}}',
            [TOOL],
        )

        self.assertEqual(parsed[0]["name"], "search")
        self.assertEqual(parsed[0]["arguments"], {"query": "mlx"})
        self.assertTrue(parsed[0]["id"].startswith("call_"))

        with self.assertRaises(ValueError):
            parse_structured_tool_calls("{", [TOOL])
        with self.assertRaises(ValueError):
            parse_structured_tool_calls(
                '{"name":"missing","arguments":{}}',
                [TOOL],
            )

    def test_tool_call_formatter(self):
        formatter = ToolCallFormatter(
            lambda text, tools: {"name": "search", "arguments": json.loads(text)},
            [TOOL],
            streaming=True,
        )

        formatted = formatter(['{"query":"mlx"}'])[0]

        self.assertEqual(formatted["index"], 0)
        self.assertEqual(formatted["type"], "function")
        self.assertEqual(formatted["function"]["name"], "search")
        self.assertEqual(json.loads(formatted["function"]["arguments"]), {"query": "mlx"})

    def test_forced_tool_instruction_names_tools(self):
        self.assertIn("search", forced_tool_instruction([TOOL]))


if __name__ == "__main__":
    unittest.main()
