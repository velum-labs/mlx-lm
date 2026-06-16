import json
import unittest
from copy import deepcopy
from pathlib import Path

import mlx_lm.openai_compat as openai_compat

MODEL_FUSION_SCHEMA_BUNDLE_HASH = openai_compat.MODEL_FUSION_SCHEMA_BUNDLE_HASH
ToolCallFormatter = openai_compat.ToolCallFormatter
forced_tool_instruction = openai_compat.forced_tool_instruction
parse_structured_tool_calls = openai_compat.parse_structured_tool_calls
resolve_tool_choice = openai_compat.resolve_tool_choice
tool_call_schema = openai_compat.tool_call_schema
validate_model_call_record_fixture = openai_compat.validate_model_call_record_fixture
validate_model_endpoint_fixture = openai_compat.validate_model_endpoint_fixture
validate_model_fusion_contract_fixture = (
    openai_compat.validate_model_fusion_contract_fixture
)
validate_tools = openai_compat.validate_tools


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

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "model-fusion-contract"
CONTRACT_FIXTURES = (
    ("model_endpoint.v1", "minimal.json"),
    ("model_endpoint.v1", "realistic.json"),
    ("model-call-record.v1", "minimal.json"),
    ("model-call-record.v1", "realistic.json"),
)


def load_contract_fixture(schema_name, fixture_name):
    with (FIXTURE_ROOT / schema_name / fixture_name).open(encoding="utf-8") as f:
        return json.load(f)


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


class TestModelFusionContractFixtures(unittest.TestCase):
    def test_local_provider_contract_fixtures_validate(self):
        for schema_name, fixture_name in CONTRACT_FIXTURES:
            with self.subTest(schema=schema_name, fixture=fixture_name):
                fixture = load_contract_fixture(schema_name, fixture_name)

                self.assertIs(
                    validate_model_fusion_contract_fixture(fixture),
                    fixture,
                )
                self.assertEqual(
                    fixture["schema_bundle_hash"],
                    MODEL_FUSION_SCHEMA_BUNDLE_HASH,
                )

    def test_dispatcher_rejects_unknown_schema(self):
        fixture = load_contract_fixture("model_endpoint.v1", "minimal.json")
        fixture["schema"] = "fusion-run-request.v1"

        with self.assertRaisesRegex(ValueError, "Unsupported model-fusion fixture"):
            validate_model_fusion_contract_fixture(fixture)

    def test_endpoint_fixture_rejects_schema_drift(self):
        fixture = load_contract_fixture("model_endpoint.v1", "realistic.json")
        fixture["capabilities"] = deepcopy(fixture["capabilities"])
        fixture["capabilities"]["streaming"] = "experimental"

        with self.assertRaisesRegex(ValueError, "unsupported status"):
            validate_model_endpoint_fixture(fixture)

    def test_endpoint_fixture_rejects_extra_fields(self):
        fixture = load_contract_fixture("model_endpoint.v1", "minimal.json")
        fixture["raw_provider_config"] = {"secret": "not allowed"}

        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            validate_model_endpoint_fixture(fixture)

    def test_call_record_fixture_rejects_bad_message_shape(self):
        fixture = load_contract_fixture("model-call-record.v1", "minimal.json")
        fixture["messages"] = [
            {"role": "user", "content": "hello", "tool_call_id": "call_1"}
        ]

        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            validate_model_call_record_fixture(fixture)

    def test_call_record_fixture_rejects_bad_hashes(self):
        fixture = load_contract_fixture("model-call-record.v1", "minimal.json")
        fixture["request_hash"] = "0000"

        with self.assertRaisesRegex(ValueError, "request_hash must be sha256"):
            validate_model_call_record_fixture(fixture)


if __name__ == "__main__":
    unittest.main()
