import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestNonMlxImportSafety(unittest.TestCase):
    def test_provider_and_metadata_helpers_import_without_mlx(self):
        script = r"""
import importlib.abc
import json
import sys


class BlockMlxImporter(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mlx" or fullname.startswith("mlx."):
            raise ModuleNotFoundError("blocked mlx import")
        return None


sys.meta_path.insert(0, BlockMlxImporter())

import mlx_lm.openai_compat as openai_compat
import mlx_lm.model_fusion_protocol as model_fusion_protocol
import mlx_lm.server_metadata as server_metadata
from mlx_lm.tool_parsers import json_tools


response = server_metadata.make_capabilities_response(
    model="mlx-community/Test-Model-4bit",
    base_url="http://127.0.0.1:8080",
    structured_output_available=False,
    embedding_model=None,
    max_output_tokens=64,
)
openai_compat.validate_model_endpoint_fixture(response["endpoint"])
parsed_tool_call = json_tools.parse_tool_call(
    '{"name": "search", "arguments": {"query": "mlx"}}',
    [
        {
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {"type": "object"},
            },
        }
    ],
)
print(
    json.dumps(
        {
            "schema": response["schema"],
            "tool_calls": response["capabilities"]["tool_calls"],
            "parsed_tool": parsed_tool_call["name"],
            "contract_source": (
                model_fusion_protocol.MODEL_FUSION_CONTRACT_SOURCE_OF_TRUTH
            ),
            "durable_records": model_fusion_protocol.MODEL_FUSION_DURABLE_RECORD_FORMAT,
            "http_api": model_fusion_protocol.MODEL_FUSION_HTTP_API_FORMAT,
            "openapi_status": model_fusion_protocol.MODEL_FUSION_OPENAPI_STATUS,
            "protobuf_status": model_fusion_protocol.MODEL_FUSION_PROTOBUF_BUF_STATUS,
            "protobuf_required": (
                model_fusion_protocol.MODEL_FUSION_PROTOBUF_BUF_REQUIRED_FOR_V1
            ),
            "schema_purpose": model_fusion_protocol.MODEL_FUSION_SCHEMA_BUNDLE_PURPOSE,
            "typescript_package": model_fusion_protocol.MODEL_FUSION_TYPESCRIPT_PACKAGE,
            "typescript_openapi_types": (
                model_fusion_protocol
                .MODEL_FUSION_TYPESCRIPT_OPENAPI_TYPES_GENERATOR
            ),
            "typescript_openapi_client": (
                model_fusion_protocol
                .MODEL_FUSION_TYPESCRIPT_OPENAPI_CLIENT_GENERATOR
            ),
            "typescript_json_schema_validator": (
                model_fusion_protocol.MODEL_FUSION_TYPESCRIPT_JSON_SCHEMA_VALIDATOR
            ),
            "python_import": model_fusion_protocol.MODEL_FUSION_PYTHON_IMPORT_NAME,
            "python_openapi": model_fusion_protocol.MODEL_FUSION_PYTHON_OPENAPI_GENERATOR,
            "python_json_schema_models": (
                model_fusion_protocol
                .MODEL_FUSION_PYTHON_JSON_SCHEMA_MODEL_GENERATOR
            ),
            "python_json_schema_validator": (
                model_fusion_protocol.MODEL_FUSION_PYTHON_JSON_SCHEMA_VALIDATOR
            ),
            "drift_strategy": (
                model_fusion_protocol.MODEL_FUSION_GENERATED_CODE_DRIFT_STRATEGY
            ),
        },
        sort_keys=True,
    )
)
"""
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )

        self.assertEqual(
            result.returncode,
            0,
            msg=f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )
        output = json.loads(result.stdout)
        self.assertEqual(output["schema"], "model_endpoint.v1")
        self.assertEqual(output["tool_calls"], "degraded")
        self.assertEqual(output["parsed_tool"], "search")
        self.assertEqual(output["contract_source"], "json_schema_openapi_3_1")
        self.assertEqual(output["durable_records"], "json_schema")
        self.assertEqual(output["http_api"], "openapi_3_1")
        self.assertEqual(output["openapi_status"], "v1_source_of_truth")
        self.assertEqual(output["protobuf_status"], "experimental_future")
        self.assertIs(output["protobuf_required"], False)
        self.assertEqual(output["schema_purpose"], "persisted_audit_benchmark_records")
        self.assertEqual(output["typescript_package"], "@velum/model-fusion-protocol")
        self.assertEqual(output["typescript_openapi_types"], "openapi-typescript")
        self.assertEqual(output["typescript_openapi_client"], "openapi-fetch")
        self.assertEqual(output["typescript_json_schema_validator"], "ajv")
        self.assertEqual(output["python_import"], "velum_model_fusion_protocol")
        self.assertEqual(output["python_openapi"], "openapi-python-client")
        self.assertEqual(output["python_json_schema_models"], "datamodel-code-generator")
        self.assertEqual(output["python_json_schema_validator"], "pydantic")
        self.assertEqual(output["drift_strategy"], "regenerate_and_fail_on_diff")


if __name__ == "__main__":
    unittest.main()
