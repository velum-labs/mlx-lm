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
            "typescript_package": model_fusion_protocol.MODEL_FUSION_TYPESCRIPT_PACKAGE,
            "python_import": model_fusion_protocol.MODEL_FUSION_PYTHON_IMPORT_NAME,
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
        self.assertEqual(output["typescript_package"], "@velum/model-fusion-protocol")
        self.assertEqual(output["python_import"], "velum_model_fusion_protocol")


if __name__ == "__main__":
    unittest.main()
