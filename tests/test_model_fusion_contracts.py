import importlib.util
import json
import sys
import tempfile
import types
import unittest
from copy import deepcopy
from pathlib import Path

import mlx_lm.model_fusion_protocol as model_fusion_protocol
import mlx_lm.openai_compat as openai_compat

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "model-fusion-contract"
SERVER_METADATA_PATH = REPO_ROOT / "mlx_lm" / "server_metadata.py"

MODEL_FUSION_SCHEMA_BUNDLE_HASH = openai_compat.MODEL_FUSION_SCHEMA_BUNDLE_HASH
validate_model_call_record_fixture = openai_compat.validate_model_call_record_fixture
validate_model_endpoint_fixture = openai_compat.validate_model_endpoint_fixture
validate_model_fusion_contract_fixture = (
    openai_compat.validate_model_fusion_contract_fixture
)


class TestModelFusionContractFixtures(unittest.TestCase):
    def test_protocol_lock_pins_fusionkit_origin(self):
        lock = model_fusion_protocol.MODEL_FUSION_PROTOCOL_LOCK

        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_CANONICAL_SPEC,
            "velum-labs/openclaw-shared/spec/"
            "2026-06-16-model-fusion-protocol-packaging-spec.md",
        )
        self.assertEqual(lock["origin"]["repo"], "velum-labs/fusionkit")
        self.assertEqual(lock["origin"]["contract_path"], "spec/model-fusion-contract")
        self.assertEqual(
            lock["origin"]["openapi_path"],
            "spec/model-fusion-contract/openapi/model-fusion.v1.openapi.json",
        )
        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_CONTRACT_SOURCE_OF_TRUTH,
            "json_schema_openapi_3_1",
        )
        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_DURABLE_RECORD_FORMAT,
            "json_schema",
        )
        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_HTTP_API_FORMAT,
            "openapi_3_1",
        )
        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_OPENAPI_STATUS,
            "v1_source_of_truth",
        )
        self.assertEqual(model_fusion_protocol.MODEL_FUSION_OPENAPI_VERSION, "3.1")
        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_PROTOBUF_BUF_STATUS,
            "experimental_future",
        )
        self.assertIs(
            model_fusion_protocol.MODEL_FUSION_PROTOBUF_BUF_REQUIRED_FOR_V1,
            False,
        )
        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_SCHEMA_BUNDLE_PURPOSE,
            "persisted_audit_benchmark_records",
        )
        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_SCHEMA_BUNDLE_HASH,
            MODEL_FUSION_SCHEMA_BUNDLE_HASH,
        )
        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_TYPESCRIPT_PACKAGE,
            "@velum-labs/model-fusion-protocol",
        )
        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_TYPESCRIPT_OPENAPI_TYPES_GENERATOR,
            "openapi-typescript",
        )
        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_TYPESCRIPT_OPENAPI_CLIENT_GENERATOR,
            "openapi-fetch",
        )
        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_TYPESCRIPT_JSON_SCHEMA_VALIDATOR,
            "ajv",
        )
        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_PYTHON_IMPORT_NAME,
            "velum_model_fusion_protocol",
        )
        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_PYTHON_OPENAPI_GENERATOR,
            "openapi-python-client",
        )
        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_PYTHON_JSON_SCHEMA_MODEL_GENERATOR,
            "datamodel-code-generator",
        )
        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_PYTHON_JSON_SCHEMA_VALIDATOR,
            "pydantic",
        )
        self.assertEqual(
            model_fusion_protocol.MODEL_FUSION_GENERATED_CODE_DRIFT_STRATEGY,
            "regenerate_and_fail_on_diff",
        )
        self.assertIn(
            "MlxProviderService",
            model_fusion_protocol.MODEL_FUSION_SERVICE_BOUNDARIES,
        )

    def test_protocol_lock_matches_bundled_fixture_schemas(self):
        fixture_schemas = {
            path.name for path in FIXTURE_ROOT.iterdir() if path.is_dir()
        }

        self.assertEqual(
            fixture_schemas,
            set(model_fusion_protocol.MODEL_FUSION_PERSISTED_RECORDS),
        )
        for schema_name in fixture_schemas:
            for fixture_path in (FIXTURE_ROOT / schema_name).glob("*.json"):
                fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
                self.assertEqual(
                    fixture["schema_bundle_hash"],
                    model_fusion_protocol.MODEL_FUSION_SCHEMA_BUNDLE_HASH,
                )

    def test_local_provider_fixtures_validate(self):
        for schema_name, validator in (
            ("model_endpoint.v1", validate_model_endpoint_fixture),
            ("model-call-record.v1", validate_model_call_record_fixture),
        ):
            for fixture_name in ("minimal.json", "realistic.json"):
                with self.subTest(schema_name=schema_name, fixture_name=fixture_name):
                    fixture = _load_fixture(schema_name, fixture_name)
                    self.assertIs(validator(fixture), fixture)
                    self.assertIs(
                        validate_model_fusion_contract_fixture(fixture),
                        fixture,
                    )

    def test_model_call_record_jsonl_sample_validates(self):
        fixture_path = (
            FIXTURE_ROOT
            / "model-call-record.v1"
            / "benchmark-sample.jsonl"
        )
        lines = fixture_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        for line in lines:
            record = json.loads(line)
            validate_model_call_record_fixture(record)
            self.assertEqual(record["producer"], "mlx-lm-benchmark")
            self.assertNotIn("realistic", record["messages"][0]["content"].lower())

    def test_endpoint_fixture_rejects_contract_drift(self):
        fixture = _load_fixture("model_endpoint.v1", "minimal.json")

        for mutation in (
            lambda item: item.pop("endpoint_id"),
            lambda item: item.__setitem__("schema", "model-call-record.v1"),
            lambda item: item.__setitem__("schema_bundle_hash", "sha256:bad"),
            lambda item: item.__setitem__("tool_calls", "supported"),
        ):
            with self.subTest(mutation=mutation):
                changed = deepcopy(fixture)
                mutation(changed)
                with self.assertRaises(ValueError):
                    validate_model_endpoint_fixture(changed)

    def test_model_call_record_rejects_invalid_nested_shapes(self):
        fixture = _load_fixture("model-call-record.v1", "realistic.json")

        invalid_usage = deepcopy(fixture)
        invalid_usage["usage"]["total_tokens"] = -1
        with self.assertRaises(ValueError):
            validate_model_call_record_fixture(invalid_usage)

        invalid_message = deepcopy(fixture)
        invalid_message["messages"][0]["role"] = "developer"
        with self.assertRaises(ValueError):
            validate_model_call_record_fixture(invalid_message)

        extra_message_field = deepcopy(fixture)
        extra_message_field["messages"][0]["name"] = "system"
        with self.assertRaises(ValueError):
            validate_model_call_record_fixture(extra_message_field)

    def test_server_endpoint_fixture_helper_validates_output(self):
        server_metadata = self._load_server_metadata_module()
        record = server_metadata.make_model_endpoint_fixture(
            "mlx-local-test",
            "mlx-community/Test-Model-4bit",
            base_url="http://127.0.0.1:8080",
            api_compatibility="openai-chat-completions",
            capabilities={
                "chat_completions": "supported",
                "streaming": "supported",
            },
            created_at="2026-06-15T22:00:00Z",
            producer_git_sha="b" * 40,
            tags=["local", "test"],
        )

        self.assertEqual(record["schema"], "model_endpoint.v1")
        self.assertEqual(record["schema_bundle_hash"], MODEL_FUSION_SCHEMA_BUNDLE_HASH)
        self.assertEqual(record["base_url"], "http://127.0.0.1:8080")
        validate_model_endpoint_fixture(record)

    def test_server_capabilities_response_helper_validates_endpoint(self):
        server_metadata = self._load_server_metadata_module()
        response = server_metadata.make_capabilities_response(
            model="mlx-community/Test-Model-4bit",
            base_url="http://127.0.0.1:8080",
            structured_output_available=True,
            embedding_model=None,
            max_output_tokens=128,
        )

        self.assertEqual(response["object"], "capabilities")
        self.assertEqual(response["schema"], "model_endpoint.v1")
        self.assertEqual(response["capabilities"]["tool_calls"], "supported")
        self.assertEqual(response["capabilities"]["embeddings"], "unsupported")
        self.assertEqual(response["endpoints"]["/v1/embeddings"], "unsupported")
        self.assertEqual(response["limits"]["max_output_tokens"], 128)
        validate_model_endpoint_fixture(response["endpoint"])

    def test_benchmark_model_call_fixture_helper_validates_output(self):
        server_benchmark = self._load_server_benchmark_module()
        messages = [{"role": "user", "content": "Say hello."}]
        request_payload = {
            "model": "default_model",
            "messages": messages,
            "max_tokens": 4,
        }
        response_payload = {"choices": [{"message": {"content": "Hello."}}]}

        record = server_benchmark.make_model_call_record_fixture(
            call_id="call_bench_001",
            endpoint_id="mlx-local-test",
            model="default_model",
            messages=messages,
            status="succeeded",
            started_at="2026-06-15T22:01:00Z",
            finished_at="2026-06-15T22:01:01Z",
            request_payload=request_payload,
            response_payload=response_payload,
            latency_ms=1000.0,
            usage={"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            output_text="Hello.",
            created_at="2026-06-15T22:01:00Z",
            producer_git_sha="c" * 40,
        )

        self.assertEqual(record["schema"], "model-call-record.v1")
        self.assertEqual(
            record["request_hash"],
            server_benchmark.contract_sha256(request_payload),
        )
        self.assertEqual(
            record["response_hash"],
            server_benchmark.contract_sha256(response_payload),
        )
        validate_model_call_record_fixture(record)

    def test_benchmark_model_call_jsonl_writer_validates_records(self):
        server_benchmark = self._load_server_benchmark_module()
        messages = [{"role": "user", "content": "Say hello."}]
        request_payload = {
            "model": "default_model",
            "messages": messages,
            "max_tokens": 4,
            "stream": True,
        }
        response_payload = {
            "http_status": 200,
            "provider_request_id": "chatcmpl-bench-001",
            "output_text": "Hello.",
            "finish_reasons": ["stop"],
            "usage": {"completion_tokens": 1},
        }
        result = {
            "request_index": 1,
            "call_id": "call_bench_test_001",
            "messages": messages,
            "request_payload": request_payload,
            "success": True,
            "tokens": [1.0],
            "started_at": "2026-06-15T22:01:00Z",
            "finished_at": "2026-06-15T22:01:01Z",
            "latency_ms": 1000.0,
            "provider_request_id": "chatcmpl-bench-001",
            "response_payload": response_payload,
            "usage": {"completion_tokens": 1},
            "output_text": "Hello.",
            "metadata": {
                "http_status": 200,
                "model_call_id_echo": "call_bench_test_001",
                "memory_peak_gb": 1.25,
            },
        }

        record = server_benchmark.make_model_call_record_from_benchmark_result(
            result=result,
            endpoint_id="mlx-local-test",
            model="default_model",
            url="http://localhost:8080/v1/chat/completions",
            max_tokens=4,
            concurrency=1,
            total_requests=1,
            platform_info={"system": "Darwin", "machine": "arm64"},
            benchmark_task_id="bench-test-001",
            schema_valid_rate=1.0,
            tool_call_valid_rate=None,
            producer_git_sha="d" * 40,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model-call-records.jsonl"
            server_benchmark.write_model_call_records_jsonl(
                str(output_path), [record]
            )
            lines = output_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        written = json.loads(lines[0])
        validate_model_call_record_fixture(written)
        self.assertEqual(written["status"], "succeeded")
        self.assertEqual(written["provider_request_id"], "chatcmpl-bench-001")
        self.assertEqual(
            written["request_hash"],
            server_benchmark.contract_sha256(request_payload),
        )
        self.assertEqual(
            written["response_hash"],
            server_benchmark.contract_sha256(response_payload),
        )
        self.assertEqual(written["metadata"]["memory_peak_gb"], 1.25)
        self.assertEqual(written["metadata"]["platform"]["system"], "Darwin")
        self.assertEqual(written["metadata"]["benchmark"]["task_id"], "bench-test-001")
        self.assertEqual(written["metadata"]["benchmark"]["schema_valid_rate"], 1.0)
        self.assertIsNone(written["metadata"]["benchmark"]["tool_call_valid_rate"])

    def test_benchmark_model_call_record_captures_failed_response(self):
        server_benchmark = self._load_server_benchmark_module()
        messages = [{"role": "user", "content": "Say hello."}]
        response_payload = {
            "http_status": 500,
            "body": '{"error":{"message":"server unavailable"}}',
        }
        result = {
            "request_index": 2,
            "call_id": "call_bench_test_002",
            "messages": messages,
            "request_payload": {
                "model": "default_model",
                "messages": messages,
                "max_tokens": 4,
                "stream": True,
            },
            "success": False,
            "tokens": [],
            "started_at": "2026-06-15T22:01:00Z",
            "finished_at": "2026-06-15T22:01:02Z",
            "latency_ms": 2000.0,
            "response_payload": response_payload,
            "error": {
                "kind": "provider_error",
                "message": "server unavailable",
                "retryable": True,
            },
            "metadata": {"http_status": 500},
        }

        record = server_benchmark.make_model_call_record_from_benchmark_result(
            result=result,
            endpoint_id="mlx-local-test",
            model="default_model",
            url="http://localhost:8080/v1/chat/completions",
            max_tokens=4,
            concurrency=1,
            total_requests=2,
            platform_info={"system": "Darwin"},
            benchmark_task_id="bench-test-002",
            schema_valid_rate=1.0,
            tool_call_valid_rate=None,
            producer_git_sha="e" * 40,
        )

        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error"]["kind"], "provider_error")
        self.assertEqual(
            record["response_hash"],
            server_benchmark.contract_sha256(response_payload),
        )
        validate_model_call_record_fixture(record)

    def _load_server_metadata_module(self):
        spec = importlib.util.spec_from_file_location(
            "mlx_lm.server_metadata",
            SERVER_METADATA_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        previous_package = sys.modules.get("mlx_lm")
        previous_openai_compat = sys.modules.get("mlx_lm.openai_compat")
        mlx_lm_package = types.ModuleType("mlx_lm")
        mlx_lm_package.__path__ = [str(REPO_ROOT / "mlx_lm")]
        sys.modules["mlx_lm"] = mlx_lm_package
        sys.modules["mlx_lm.openai_compat"] = openai_compat
        try:
            spec.loader.exec_module(module)
        finally:
            if previous_package is None:
                sys.modules.pop("mlx_lm", None)
            else:
                sys.modules["mlx_lm"] = previous_package
            if previous_openai_compat is None:
                sys.modules.pop("mlx_lm.openai_compat", None)
            else:
                sys.modules["mlx_lm.openai_compat"] = previous_openai_compat
        return module

    def _load_server_benchmark_module(self):
        spec = importlib.util.spec_from_file_location(
            "mlx_lm_server_benchmark",
            REPO_ROOT / "benchmarks" / "server_benchmark.py",
        )
        module = importlib.util.module_from_spec(spec)
        previous_package = sys.modules.get("mlx_lm")
        previous_openai_compat = sys.modules.get("mlx_lm.openai_compat")
        mlx_lm_package = types.ModuleType("mlx_lm")
        mlx_lm_package.__path__ = [str(REPO_ROOT / "mlx_lm")]
        sys.modules["mlx_lm"] = mlx_lm_package
        sys.modules["mlx_lm.openai_compat"] = openai_compat
        try:
            spec.loader.exec_module(module)
        except ModuleNotFoundError as e:
            if e.name in {"aiohttp", "tqdm"}:
                self.skipTest(f"benchmark dependency is not installed: {e.name}")
            raise
        finally:
            if previous_package is None:
                sys.modules.pop("mlx_lm", None)
            else:
                sys.modules["mlx_lm"] = previous_package
            if previous_openai_compat is None:
                sys.modules.pop("mlx_lm.openai_compat", None)
            else:
                sys.modules["mlx_lm.openai_compat"] = previous_openai_compat
        return module


def _load_fixture(schema_name, fixture_name):
    with (FIXTURE_ROOT / schema_name / fixture_name).open(encoding="utf-8") as handle:
        fixture = json.load(handle)
    if not isinstance(fixture, dict):
        raise TypeError(f"{schema_name}/{fixture_name} must be a JSON object")
    return fixture


if __name__ == "__main__":
    unittest.main()
