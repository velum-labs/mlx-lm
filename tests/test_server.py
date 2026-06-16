# Copyright © 2024 Apple Inc.

import http
import io
import json
import threading
import types
import unittest

import mlx.core as mx
import requests

import mlx_lm.server as server_module
from mlx_lm.embeddings import EmbeddingNotConfiguredError
from mlx_lm.models.cache import KVCache
from mlx_lm.openai_compat import tool_call_schema, validate_model_endpoint_fixture
from mlx_lm.server import (
    APIHandler,
    LRUPromptCache,
    Response,
    ResponseGenerator,
    _process_control_tokens,
)
from mlx_lm.utils import load


class DummyModelProvider:
    def __init__(self, with_draft=False):
        HF_MODEL_PATH = "mlx-community/Qwen1.5-0.5B-Chat-4bit"
        self.model, self.tokenizer = load(HF_MODEL_PATH)
        self.model_key = (HF_MODEL_PATH, None)
        self.is_batchable = True

        # Add draft model support
        self.draft_model = None
        self.draft_model_key = None
        self.cli_args = type(
            "obj",
            (object,),
            {
                "adapter_path": None,
                "chat_template": None,
                "use_default_chat_template": False,
                "trust_remote_code": False,
                "draft_model": None,
                "num_draft_tokens": 3,
                "temp": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "max_tokens": 512,
                "chat_template_args": {},
                "model": None,
                "decode_concurrency": 32,
                "prompt_concurrency": 8,
                "prefill_step_size": 2048,
                "prompt_cache_size": 10,
                "prompt_cache_bytes": 1 << 63,
                "prompt_cache_total_bytes": None,
                "allowed_origins": ["*"],
            },
        )

        if with_draft:
            # Use the same model as the draft model for testing
            self.draft_model, _ = load(HF_MODEL_PATH)
            self.draft_model_key = HF_MODEL_PATH
            self.cli_args.draft_model = HF_MODEL_PATH

    def load(self, model, adapter=None, draft_model=None):
        assert model in ["default_model", "chat_model"]
        return self.model, self.tokenizer

    def load_default(self):
        return self.load("default_model", None, "default_model")


class FakeContext:
    def __init__(self):
        self.tool_parser = None
        self.prompt = [1, 2, 3]
        self.prompt_cache_count = -1
        self.stopped = False

    def stop(self):
        self.stopped = True


class FakeChatResponseGenerator:
    def __init__(self, responses):
        self.responses = responses
        self.last_request = None
        self.last_args = None
        self.context = FakeContext()
        self.cli_args = types.SimpleNamespace(
            allowed_origins=["*"],
            num_draft_tokens=3,
            temp=0.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            max_tokens=512,
            model=None,
            embedding_model=None,
        )

    def generate(self, request, generation_args, progress_callback=None):
        self.last_request = request
        self.last_args = generation_args
        return self.context, iter(self.responses)


class FakeEmbeddingResponseGenerator:
    def __init__(self, embedding_model="embed-model", fail=False):
        self.inputs = None
        self.fail = fail
        self.cli_args = types.SimpleNamespace(
            allowed_origins=["*"],
            model=None,
            embedding_model=embedding_model,
            max_tokens=512,
        )

    def embed(self, inputs):
        self.inputs = inputs
        if self.fail:
            raise EmbeddingNotConfiguredError(
                "No embedding model configured; start the server with --embedding-model"
            )
        embeddings = [
            [float(index), float(index + 1), float(len(text))]
            for index, text in enumerate(inputs)
        ]
        return embeddings, sum(max(1, len(text.split())) for text in inputs)


class TestModelFusionMetadataEndpoints(unittest.TestCase):
    def _serve(self, response_generator):
        httpd = http.server.HTTPServer(
            ("localhost", 0),
            lambda *args, **kwargs: APIHandler(response_generator, *args, **kwargs),
        )
        thread = threading.Thread(target=httpd.serve_forever)
        thread.daemon = True
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return httpd.server_port

    def test_v1_health_shape_and_legacy_health_unchanged(self):
        port = self._serve(FakeEmbeddingResponseGenerator())

        response = requests.get(f"http://localhost:{port}/v1/health")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["provider"], "mlx-lm")
        self.assertIsInstance(body["version"], str)
        self.assertEqual(body["model"], "default_model")
        self.assertIsInstance(body["mlx_version"], str)
        self.assertIsInstance(body["platform"], str)
        self.assertEqual(body["commit"], "unknown")
        self.assertEqual(body["loaded_model_status"], "unknown")

        legacy_response = requests.get(f"http://localhost:{port}/health")
        self.assertEqual(legacy_response.status_code, 200)
        self.assertEqual(legacy_response.json(), {"status": "ok"})

    def test_v1_capabilities_returns_model_endpoint_contract(self):
        response_generator = FakeEmbeddingResponseGenerator()
        response_generator.cli_args.model = "mlx-community/Test-Chat-4bit"
        port = self._serve(response_generator)

        response = requests.get(f"http://localhost:{port}/v1/capabilities")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        endpoint = body["endpoint"]
        validate_model_endpoint_fixture(endpoint)
        self.assertEqual(body["object"], "capabilities")
        self.assertEqual(body["schema"], "model_endpoint.v1")
        self.assertEqual(body["provider"], "mlx-lm")
        self.assertEqual(body["model"], "mlx-community/Test-Chat-4bit")
        self.assertEqual(body["status"], "succeeded")
        self.assertEqual(body["limits"]["max_context_tokens"], None)
        self.assertEqual(body["limits"]["max_output_tokens"], 512)
        self.assertEqual(body["model_info"]["estimated_memory_gb"], None)
        self.assertEqual(body["model_info"]["quantization"], "unknown")
        self.assertEqual(body["endpoints"]["/v1/chat/completions"], "supported")
        self.assertEqual(body["endpoints"]["/v1/completions"], "supported")
        self.assertEqual(body["endpoints"]["/v1/embeddings"], "supported")
        self.assertEqual(endpoint["schema"], "model_endpoint.v1")
        self.assertEqual(endpoint["provider"], "mlx-lm")
        self.assertEqual(endpoint["owner"], "mlx-lm")
        self.assertEqual(endpoint["model"], "mlx-community/Test-Chat-4bit")
        self.assertEqual(endpoint["base_url"], f"http://localhost:{port}")
        self.assertEqual(endpoint["api_compatibility"], "mlx-lm-server")
        self.assertEqual(endpoint["status"], "succeeded")
        self.assertEqual(body["capabilities"]["chat_completions"], "supported")
        self.assertEqual(body["capabilities"]["text_completions"], "supported")
        self.assertEqual(body["capabilities"]["streaming"], "supported")
        self.assertEqual(body["capabilities"]["embeddings"], "supported")

        expected_structured = (
            "supported"
            if server_module.parse_request_constraint is not None
            else "unsupported"
        )
        expected_tool_calls = (
            "supported"
            if server_module.parse_request_constraint is not None
            else "degraded"
        )
        self.assertEqual(body["capabilities"]["structured_output"], expected_structured)
        self.assertEqual(body["capabilities"]["tool_calls"], expected_tool_calls)

    def test_v1_capabilities_reports_unconfigured_embeddings(self):
        response_generator = FakeEmbeddingResponseGenerator(embedding_model=None)
        port = self._serve(response_generator)

        response = requests.get(f"http://localhost:{port}/v1/capabilities")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["capabilities"]["embeddings"], "unsupported")


class MockCache:
    def __init__(self, value, is_trimmable: bool = True):
        self.value = value
        self._is_trimmable = is_trimmable

    @property
    def nbytes(self):
        return len(self.value)

    def __eq__(self, other):
        return other.value == self.value

    def is_trimmable(self):
        return self._is_trimmable

    def trim(self, n):
        assert self._is_trimmable
        return n


class TestProcessControlTokens(unittest.TestCase):
    @staticmethod
    def _r(text, state, match=None):
        return Response(text, 0, state, match, 0.0, None, ())

    def test_single_tool_call_passes_body_with_open_and_close_crossings(self):
        r = self._r
        stream = [
            r("hi ", "normal"),
            r("<tool_call>", "tool", match=(0,)),
            r("body", "tool"),
            r("</tool_call>", "normal", match=(1,)),
            r(" bye", "normal"),
        ]
        ctx = types.SimpleNamespace(
            sequences={(0,): "<tool_call>", (1,): "</tool_call>"}
        )
        out = list(_process_control_tokens(ctx, iter(stream)))

        self.assertEqual("".join(t.text for t in out), "hi body bye")
        states = [t.state for t in out]
        self.assertEqual(sum(1 for a, b in zip(states, states[1:]) if a != b), 2)

    def test_back_to_back_tool_calls_emit_state_crossings(self):
        r = self._r
        stream = [
            r("<tool_call>", "tool", match=(0,)),
            r("call1_body", "tool"),
            r("</tool_call>", "normal", match=(1,)),
            r("<tool_call>", "tool", match=(0,)),
            r("call2_body", "tool"),
            r("</tool_call>", "normal", match=(1,)),
        ]
        ctx = types.SimpleNamespace(
            sequences={(0,): "<tool_call>", (1,): "</tool_call>"}
        )
        out = list(_process_control_tokens(ctx, iter(stream)))

        self.assertEqual("".join(t.text for t in out), "call1_bodycall2_body")
        states = [t.state for t in out]
        crossings = sum(
            1 for a, b in zip(states, states[1:]) if a == "tool" and b == "normal"
        )
        self.assertEqual(crossings, 2)

    def test_multi_token_match_preserves_order(self):
        r = self._r
        match = (10, 11, 12)
        stream = [
            r("body", "tool"),
            r("</", "tool"),
            r("tool", "tool"),
            r("_call>", "normal", match=match),
            r(" ok", "normal"),
        ]
        ctx = types.SimpleNamespace(sequences={match: "</tool_call>"})
        out = list(_process_control_tokens(ctx, iter(stream)))

        self.assertEqual([t.text for t in out], ["body", "", "", "", " ok"])
        self.assertEqual(
            [t.state for t in out],
            ["tool", "tool", "tool", "normal", "normal"],
        )


class TestOpenAIEmbeddings(unittest.TestCase):
    def _serve(self, response_generator):
        httpd = http.server.HTTPServer(
            ("localhost", 0),
            lambda *args, **kwargs: APIHandler(response_generator, *args, **kwargs),
        )
        thread = threading.Thread(target=httpd.serve_forever)
        thread.daemon = True
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return httpd.server_port

    def test_embeddings_shape_and_usage(self):
        response_generator = FakeEmbeddingResponseGenerator()
        port = self._serve(response_generator)

        response = requests.post(
            f"http://localhost:{port}/v1/embeddings",
            json={"model": "embed-model", "input": ["hello world", "x"]},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["object"], "list")
        self.assertEqual(body["model"], "embed-model")
        self.assertEqual(body["usage"], {"prompt_tokens": 3, "total_tokens": 3})
        self.assertEqual([item["index"] for item in body["data"]], [0, 1])
        self.assertEqual(body["data"][0]["object"], "embedding")
        self.assertTrue(
            all(isinstance(v, float) for v in body["data"][0]["embedding"])
        )
        self.assertEqual(response_generator.inputs, ["hello world", "x"])

    def test_embeddings_default_model_alias_reports_configured_model(self):
        response_generator = FakeEmbeddingResponseGenerator()
        port = self._serve(response_generator)

        response = requests.post(
            f"http://localhost:{port}/v1/embeddings",
            json={"model": "default_model", "input": "hello"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "embed-model")

    def test_embeddings_rejects_unconfigured_model_id(self):
        port = self._serve(FakeEmbeddingResponseGenerator())

        response = requests.post(
            f"http://localhost:{port}/v1/embeddings",
            json={"model": "other-model", "input": "hello"},
        )

        self.assertEqual(response.status_code, 404)
        self.assertIn("not available", response.json()["error"]["message"])

    def test_embeddings_without_configured_model_is_400(self):
        response_generator = FakeEmbeddingResponseGenerator(
            embedding_model=None, fail=True
        )
        port = self._serve(response_generator)

        response = requests.post(
            f"http://localhost:{port}/v1/embeddings",
            json={"model": "embed-model", "input": "hello"},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "No embedding model configured", response.json()["error"]["message"]
        )

    def test_embeddings_reject_unsupported_fields(self):
        port = self._serve(FakeEmbeddingResponseGenerator())

        for extra in (
            {"dimensions": 128},
            {"encoding_format": "base64"},
            {"input": [1, 2, 3]},
        ):
            with self.subTest(extra=extra):
                response = requests.post(
                    f"http://localhost:{port}/v1/embeddings",
                    json={"model": "embed-model", "input": "hello", **extra},
                )

                self.assertEqual(response.status_code, 400)

    def test_models_lists_configured_embedding_model(self):
        port = self._serve(FakeEmbeddingResponseGenerator())

        response = requests.get(f"http://localhost:{port}/v1/models")

        self.assertEqual(response.status_code, 200)
        model_ids = {model["id"] for model in response.json()["data"]}
        self.assertIn("embed-model", model_ids)


class TestOpenAIToolCalling(unittest.TestCase):
    TOOL = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city", "unit"],
                "additionalProperties": False,
            },
        },
    }

    def _serve(self, responses):
        response_generator = FakeChatResponseGenerator(responses)
        httpd = http.server.HTTPServer(
            ("localhost", 0),
            lambda *args, **kwargs: APIHandler(response_generator, *args, **kwargs),
        )
        thread = threading.Thread(target=httpd.serve_forever)
        thread.daemon = True
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return httpd.server_port, response_generator

    def _tool_call_responses(self):
        return [
            Response(
                '{"name":"get_weather","arguments":',
                10,
                "normal",
                None,
                0.0,
                None,
                (),
            ),
            Response(
                '{"city":"Paris","unit":"celsius"}}',
                11,
                "normal",
                None,
                0.0,
                "stop",
                (),
            ),
        ]

    def test_tool_call_schema_constrains_selected_arguments(self):
        schema = tool_call_schema([self.TOOL])

        self.assertEqual(schema["properties"]["name"], {"enum": ["get_weather"]})
        self.assertEqual(
            schema["properties"]["arguments"],
            self.TOOL["function"]["parameters"],
        )
        self.assertFalse(schema["additionalProperties"])

    def test_forced_tool_call_non_stream(self):
        port, response_generator = self._serve(self._tool_call_responses())

        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": "chat_model",
                "messages": [{"role": "user", "content": "weather in paris?"}],
                "tools": [self.TOOL],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
                "max_tokens": 64,
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        choice = body["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        tool_call = choice["message"]["tool_calls"][0]
        self.assertEqual(tool_call["type"], "function")
        self.assertTrue(tool_call["id"].startswith("call_"))
        self.assertEqual(tool_call["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(tool_call["function"]["arguments"]),
            {"city": "Paris", "unit": "celsius"},
        )
        self.assertEqual(body["usage"]["completion_tokens"], 2)
        self.assertIn(
            "You must call a function",
            response_generator.last_request.messages[0]["content"],
        )

    def test_forced_tool_call_stream(self):
        port, _ = self._serve(self._tool_call_responses())

        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            stream=True,
            json={
                "model": "chat_model",
                "messages": [{"role": "user", "content": "weather in paris?"}],
                "tools": [self.TOOL],
                "tool_choice": "required",
                "stream": True,
                "stream_options": {"include_usage": True},
                "max_tokens": 64,
            },
        )

        self.assertEqual(response.status_code, 200)
        chunks = []
        for line in response.iter_lines():
            if not line:
                continue
            data = line.decode("utf-8")
            if data == "data: [DONE]" or not data.startswith("data: "):
                continue
            chunks.append(json.loads(data[6:]))

        tool_chunks = [
            chunk
            for chunk in chunks
            if chunk["choices"]
            and chunk["choices"][0]["delta"].get("tool_calls")
        ]
        self.assertEqual(len(tool_chunks), 1)
        choice = tool_chunks[0]["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        tool_call = choice["delta"]["tool_calls"][0]
        self.assertEqual(tool_call["index"], 0)
        self.assertTrue(tool_call["id"].startswith("call_"))
        self.assertEqual(tool_call["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(tool_call["function"]["arguments"]),
            {"city": "Paris", "unit": "celsius"},
        )

        usage_chunks = [chunk for chunk in chunks if chunk.get("usage")]
        self.assertEqual(len(usage_chunks), 1)
        self.assertEqual(usage_chunks[0]["usage"]["completion_tokens"], 2)

    def test_model_call_id_header_echoes_on_chat_response(self):
        port, _ = self._serve(
            [Response("hello", 10, "normal", None, 0.0, "stop", ())]
        )

        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            headers={"x-velum-model-call-id": "velum-call-123"},
            json={
                "model": "chat_model",
                "messages": [{"role": "user", "content": "say hello"}],
                "max_tokens": 64,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-velum-model-call-id"], "velum-call-123")
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "hello")

    def test_missing_model_call_id_does_not_add_echo_header(self):
        port, _ = self._serve(
            [Response("hello", 10, "normal", None, 0.0, "stop", ())]
        )

        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": "chat_model",
                "messages": [{"role": "user", "content": "say hello"}],
                "max_tokens": 64,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("x-velum-model-call-id", response.headers)

    def test_tool_choice_requires_tools(self):
        port, _ = self._serve([])

        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            headers={"x-velum-model-call-id": "velum-error-456"},
            json={
                "model": "chat_model",
                "messages": [{"role": "user", "content": "weather in paris?"}],
                "tool_choice": "required",
                "max_tokens": 64,
            },
        )

        self.assertEqual(response.status_code, 400)
        error = response.json()["error"]
        self.assertIn("non-empty 'tools' array", error["message"])
        self.assertEqual(error["type"], "invalid_request_error")
        self.assertEqual(error["code"], "invalid_tool_choice")
        self.assertEqual(response.headers["x-velum-model-call-id"], "velum-error-456")

    def test_invalid_request_body_error_does_not_log_prompt_content(self):
        port, _ = self._serve([])
        secret = "secret prompt content"

        with self.assertLogs(level="ERROR") as logs:
            response = requests.post(
                f"http://localhost:{port}/v1/chat/completions",
                data=json.dumps([{"role": "user", "content": secret}]),
                headers={"Content-Type": "application/json"},
            )

        self.assertEqual(response.status_code, 400)
        error = response.json()["error"]
        self.assertEqual(error["message"], "Request should be a JSON dictionary")
        self.assertNotIn(secret, "\n".join(logs.output))
        self.assertNotIn(secret, response.text)

    def test_forced_tool_call_requires_structured_extra(self):
        original = server_module.parse_request_constraint
        server_module.parse_request_constraint = None
        self.addCleanup(setattr, server_module, "parse_request_constraint", original)
        port, _ = self._serve(self._tool_call_responses())

        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": "chat_model",
                "messages": [{"role": "user", "content": "weather in paris?"}],
                "tools": [self.TOOL],
                "tool_choice": "required",
                "max_tokens": 64,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("mlx-lm[structured]", response.json()["error"]["message"])

    def test_invalid_forced_tool_json_returns_400(self):
        port, _ = self._serve(
            [Response("{", 10, "normal", None, 0.0, "stop", ())]
        )

        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": "chat_model",
                "messages": [{"role": "user", "content": "weather in paris?"}],
                "tools": [self.TOOL],
                "tool_choice": "required",
                "max_tokens": 64,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("valid JSON tool call", response.json()["error"]["message"])

    def test_invalid_forced_tool_json_stream_returns_400_before_sse(self):
        port, _ = self._serve(
            [Response("{", 10, "normal", None, 0.0, "stop", ())]
        )

        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            stream=True,
            json={
                "model": "chat_model",
                "messages": [{"role": "user", "content": "weather in paris?"}],
                "tools": [self.TOOL],
                "tool_choice": "required",
                "stream": True,
                "max_tokens": 64,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.headers["Content-Type"], "application/json")
        self.assertIn("valid JSON tool call", response.json()["error"]["message"])


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_handle_completions(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.5,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "repetition_context_size": 20,
            "seed": 999,
            "stop": "stop sequence",
        }

        response = requests.post(url, json=post_data)

        response_body = json.loads(response.text)

        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        first_text = response_body["choices"][0]["text"]
        self.assertEqual(
            first_text,
            json.loads(requests.post(url, json=post_data).text)["choices"][0]["text"],
        )

    def test_handle_chat_completions(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_content_fragments(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."}
                    ],
                },
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_handle_chat_completions_with_null_tool_content(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.7,
            "top_p": 0.85,
            "repetition_penalty": 1.2,
            "messages": [
                {"role": "user", "content": "what is 2+3?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "id": "123",
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 2, "b": 3}',
                            },
                        }
                    ],
                },
                {"role": "tool", "content": "5", "tool_call_id": "123"},
            ],
        }
        response = requests.post(url, json=chat_post_data)
        response_body = response.text
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)

    def test_make_state_machine_empty_tool_call_end(self):
        class FakeTokenizer:
            has_thinking = False
            has_tool_calling = True
            tool_call_start = "[TOOL_CALLS]"
            tool_call_end = ""
            tool_call_start_tokens = (100,)
            tool_call_end_tokens = ()
            eos_token_ids = [2]

            def convert_ids_to_tokens(self, t):
                return f"<eos{t}>"

        sm, _ = self.response_generator._make_state_machine(
            ("fake-empty-end", None, None),
            FakeTokenizer(),
            stop_words=[],
        )
        state = sm.make_state()
        state, _, s = sm.match(state, 100)
        self.assertEqual(s, "tool")
        for tok in [42, 43, 44]:
            state, _, s = sm.match(state, tok)
            self.assertEqual(s, "tool")
        state, _, s = sm.match(state, 2)
        self.assertIsNone(s)

    def test_handle_models(self):
        url = f"http://localhost:{self.port}/v1/models"
        response = requests.get(url)
        self.assertEqual(response.status_code, 200)
        response_body = json.loads(response.text)
        self.assertEqual(response_body["object"], "list")
        self.assertIsInstance(response_body["data"], list)
        self.assertGreater(len(response_body["data"]), 0)
        model = response_body["data"][0]
        self.assertIn("id", model)
        self.assertEqual(model["object"], "model")
        self.assertIn("created", model)


class TestServerWithDraftModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.response_generator = ResponseGenerator(
            DummyModelProvider(with_draft=True), LRUPromptCache()
        )
        cls.server_address = ("localhost", 0)
        cls.httpd = http.server.HTTPServer(
            cls.server_address,
            lambda *args, **kwargs: APIHandler(cls.response_generator, *args, **kwargs),
        )
        cls.port = cls.httpd.server_port
        cls.server_thread = threading.Thread(target=cls.httpd.serve_forever)
        cls.server_thread.daemon = True
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.server_thread.join()
        cls.response_generator.stop_and_join()

    def test_handle_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/completions"

        post_data = {
            "model": "default_model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 0.0,
            "top_p": 1.0,
        }

        response = requests.post(url, json=post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_handle_chat_completions_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data)
        self.assertEqual(response.status_code, 200)

        response_body = json.loads(response.text)
        self.assertIn("id", response_body)
        self.assertIn("choices", response_body)
        self.assertIn("usage", response_body)

        # Check that tokens were generated
        self.assertTrue(response_body["usage"]["completion_tokens"] > 0)

    def test_streaming_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 10,
            "temperature": 0.0,
            "stream": True,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
        }

        response = requests.post(url, json=chat_post_data, stream=True)
        self.assertEqual(response.status_code, 200)

        chunk_count = 0
        for chunk in response.iter_lines():
            if chunk:
                data = chunk.decode("utf-8")
                if data.startswith("data: ") and data != "data: [DONE]":
                    chunk_data = json.loads(data[6:])  # Skip the "data: " prefix
                    self.assertIn("choices", chunk_data)
                    self.assertEqual(len(chunk_data["choices"]), 1)
                    self.assertIn("delta", chunk_data["choices"][0])
                    chunk_count += 1

        # Make sure we got some streaming chunks
        self.assertGreater(chunk_count, 0)

    def test_prompt_cache_with_draft_model(self):
        url = f"http://localhost:{self.port}/v1/chat/completions"

        # First request to initialize cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about"},
            ],
        }

        first_response = requests.post(url, json=chat_post_data)
        self.assertEqual(first_response.status_code, 200)

        # Second request with same prefix should use cache
        chat_post_data = {
            "model": "chat_model",
            "max_tokens": 5,
            "temperature": 0.0,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a story about dragons."},
            ],
        }

        second_response = requests.post(url, json=chat_post_data)
        self.assertEqual(second_response.status_code, 200)

        # Both responses should have content
        first_response_body = json.loads(first_response.text)
        second_response_body = json.loads(second_response.text)

        self.assertIn("choices", first_response_body)
        self.assertIn("choices", second_response_body)
        self.assertIn("message", first_response_body["choices"][0])
        self.assertIn("message", second_response_body["choices"][0])
        self.assertIn("content", first_response_body["choices"][0]["message"])
        self.assertIn("content", second_response_body["choices"][0]["message"])

        # Ensure both generated content
        self.assertIsNotNone(first_response_body["choices"][0]["message"]["content"])
        self.assertIsNotNone(second_response_body["choices"][0]["message"]["content"])


class TestKeepalive(unittest.TestCase):
    def test_keepalive_callback(self):
        """Test keepalive callback sends SSE comments and handles errors"""
        from unittest.mock import Mock

        # Mock handler
        mock_wfile = io.BytesIO()
        handler = Mock()
        handler.wfile = mock_wfile

        # Test callback logic (same as in server.py)
        def keepalive_callback(processed_tokens, total_tokens):
            if handler.stream:
                try:
                    handler.wfile.write(
                        f": keepalive {processed_tokens}/{total_tokens}\n\n".encode()
                    )
                    handler.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass

        # Test streaming enabled
        handler.stream = True
        keepalive_callback(1024, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, ": keepalive 1024/4096\n\n")

        # Test streaming disabled
        handler.stream = False
        mock_wfile.seek(0)
        mock_wfile.truncate(0)
        keepalive_callback(2048, 4096)

        output = mock_wfile.getvalue().decode("utf-8")
        self.assertEqual(output, "")

        # Test error handling
        handler.stream = True
        handler.wfile = Mock()
        handler.wfile.write.side_effect = BrokenPipeError("Connection broken")

        # Should not raise exception
        try:
            keepalive_callback(3072, 4096)
        except Exception as e:
            self.fail(f"Callback should handle BrokenPipeError: {e}")


class TestLRUPromptCache(unittest.TestCase):
    def test_caching(self):
        cache = LRUPromptCache(max_size=10)

        def get_kv(n):
            keys = mx.arange(n).reshape(1, 1, n, 1)
            return keys, keys

        model = ("test", None, None)
        tokens = [10] * 24

        c, t = cache.fetch_nearest_cache(model, tokens)
        self.assertTrue(c is None)
        self.assertEqual(t, tokens)

        c = [KVCache()]
        c[0].update_and_fetch(*get_kv(24))
        cache.insert_cache(model, t, c)

        # Fetching a cache that is strictly a prefix doesn't remove it from the
        # lru cache
        tokens = tokens + [20] * 5
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].state
        self.assertTrue((k == v).all().item())
        self.assertTrue((k.flatten() == mx.arange(24)).all().item())
        self.assertEqual(t, [20] * 5)
        self.assertEqual(len(cache), 1)

        # Inserting a trimmable cache with shared prefix removes the prefixes
        tokens = tokens + [30] * 3
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)
        self.assertEqual(len(cache), 1)

        # Fetching a cache with a shared prefix doesn't remove it either
        tokens = tokens[:26] + [40] * 8
        c, t = cache.fetch_nearest_cache(model, tokens)
        k, v = c[0].state
        self.assertTrue((k == v).all().item())
        self.assertTrue(
            (k.flatten() == mx.concatenate([mx.arange(24), mx.arange(2)])).all().item()
        )
        self.assertEqual(t, [40] * 8)
        self.assertEqual(len(cache), 1)

        # Inserting a diverged cache actually creates another entry
        c[0].update_and_fetch(*get_kv(8))
        cache.insert_cache(model, tokens, c)
        self.assertEqual(len(cache), 2)

    def test_lru(self):
        cache = LRUPromptCache(max_size=2)
        model = ("test", None, None)
        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [1])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [1])
        c, t = cache.fetch_nearest_cache(model, [1, 3, 4])
        self.assertEqual(c, [MockCache("test1")])
        self.assertEqual(t, [3, 4])
        c, t = cache.fetch_nearest_cache(model, [2, 3, 4])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [4])
        c, t = cache.fetch_nearest_cache(model, [2, 4, 5])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [4, 5])

        cache.insert_cache(model, [1, 2], [MockCache("test1")])
        cache.insert_cache(model, [2, 3], [MockCache("test2")])
        cache.insert_cache(model, [3, 4], [MockCache("test3")])

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, [MockCache("test2")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [])

        cache.insert_cache(model, [4, 5], [MockCache("test4")], cache_type="user")
        c, t = cache.fetch_nearest_cache(model, [2, 3])
        self.assertEqual(c, None)
        self.assertEqual(t, [2, 3])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, [MockCache("test3")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [4, 5])
        self.assertEqual(c, [MockCache("test4")])
        self.assertEqual(t, [])

        cache.insert_cache(model, [5, 6], [MockCache("test5")])
        cache.insert_cache(model, [6, 7], [MockCache("test6")])
        c, t = cache.fetch_nearest_cache(model, [5, 6])
        self.assertEqual(c, None)
        self.assertEqual(t, [5, 6])
        c, t = cache.fetch_nearest_cache(model, [6, 7])
        self.assertEqual(c, [MockCache("test6")])
        self.assertEqual(t, [])
        c, t = cache.fetch_nearest_cache(model, [4, 5])
        self.assertEqual(c, [MockCache("test4")])
        self.assertEqual(t, [])

    def test_insert_trimmable_cache_removes_immediate_prefix(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("ab")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 2)

        cache.insert_cache(model, [1, 2, 3], [MockCache("abc")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 3)

    def test_insert_empty_tokens_does_not_self_destruct(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [], [MockCache("root")])
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.nbytes, 4)

        c, t = cache.fetch_nearest_cache(model, [])
        self.assertIsNotNone(c)
        self.assertEqual(t, [])

    def test_fetch_empty_tokens_after_root_eviction(self):
        cache = LRUPromptCache(max_size=10)
        model = ("test", None, None)

        cache.insert_cache(model, [], [MockCache("root")])
        cache.insert_cache(model, [1], [MockCache("a")])

        c, t = cache.fetch_nearest_cache(model, [])
        self.assertIsNone(c)
        self.assertEqual(t, [])

    def test_lru_bytes(self):
        cache = LRUPromptCache(max_size=100, max_bytes=10)
        model = ("test", None, None)

        cache.insert_cache(model, [1, 2], [MockCache("aaa")])
        cache.insert_cache(model, [3, 4], [MockCache("bbb")])
        cache.insert_cache(model, [4, 5], [MockCache("ccc")])
        cache.insert_cache(model, [6, 7], [MockCache("ddd")])

        self.assertEqual(len(cache), 3)
        self.assertEqual(cache.nbytes, 9)

        cache.trim_to(n_bytes=7)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.nbytes, 6)

        c, t = cache.fetch_nearest_cache(model, [1, 2])
        self.assertEqual(c, None)
        self.assertEqual(t, [1, 2])
        c, t = cache.fetch_nearest_cache(model, [3, 4])
        self.assertEqual(c, None)
        self.assertEqual(t, [3, 4])


if __name__ == "__main__":
    unittest.main()
