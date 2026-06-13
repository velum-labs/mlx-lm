import threading
import unittest

import mlx.core as mx

from mlx_lm.embeddings import (
    EmbeddingError,
    EmbeddingModelError,
    EmbeddingModelUnavailableError,
    EmbeddingNotConfiguredError,
    EmbeddingProvider,
    masked_mean_pool,
    parse_embedding_request,
)


class FakeTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return list(range(1, len(text.split()) + 1))


class FakeEncoder:
    def __call__(self, input_ids):
        length = input_ids.shape[1]
        return mx.ones((1, length, 2))


class FakeModel:
    def __init__(self):
        self.model = FakeEncoder()


class BadModel:
    def __call__(self, input_ids):
        return mx.ones((1, 2, 3, 4))


class TestEmbeddingRequestParsing(unittest.TestCase):
    def test_rejects_non_object_body(self):
        with self.assertRaises(EmbeddingError):
            parse_embedding_request(["not", "an", "object"], configured_model="embed")

    def test_string_and_list_inputs(self):
        request = parse_embedding_request(
            {"model": "embed", "input": "hello"},
            configured_model="embed",
        )
        self.assertEqual(request.model, "embed")
        self.assertEqual(request.inputs, ["hello"])

        request = parse_embedding_request(
            {"model": "default_model", "input": ["a", "b"]},
            configured_model="embed",
        )
        self.assertEqual(request.model, "embed")
        self.assertEqual(request.inputs, ["a", "b"])

    def test_rejects_unsupported_fields(self):
        for body in (
            {"model": "embed", "input": "x", "dimensions": 128},
            {"model": "embed", "input": "x", "user": "u"},
        ):
            with self.subTest(body=body):
                with self.assertRaises(EmbeddingError):
                    parse_embedding_request(body, configured_model="embed")

    def test_rejects_unsupported_encoding_format(self):
        with self.assertRaises(EmbeddingError):
            parse_embedding_request(
                {"model": "embed", "input": "x", "encoding_format": "base64"},
                configured_model="embed",
            )

    def test_rejects_token_arrays_and_empty_inputs(self):
        for value in ([1, 2, 3], [], [["a"]]):
            with self.subTest(value=value):
                with self.assertRaises(EmbeddingError):
                    parse_embedding_request(
                        {"model": "embed", "input": value},
                        configured_model="embed",
                    )

    def test_rejects_unknown_model(self):
        with self.assertRaises(EmbeddingModelUnavailableError):
            parse_embedding_request(
                {"model": "other", "input": "x"},
                configured_model="embed",
            )


class TestEmbeddingProvider(unittest.TestCase):
    def test_masked_mean_pool_normalizes(self):
        hidden = mx.array([[[3.0, 0.0], [0.0, 4.0], [10.0, 10.0]]])
        mask = mx.array([[1.0, 1.0, 0.0]])

        pooled = masked_mean_pool(hidden, mask).tolist()[0]

        self.assertAlmostEqual(pooled[0], 0.6, places=5)
        self.assertAlmostEqual(pooled[1], 0.8, places=5)

    def test_lazy_loads_once_under_concurrency(self):
        calls = []

        def load_fn(*args, **kwargs):
            calls.append((args, kwargs))
            return FakeModel(), FakeTokenizer()

        provider = EmbeddingProvider(
            model_id="embed",
            tokenizer_config={},
            trust_remote_code=False,
            is_distributed=False,
            load_fn=load_fn,
        )
        outputs = []
        threads = [
            threading.Thread(target=lambda: outputs.append(provider.embed(["hello"])))
            for _ in range(4)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(calls), 1)
        self.assertEqual(len(outputs), 4)

    def test_not_configured(self):
        provider = EmbeddingProvider(None, {}, False, False, load_fn=None)

        with self.assertRaises(EmbeddingNotConfiguredError):
            provider.embed(["hello"])

    def test_bad_model_output_shape(self):
        provider = EmbeddingProvider(
            model_id="embed",
            tokenizer_config={},
            trust_remote_code=False,
            is_distributed=False,
            load_fn=lambda *args, **kwargs: (BadModel(), FakeTokenizer()),
        )

        with self.assertRaises(EmbeddingModelError):
            provider.embed(["hello"])


if __name__ == "__main__":
    unittest.main()
