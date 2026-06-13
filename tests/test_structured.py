"""Tests for mlx_lm.structured (constrained/structured decoding).

The spec-parsing tests are dependency-free. FSM, processor, and server-hook
tests require the `structured` extra (outlines-core) and, for the kernel
used by these tests, numpy+numba — they skip when those are not installed.
"""

import json
import pickle
import unittest

from mlx_lm.structured.spec import (
    JSON_OBJECT_SCHEMA,
    ConstraintSpecError,
    choices_to_regex,
    parse_constraint_spec,
)

try:
    import numpy as np

    from mlx_lm.structured.backends import get_backend
    from mlx_lm.structured.fsm import IndexCache, build_vocabulary, spec_to_regex
    from mlx_lm.structured.integration import (
        make_constraint_processor,
        parse_request_constraint,
    )
    from mlx_lm.structured.processor import StructuredLogitsProcessor
    from mlx_lm.structured.spec import ConstraintSpec

    HAS_STRUCTURED = True
except ImportError:
    HAS_STRUCTURED = False

try:
    import mlx.core as mx

    from mlx_lm.generate import BatchGenerator
    from mlx_lm.models import llama

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

if HAS_STRUCTURED:
    try:
        NUMPY_BACKEND = get_backend("numpy")
    except ImportError:  # numba (test-only dependency) not installed
        NUMPY_BACKEND = None
else:
    NUMPY_BACKEND = None

needs_backend = unittest.skipIf(
    NUMPY_BACKEND is None,
    "requires the structured extra (outlines-core) plus numpy and numba",
)

needs_mlx_structured = unittest.skipIf(
    not (HAS_MLX and HAS_STRUCTURED),
    "requires mlx and the structured extra (outlines-core)",
)

# A deliberately small vocabulary rich enough for small JSON documents and
# regexes; multi-character tokens exercise the multi-token-per-string paths.
TOKEN_STRINGS = [
    "{", "}", '"', ":", ",", " ",
    "a", "b", "c", "0", "1", "2",
    "name", "age", "[", "]", "ab", "12", "x",
]
TOKEN_IDS = {s: i for i, s in enumerate(TOKEN_STRINGS)}
EOS_ID = 19
SECONDARY_EOS_ID = 20
VOCAB_SIZE = 21
PROMPT = [TOKEN_IDS["x"], TOKEN_IDS["x"], TOKEN_IDS["x"]]


class FakeTokenizer:
    """Mimics the slice of TokenizerWrapper the structured package uses."""

    def __init__(self):
        self.eos_token_id = EOS_ID
        self.eos_token_ids = {EOS_ID, SECONDARY_EOS_ID}
        self._vocab = {s: i for i, s in enumerate(TOKEN_STRINGS)}
        self._vocab["</s>"] = EOS_ID
        self._vocab["<|stop|>"] = SECONDARY_EOS_ID

    def get_vocab(self):
        return dict(self._vocab)

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)


class TestSpecParsing(unittest.TestCase):
    SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}}

    def test_no_constraint_fields(self):
        self.assertIsNone(parse_constraint_spec({}))
        self.assertIsNone(parse_constraint_spec({"temperature": 0.2}))
        self.assertIsNone(parse_constraint_spec({"response_format": {"type": "text"}}))

    def test_response_format_json_object(self):
        spec = parse_constraint_spec({"response_format": {"type": "json_object"}})
        self.assertEqual(spec.kind, "json_schema")
        self.assertEqual(spec.payload, JSON_OBJECT_SCHEMA)

    def test_response_format_json_schema_openai_shape(self):
        body = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "thing", "schema": self.SCHEMA},
            }
        }
        spec = parse_constraint_spec(body)
        self.assertEqual(spec.kind, "json_schema")
        self.assertEqual(json.loads(spec.payload), self.SCHEMA)

    def test_response_format_errors(self):
        for body in (
            {"response_format": {"type": "json_schema"}},
            {"response_format": {"type": "yaml"}},
            {"response_format": "json"},
        ):
            with self.assertRaises(ConstraintSpecError):
                parse_constraint_spec(body)

    def test_guided_json_dict_and_string_are_equivalent(self):
        a = parse_constraint_spec({"guided_json": self.SCHEMA})
        b = parse_constraint_spec({"guided_json": json.dumps(self.SCHEMA)})
        self.assertEqual(a, b)

    def test_guided_json_errors(self):
        for value in ("{nope", True, 3):
            with self.assertRaises(ConstraintSpecError):
                parse_constraint_spec({"guided_json": value})

    def test_guided_regex(self):
        spec = parse_constraint_spec({"guided_regex": "[0-9]+"})
        self.assertEqual((spec.kind, spec.payload), ("regex", "[0-9]+"))
        with self.assertRaises(ConstraintSpecError):
            parse_constraint_spec({"guided_regex": ""})

    def test_guided_choice(self):
        spec = parse_constraint_spec({"guided_choice": ["yes", "no", 3]})
        self.assertEqual(spec.kind, "choice")
        self.assertEqual(spec.choices(), ["yes", "no", "3"])
        for value in ([], [""], [{"a": 1}]):
            with self.assertRaises(ConstraintSpecError):
                parse_constraint_spec({"guided_choice": value})

    def test_conflicting_constraints_rejected(self):
        with self.assertRaises(ConstraintSpecError):
            parse_constraint_spec(
                {"guided_regex": "a+", "response_format": {"type": "json_object"}}
            )
        # A "text" response_format does not conflict.
        spec = parse_constraint_spec(
            {"response_format": {"type": "text"}, "guided_regex": "a+"}
        )
        self.assertEqual(spec.kind, "regex")

    def test_cache_key_is_canonical(self):
        a = parse_constraint_spec({"guided_json": {"type": "object", "properties": {}}})
        b = parse_constraint_spec(
            {"guided_json": '{"properties": {}, "type": "object"}'}
        )
        self.assertEqual(a.cache_key, b.cache_key)

    def test_choices_to_regex_escapes_metacharacters(self):
        self.assertEqual(choices_to_regex(["a.b", "c|d"]), r"(a\.b|c\|d)")


@needs_backend
class TestFsmCompilation(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_spec_to_regex_json_schema(self):
        import re

        regex = spec_to_regex(
            ConstraintSpec(kind="json_schema", payload='{"type": "integer"}')
        )
        self.assertTrue(re.fullmatch(regex, "42"))
        self.assertFalse(re.fullmatch(regex, '"x"'))

    def test_spec_to_regex_invalid_schema(self):
        with self.assertRaises(ConstraintSpecError):
            spec_to_regex(
                ConstraintSpec(kind="json_schema", payload='{"type": "nonsense"}')
            )

    def test_build_vocabulary_excludes_all_eos_ids(self):
        _, eos_id, eos_ids = build_vocabulary(self.tokenizer)
        self.assertEqual(eos_id, EOS_ID)
        self.assertEqual(eos_ids, sorted({EOS_ID, SECONDARY_EOS_ID}))

    def test_index_cache_reuse_and_keying(self):
        cache = IndexCache()
        spec_a = ConstraintSpec(kind="regex", payload="ab")
        spec_b = ConstraintSpec(kind="regex", payload="ba")
        a1, _, _ = cache.index(("m1", None), self.tokenizer, spec_a)
        a2, _, _ = cache.index(("m1", None), self.tokenizer, spec_a)
        b, _, _ = cache.index(("m1", None), self.tokenizer, spec_b)
        other_model, _, _ = cache.index(("m2", None), self.tokenizer, spec_a)
        self.assertIs(a1, a2)
        self.assertIsNot(a1, b)
        self.assertIsNot(a1, other_model)

    def test_index_cache_compile_error_is_spec_error(self):
        cache = IndexCache()
        with self.assertRaises(ConstraintSpecError):
            cache.index(
                ("m", None),
                self.tokenizer,
                ConstraintSpec(kind="regex", payload="(?=a)a"),
            )


@needs_backend
class TestProcessor(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.cache = IndexCache()

    def make_processor(self, kind, payload):
        spec = ConstraintSpec(kind=kind, payload=payload)
        index, _, eos_ids = self.cache.index(("m", None), self.tokenizer, spec)
        return StructuredLogitsProcessor(index, eos_ids, backend=NUMPY_BACKEND)

    def allowed(self, processor, history):
        logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        out = processor(np.array(history, dtype=np.int64), logits)
        return set(np.flatnonzero(out[0] > -np.inf).tolist())

    def test_regex_walk_append_only(self):
        p = self.make_processor("regex", "abc")
        self.assertEqual(self.allowed(p, PROMPT), {TOKEN_IDS["a"], TOKEN_IDS["ab"]})
        self.assertEqual(
            self.allowed(p, PROMPT + [TOKEN_IDS["a"]]), {TOKEN_IDS["b"]}
        )
        self.assertEqual(
            self.allowed(p, PROMPT + [TOKEN_IDS["a"], TOKEN_IDS["b"]]),
            {TOKEN_IDS["c"]},
        )
        done = self.allowed(
            p, PROMPT + [TOKEN_IDS["a"], TOKEN_IDS["b"], TOKEN_IDS["c"]]
        )
        self.assertEqual(done, {EOS_ID})

    def test_multichar_token_path(self):
        p = self.make_processor("regex", "abc")
        self.allowed(p, PROMPT)
        self.assertEqual(self.allowed(p, PROMPT + [TOKEN_IDS["ab"]]), {TOKEN_IDS["c"]})

    def test_eos_in_history_masks_all_but_eos(self):
        p = self.make_processor("regex", "abc")
        self.allowed(p, PROMPT)
        full = PROMPT + [TOKEN_IDS["a"], TOKEN_IDS["b"], TOKEN_IDS["c"], EOS_ID]
        self.assertEqual(self.allowed(p, full), {EOS_ID, SECONDARY_EOS_ID})

    def test_unconsumable_token_degrades_to_eos(self):
        p = self.make_processor("regex", "abc")
        self.allowed(p, PROMPT)
        self.assertEqual(
            self.allowed(p, PROMPT + [TOKEN_IDS["x"]]), {EOS_ID, SECONDARY_EOS_ID}
        )

    def test_speculative_rewind_resync(self):
        p = self.make_processor("regex", "a(b|c)c")
        base = PROMPT + [TOKEN_IDS["a"]]
        self.allowed(p, PROMPT)
        self.assertEqual(self.allowed(p, base), {TOKEN_IDS["b"], TOKEN_IDS["c"]})
        self.assertEqual(self.allowed(p, base + [TOKEN_IDS["b"]]), {TOKEN_IDS["c"]})
        self.assertEqual(
            self.allowed(p, base + [TOKEN_IDS["b"], TOKEN_IDS["c"]]), {EOS_ID}
        )
        # The drafted 'b' is rejected; the engine rewinds and takes 'c'.
        self.assertEqual(self.allowed(p, base + [TOKEN_IDS["c"]]), {TOKEN_IDS["c"]})
        self.assertEqual(
            self.allowed(p, base + [TOKEN_IDS["c"], TOKEN_IDS["c"]]), {EOS_ID}
        )

    def test_rewound_eos_unfinishes(self):
        p = self.make_processor("regex", "ab?")
        self.allowed(p, PROMPT)
        done = self.allowed(p, PROMPT + [TOKEN_IDS["a"], EOS_ID])
        self.assertEqual(done, {EOS_ID, SECONDARY_EOS_ID})
        after = self.allowed(p, PROMPT + [TOKEN_IDS["a"], TOKEN_IDS["b"]])
        self.assertEqual(after, {EOS_ID})

    def test_rollback_overflow_falls_back_to_replay(self):
        p = self.make_processor("regex", "a*bc")
        self.allowed(p, PROMPT)
        self.assertIn(
            TOKEN_IDS["b"], self.allowed(p, PROMPT + [TOKEN_IDS["a"]] * 100)
        )
        rewound = PROMPT + [TOKEN_IDS["a"]] * 20 + [TOKEN_IDS["b"]]
        self.assertEqual(self.allowed(p, rewound), {TOKEN_IDS["c"]})

    def test_json_schema_end_to_end_greedy(self):
        schema = json.dumps(
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            }
        )
        p = self.make_processor("json_schema", schema)
        target = '{"name": "ab"}'
        history, emitted = list(PROMPT), ""
        for _ in range(40):
            allowed_ids = self.allowed(p, history)
            if emitted == target:
                self.assertEqual(allowed_ids, {EOS_ID})
                break
            remaining = target[len(emitted):]
            candidates = [
                tid
                for tid in allowed_ids
                if tid < len(TOKEN_STRINGS)
                and remaining.startswith(TOKEN_STRINGS[tid])
            ]
            self.assertTrue(candidates, f"constraint blocked target at {emitted!r}")
            token_id = max(candidates, key=lambda tid: len(TOKEN_STRINGS[tid]))
            history.append(token_id)
            emitted += TOKEN_STRINGS[token_id]
        else:
            self.fail("did not finish the target document")
        self.assertEqual(json.loads(emitted), {"name": "ab"})

    def test_does_not_mutate_input_logits(self):
        p = self.make_processor("regex", "ab")
        logits = np.zeros((1, VOCAB_SIZE), dtype=np.float32)
        p(np.array(PROMPT, dtype=np.int64), logits)
        self.assertTrue(np.all(logits == 0.0))


@needs_backend
class TestIntegrationSurface(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def test_parse_request_constraint(self):
        self.assertIsNone(parse_request_constraint({"messages": []}))
        spec = parse_request_constraint({"guided_regex": "ab"})
        self.assertEqual(spec.kind, "regex")
        # ConstraintSpecError must remain a ValueError: the server catches
        # ValueError to produce its HTTP 400.
        with self.assertRaises(ValueError):
            parse_request_constraint({"response_format": {"type": "yaml"}})
        with self.assertRaises(ConstraintSpecError):
            parse_request_constraint({"guided_json": {"type": "nonsense"}})

    def test_make_constraint_processor(self):
        spec = parse_request_constraint({"guided_regex": "ab"})
        p1 = make_constraint_processor(spec, self.tokenizer, ("model-int", None))
        p2 = make_constraint_processor(spec, self.tokenizer, ("model-int", None))
        self.assertIsNot(p1, p2)
        self.assertIs(p1._index, p2._index)


class TestServerHooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mlx_lm.server as mlx_server
        except ImportError:
            raise unittest.SkipTest("mlx is required to import mlx_lm.server")
        cls.server = mlx_server

    def make_args(self, structured):
        s = self.server
        return s.GenerationArguments(
            model=s.ModelDescription(model="m", draft="d", adapter=None),
            sampling=s.SamplingArguments(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                xtc_probability=0.0,
                xtc_threshold=0.0,
            ),
            logits=s.LogitsProcessorArguments(
                logit_bias=None,
                repetition_penalty=0.0,
                repetition_context_size=20,
                presence_penalty=0.0,
                presence_context_size=20,
                frequency_penalty=0.0,
                frequency_context_size=20,
                structured=structured,
            ),
            stop_words=[],
            max_tokens=16,
            num_draft_tokens=0,
            logprobs=False,
            top_logprobs=-1,
            seed=None,
            chat_template_kwargs=None,
        )

    def test_make_logits_processors_without_constraint(self):
        processors = self.server._make_logits_processors(
            self.make_args(None), FakeTokenizer(), ("m", None)
        )
        self.assertEqual(processors, [])

    @needs_backend
    def test_hooks_bound_and_constraint_appended(self):
        self.assertIsNotNone(self.server.parse_request_constraint)
        spec = self.server.parse_request_constraint({"guided_regex": "ab"})
        processors = self.server._make_logits_processors(
            self.make_args(spec), FakeTokenizer(), ("hook-model", None)
        )
        self.assertEqual(len(processors), 1)
        self.assertIsInstance(processors[0], StructuredLogitsProcessor)

    @needs_backend
    def test_structured_args_survive_pickling(self):
        spec = self.server.parse_request_constraint({"guided_regex": "a+"})
        args = self.make_args(spec)
        restored = pickle.loads(pickle.dumps(args))
        self.assertEqual(restored.logits.structured, spec)


def _decode(tokens):
    return "".join(
        TOKEN_STRINGS[t] for t in tokens if t not in (EOS_ID, SECONDARY_EOS_ID)
    )


@needs_mlx_structured
class TestBatchGeneratorConstraints(unittest.TestCase):
    """Constraint enforcement on a long-lived BatchGenerator.

    mlx_lm.server keeps one BatchGenerator alive across requests and inserts
    new requests into it, so a constrained request must be fully enforced no
    matter what ran on the generator before it. The plain-then-structured
    cases are regressions for a bug in GenerationBatch.filter: when a batch
    whose sequences had no logits processors drained, the stale processor
    slots were left behind and misaligned the processors of sequences
    inserted later, silently disabling their constraints.
    """

    SCHEMA = json.dumps(
        {
            "type": "object",
            "properties": {"name": {"type": "string", "maxLength": 2}},
            "required": ["name"],
            "additionalProperties": False,
        },
        sort_keys=True,
        separators=(",", ":"),
    )

    @classmethod
    def setUpClass(cls):
        mx.random.seed(7)
        args = llama.ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            rms_norm_eps=1e-5,
            vocab_size=VOCAB_SIZE,
        )
        cls.model = llama.Model(args)
        cls.tokenizer = FakeTokenizer()
        cls.index_cache = IndexCache()

    def _generator(self):
        return BatchGenerator(
            self.model,
            stop_tokens=[[EOS_ID], [SECONDARY_EOS_ID]],
            max_tokens=24,
        )

    def _constraint(self, kind, payload):
        spec = ConstraintSpec(kind=kind, payload=payload)
        index, _, eos_ids = self.index_cache.index(("tiny", None), self.tokenizer, spec)
        return StructuredLogitsProcessor(index, eos_ids)

    def _run(self, gen, processors=None):
        """Insert one request, run it to completion, return its tokens."""
        kwargs = {}
        if processors is not None:
            kwargs["logits_processors"] = [processors]
        (uid,) = gen.insert([list(PROMPT)], max_tokens=[24], **kwargs)
        tokens = []
        for _ in range(200):
            responses = gen.next_generated()
            self.assertTrue(responses, "generator stalled before request finished")
            for r in responses:
                self.assertEqual(r.uid, uid)
                tokens.append(r.token)
                if r.finish_reason is not None:
                    return tokens
        self.fail("request did not finish")

    def assertMatchesRegexConstraint(self, tokens, text):
        self.assertEqual(_decode(tokens), text)
        self.assertIn(tokens[-1], (EOS_ID, SECONDARY_EOS_ID))

    def test_structured_only(self):
        gen = self._generator()
        tokens = self._run(gen, [self._constraint("regex", "abc")])
        self.assertMatchesRegexConstraint(tokens, "abc")

    def test_plain_then_structured(self):
        gen = self._generator()
        self._run(gen)
        tokens = self._run(gen, [self._constraint("regex", "abc")])
        self.assertMatchesRegexConstraint(tokens, "abc")

    def test_structured_then_plain(self):
        gen = self._generator()
        tokens = self._run(gen, [self._constraint("regex", "abc")])
        self.assertMatchesRegexConstraint(tokens, "abc")
        plain_tokens = self._run(gen)
        self.assertTrue(plain_tokens)

    def test_structured_then_structured_same_schema(self):
        gen = self._generator()
        for _ in range(2):
            tokens = self._run(gen, [self._constraint("regex", "abc")])
            self.assertMatchesRegexConstraint(tokens, "abc")

    def test_structured_then_structured_different_schema(self):
        gen = self._generator()
        tokens = self._run(gen, [self._constraint("regex", "abc")])
        self.assertMatchesRegexConstraint(tokens, "abc")
        tokens = self._run(gen, [self._constraint("choice", '["ab","12"]')])
        self.assertIn(_decode(tokens), ("ab", "12"))

    def test_plain_then_structured_json_schema(self):
        # The reported bug: after a plain request, a json_schema-constrained
        # request emitted invalid JSON like '{The capital of France is
        # Paris.}' because only the first token was constrained.
        gen = self._generator()
        self._run(gen)
        tokens = self._run(gen, [self._constraint("json_schema", self.SCHEMA)])
        decoded = _decode(tokens)
        try:
            document = json.loads(decoded)
        except json.JSONDecodeError:
            self.fail(f"constrained output is not valid JSON: {decoded!r}")
        self.assertIsInstance(document, dict)
        self.assertIn("name", document)


if __name__ == "__main__":
    unittest.main()
