"""Structured/constrained decoding for the mlx-lm server.

Enforces OpenAI ``response_format`` and vLLM-style ``guided_json`` /
``guided_regex`` / ``guided_choice`` request parameters by masking logits
with an FSM compiled by outlines-core, so the model can only emit tokens
that keep the output valid.

Requires the ``structured`` extra (``pip install mlx-lm[structured]``); the
server activates the hooks when this package imports successfully and
behaves like stock mlx-lm otherwise.
"""

from mlx_lm.structured.spec import (
    ConstraintSpec,
    ConstraintSpecError,
    parse_constraint_spec,
)

# Everything beyond spec parsing needs outlines-core (the structured extra);
# without it this package still imports so the pure parts stay usable and
# the server's optional hook import fails cleanly at the integration module.
try:
    from mlx_lm.structured.integration import (
        make_constraint_processor,
        parse_request_constraint,
    )
    from mlx_lm.structured.processor import StructuredLogitsProcessor
except ImportError:
    make_constraint_processor = None
    parse_request_constraint = None
    StructuredLogitsProcessor = None

__all__ = [
    "ConstraintSpec",
    "ConstraintSpecError",
    "StructuredLogitsProcessor",
    "make_constraint_processor",
    "parse_constraint_spec",
    "parse_request_constraint",
]
