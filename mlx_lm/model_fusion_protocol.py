"""Pinned model-fusion protocol artifact metadata.

The contract and IDL source of truth lives in fusionkit. This module is an
import-safe consumer shim for mlx-lm until the generated Python package is
available from a private package index.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple


MODEL_FUSION_PROTOCOL_LOCK_PATH = (
    Path(__file__).with_name("model_fusion_protocol.lock.json")
)


def load_model_fusion_protocol_lock() -> Dict[str, Any]:
    with MODEL_FUSION_PROTOCOL_LOCK_PATH.open(encoding="utf-8") as handle:
        lock = json.load(handle)
    if not isinstance(lock, dict):
        raise ValueError("model-fusion protocol lock must be a JSON object")
    return lock


MODEL_FUSION_PROTOCOL_LOCK = load_model_fusion_protocol_lock()
MODEL_FUSION_CANONICAL_SPEC = MODEL_FUSION_PROTOCOL_LOCK["canonical_spec"]
MODEL_FUSION_IDL_SOURCE_OF_TRUTH = MODEL_FUSION_PROTOCOL_LOCK["idl"][
    "source_of_truth"
]
MODEL_FUSION_OPENAPI_STATUS = MODEL_FUSION_PROTOCOL_LOCK["idl"]["openapi"]["status"]
MODEL_FUSION_OPENAPI_HAND_AUTHORED = MODEL_FUSION_PROTOCOL_LOCK["idl"]["openapi"][
    "hand_authored"
]
MODEL_FUSION_SCHEMA_BUNDLE_HASH = MODEL_FUSION_PROTOCOL_LOCK["schema_bundle"]["hash"]
MODEL_FUSION_SCHEMA_BUNDLE_PURPOSE = MODEL_FUSION_PROTOCOL_LOCK["schema_bundle"][
    "purpose"
]
MODEL_FUSION_PERSISTED_RECORDS: Tuple[str, ...] = tuple(
    MODEL_FUSION_PROTOCOL_LOCK["schema_bundle"]["persisted_records"]
)
MODEL_FUSION_TYPESCRIPT_PACKAGE = MODEL_FUSION_PROTOCOL_LOCK["generated_packages"][
    "typescript"
]["package"]
MODEL_FUSION_PYTHON_IMPORT_NAME = MODEL_FUSION_PROTOCOL_LOCK["generated_packages"][
    "python"
]["import_name"]
MODEL_FUSION_SERVICE_BOUNDARIES: Tuple[str, ...] = tuple(
    MODEL_FUSION_PROTOCOL_LOCK["service_boundaries"]
)
