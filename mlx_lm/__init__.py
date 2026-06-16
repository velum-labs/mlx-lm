# Copyright © 2023-2024 Apple Inc.

import importlib
import os

from ._version import __version__

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

_LAZY_EXPORTS = {
    "convert": ("convert", "convert"),
    "batch_generate": ("generate", "batch_generate"),
    "generate": ("generate", "generate"),
    "stream_generate": ("generate", "stream_generate"),
    "load": ("utils", "load"),
}


__all__ = [
    "__version__",
    "convert",
    "batch_generate",
    "generate",
    "stream_generate",
    "load",
]


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
