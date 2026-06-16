#!/usr/bin/env python3
"""Validate Velum private release metadata before publishing."""

import argparse
import json
import os
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VERSION_PATH = REPO_ROOT / "mlx_lm" / "_version.py"
SETUP_PATH = REPO_ROOT / "setup.py"
PROTOCOL_LOCK_PATH = REPO_ROOT / "mlx_lm" / "model_fusion_protocol.lock.json"
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "model-fusion-contract"
TAG_PATTERN = re.compile(r"^velum-mlx-lm-v(?P<version>[0-9][0-9A-Za-z.!\-+_]*)$")
PUBLIC_PYPI_HOSTS = ("pypi.org", "test.pypi.org", "upload.pypi.org")
GITHUB_PACKAGE_HOSTS = ("pkg.github.com", "npm.pkg.github.com")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="", help="Release tag to validate")
    parser.add_argument(
        "--require-tag",
        action="store_true",
        help="Require --tag to match the Velum release tag pattern",
    )
    args = parser.parse_args()

    version = read_version()
    if args.require_tag:
        validate_tag(args.tag, version)
    validate_setup_metadata()
    validate_protocol_lock()
    validate_publish_configuration()
    print("Velum release metadata validation passed.")
    return 0


def read_version() -> str:
    namespace = {}
    exec(VERSION_PATH.read_text(encoding="utf-8"), namespace)
    version = namespace.get("__version__")
    if not isinstance(version, str) or not version:
        raise SystemExit("mlx_lm/_version.py must define a non-empty __version__")
    return version


def validate_tag(tag: str, version: str) -> None:
    match = TAG_PATTERN.match(tag)
    if match is None:
        raise SystemExit(
            f"release tag must match velum-mlx-lm-v<version>; got {tag!r}"
        )
    tag_version = match.group("version")
    if tag_version != version:
        raise SystemExit(
            f"release tag version {tag_version!r} does not match package "
            f"version {version!r}"
        )


def validate_setup_metadata() -> None:
    setup_text = SETUP_PATH.read_text(encoding="utf-8")
    required_snippets = (
        'name="mlx-lm"',
        "version=__version__",
        '"mlx_lm": ["model_fusion_protocol.lock.json"]',
    )
    missing = [snippet for snippet in required_snippets if snippet not in setup_text]
    if missing:
        raise SystemExit(
            "setup.py is missing release metadata snippets: " + ", ".join(missing)
        )


def validate_protocol_lock() -> None:
    lock = json.loads(PROTOCOL_LOCK_PATH.read_text(encoding="utf-8"))
    expect(lock["origin"]["repo"] == "velum-labs/fusionkit", "fusionkit origin")
    expect(
        lock["v1_contracts"]["source_of_truth"] == "json_schema_openapi_3_1",
        "v1 JSON Schema/OpenAPI source of truth",
    )
    expect(
        lock["v1_contracts"]["durable_records"] == "json_schema",
        "JSON Schema durable record source",
    )
    expect(
        lock["v1_contracts"]["http_service_apis"] == "openapi_3_1",
        "OpenAPI 3.1 HTTP API source",
    )
    expect(
        lock["future_transports"]["protobuf_buf"]["required_for_v1"] is False,
        "protobuf/Buf not required for v1",
    )
    expect(
        lock["generated_packages"]["python"]["import_name"]
        == "velum_model_fusion_protocol",
        "Python generated package import name",
    )
    schema_hash = lock["schema_bundle"]["hash"]
    persisted_records = set(lock["schema_bundle"]["persisted_records"])
    fixture_schemas = {path.name for path in FIXTURE_ROOT.iterdir() if path.is_dir()}
    expect(
        fixture_schemas == persisted_records,
        "fixture schemas match persisted records",
    )
    for schema_name in fixture_schemas:
        for fixture_path in (FIXTURE_ROOT / schema_name).glob("*.json"):
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            expect(
                fixture.get("schema_bundle_hash") == schema_hash,
                f"{fixture_path} schema_bundle_hash matches protocol lock",
            )


def validate_publish_configuration() -> None:
    registry_url = os.environ.get("PRIVATE_PYPI_URL", "").strip()
    token = os.environ.get("PRIVATE_PYPI_TOKEN", "").strip()
    username = os.environ.get("PRIVATE_PYPI_USERNAME", "").strip()
    password = os.environ.get("PRIVATE_PYPI_PASSWORD", "").strip()
    allow_fallback = os.environ.get("ALLOW_GITHUB_RELEASE_FALLBACK") == "1"

    if registry_url:
        lowered = registry_url.lower()
        if any(host in lowered for host in PUBLIC_PYPI_HOSTS):
            raise SystemExit("PRIVATE_PYPI_URL must not point at public PyPI")
        if any(host in lowered for host in GITHUB_PACKAGE_HOSTS):
            raise SystemExit("GitHub Packages is not a PyPI-compatible target")
        if not (token or (username and password)):
            raise SystemExit(
                "PRIVATE_PYPI_URL requires PRIVATE_PYPI_TOKEN or "
                "PRIVATE_PYPI_USERNAME/PRIVATE_PYPI_PASSWORD"
            )
        return

    if not allow_fallback:
        raise SystemExit(
            "PRIVATE_PYPI_URL is unset; set ALLOW_GITHUB_RELEASE_FALLBACK=1 "
            "to upload wheel/sdist files to a GitHub Release instead"
        )


def expect(condition: bool, description: str) -> None:
    if not condition:
        raise SystemExit(f"release validation failed: {description}")


if __name__ == "__main__":
    raise SystemExit(main())
