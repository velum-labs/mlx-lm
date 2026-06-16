# Model Fusion Protocol Consumption

`mlx-lm` is a provider implementation and consumer of the model-fusion
protocol. It is not the contract or IDL origin.

## Source of truth

Canonical packaging decision:
`velum-labs/openclaw-shared/spec/2026-06-16-model-fusion-protocol-packaging-spec.md`.
This repo records the decision in `model_fusion_protocol.lock.json` so CI can
catch accidental local drift even when the generated packages are not installed.

- Contract origin: `velum-labs/fusionkit`.
- JSON Schema bundle path in the origin repo: `contracts/model-fusion`.
- OpenAPI 3.1 path in the origin repo: `openapi/model-fusion.v1.yaml`.
- Current schema bundle hash consumed by this repo:
  `sha256:75792f89c091b6ab4fd317a15fb03fd73438563dceff5ccf9f5d7c752dbf35f3`.

`mlx_lm/model_fusion_protocol.lock.json` pins the fusionkit-origin artifact and
the generated package coordinates that the local compatibility helpers consume.
It is intentionally a lock file, not a new protocol repo. When generated Python
bindings are available, `mlx-lm` should import them instead of expanding local
hand-written validators.

## V1 contracts vs future transports

For v1, use JSON Schema as the source of truth for durable records and
OpenAPI 3.1 as the source of truth for HTTP/JSON service APIs. Generate the
TypeScript and Python SDK packages from those contracts. Keep JSON Schema as the
persisted record and audit/benchmark format for `model_endpoint.v1` and
`model-call-record.v1` JSONL.

Do not make protobuf/Buf a required v1 path in this repo. Protobuf/Buf is
reserved for later internal streaming, Connect, or gRPC if a service boundary
hardens enough to justify it.

Minimum fusionkit-origin OpenAPI 3.1 HTTP/JSON service surfaces:

- `HarnessExecutorService`: fusionkit to handoffkit coding task execution.
- `CursorHarnessService`: fusionkit to cursorkit adapter output.
- `MlxProviderService`: provider capability and model-call metadata for mlx-lm.
- Benchmark execution/join envelopes for fusionkit evals.

The concrete OpenAPI operations and schemas should be defined and versioned in
fusionkit. For `MlxProviderService`, HTTP responses can carry status and routing
metadata, but durable model endpoint and model call records should continue to
be emitted as schema-validated JSON/JSONL using the bundled JSON Schema
contracts.

## Generated packages

Fusionkit should publish generated artifacts from the same JSON Schema/OpenAPI
contract release:

- TypeScript: `@velum/model-fusion-protocol` on npm/GitHub Packages for TS
  consumers. It should expose generated OpenAPI client/types plus JSON Schema
  record validators. The pinned generator stack is `openapi-typescript` for
  service request/response types, `openapi-fetch` for the HTTP client, and `ajv`
  for JSON Schema record validation.
- Python: `velum_model_fusion_protocol` on a private PyPI-compatible registry.
  GitHub Packages is not sufficient for Python package consumption. Supported
  private-registry options should include Cloudsmith, AWS CodeArtifact, or
  Gemfury. It should expose generated OpenAPI client/models plus JSON
  Schema/Pydantic record validators. The pinned generator stack is
  `openapi-python-client` for service clients/models and
  `datamodel-code-generator` plus `pydantic` for JSON Schema record models and
  validators.
- Short-term Python fallback, before a private registry is ready: wheels
  attached to GitHub Releases, or a pinned `uv` git dependency that fetches the
  generated Python package from fusionkit.

Consumer repos should not hand-copy service interfaces, request/response
models, or schema-derived record types except for temporary fixtures that carry
clear provenance. Replace local compatibility shims with imports from generated
packages as soon as the fusionkit artifacts are published.

## Drift checks

The origin repo should run CI that:

1. validates the OpenAPI 3.1 HTTP/JSON service surface;
2. regenerates TS/Python SDK packages from JSON Schema/OpenAPI contracts;
3. regenerates JSON Schema bundle hashes for persisted audit/benchmark records;
4. fails if generated package artifacts, OpenAPI outputs, JSON Schema outputs,
   or bundle hashes differ from committed outputs; and
5. publishes TS/Python packages only from the exact same contract release.

This repo runs consumer-side checks that:

- the pinned lock names fusionkit as the origin;
- local fixture schema names match the persisted records listed in the lock;
- every bundled fixture uses the pinned schema bundle hash;
- `mlx_lm.openai_compat.MODEL_FUSION_SCHEMA_BUNDLE_HASH` comes from the lock;
- the lock says JSON Schema and OpenAPI 3.1 are the v1 source of truth;
- the lock says protobuf/Buf is future-facing and not required for v1;
- the lock pins OpenAPI and JSON Schema codegen stacks for TS/Python packages;
- release/PR validation fails if those codegen choices or drift-check strategy
  are removed from the lock;
- import-safe tests can load protocol metadata without importing `mlx`.

## Follow-up outside this repo

These items belong in fusionkit or the shared spec repo, not in mlx-lm:

- define the canonical OpenAPI 3.1 operations for `MlxProviderService`,
  `HarnessExecutorService`, `CursorHarnessService`, and benchmark envelopes;
- add OpenAPI validation and generated-code drift checks that regenerate
  TypeScript OpenAPI client/types, TypeScript JSON Schema validators, Python
  OpenAPI client/models, and Python JSON Schema/Pydantic validators, then fail
  if the working tree differs;
- publish `@velum/model-fusion-protocol`;
- publish `velum_model_fusion_protocol` to a private PyPI-compatible registry;
  and
- publish the JSON Schema bundle and hash from the same protocol release.

If protobuf/Buf becomes useful later for internal streaming, Connect, or gRPC,
that should be introduced as a separate hardening step in the protocol-owning
repo rather than as part of this mlx-lm provider import-safety PR.

The mlx-lm follow-up after those artifacts exist is small: replace the local
compatibility shim with imports from the generated Python package while keeping
the JSONL live smoke and fixture validation tests.
