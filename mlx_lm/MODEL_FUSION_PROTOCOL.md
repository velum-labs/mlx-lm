# Model Fusion Protocol Consumption

`mlx-lm` is a provider implementation and consumer of the model-fusion
protocol. It is not the contract or IDL origin.

## Source of truth

- Contract/IDL origin: `velum-labs/fusionkit`.
- JSON Schema bundle path in the origin repo: `contracts/model-fusion`.
- Protobuf/Buf path in the origin repo: `proto/velum/model_fusion/v1`.
- Buf module: `buf.build/velum/model-fusion`.
- Current schema bundle hash consumed by this repo:
  `sha256:75792f89c091b6ab4fd317a15fb03fd73438563dceff5ccf9f5d7c752dbf35f3`.

`mlx_lm/model_fusion_protocol.lock.json` pins the fusionkit-origin artifact that
the local compatibility helpers consume. It is intentionally a lock file, not a
new protocol repo. When generated Python bindings are available, `mlx-lm` should
import them instead of expanding local hand-written validators.

## Transport IDL vs persisted records

Use protobuf/Buf for service and transport boundaries. Keep JSON Schema as the
persisted record and audit format for `model_endpoint.v1` and
`model-call-record.v1` JSONL.

Minimum fusionkit-origin service boundaries:

```proto
syntax = "proto3";

package velum.model_fusion.v1;

service HarnessExecutorService {
  rpc ExecuteCodingTask(ExecuteCodingTaskRequest)
      returns (ExecuteCodingTaskResponse);
}

service CursorHarnessService {
  rpc ProduceAdapterOutput(ProduceAdapterOutputRequest)
      returns (ProduceAdapterOutputResponse);
}

service MlxProviderService {
  rpc GetCapabilities(GetCapabilitiesRequest)
      returns (GetCapabilitiesResponse);
  rpc RecordModelCall(RecordModelCallRequest)
      returns (RecordModelCallResponse);
}

message BenchmarkExecutionEnvelope {
  string task_id = 1;
  string endpoint_id = 2;
  string schema_bundle_hash = 3;
}

message BenchmarkJoinEnvelope {
  string task_id = 1;
  string run_id = 2;
  string schema_bundle_hash = 3;
}
```

The concrete messages above should be defined and versioned in fusionkit. For
`MlxProviderService`, the transport response can embed protobuf fields for
status and routing metadata, but durable model endpoint and model call records
should continue to be emitted as schema-validated JSON/JSONL using the bundled
JSON Schema contracts.

## Generated packages

Fusionkit should publish generated artifacts from the same Buf/JSON Schema
bundle:

- TypeScript: `@velum/model-fusion-protocol` on npm/GitHub Packages for TS
  consumers.
- Python: `velum_model_fusion_protocol` on a private PyPI-compatible registry.
  GitHub Packages is not sufficient for Python package consumption. Supported
  private-registry options should include Cloudsmith, AWS CodeArtifact, or
  Gemfury.
- Short-term Python fallback, before a private registry is ready: wheels
  attached to GitHub Releases, or a pinned `uv` git dependency that fetches the
  generated Python package from fusionkit.

## Drift checks

The origin repo should run CI that:

1. runs `buf lint` and `buf breaking`;
2. regenerates TS/Python bindings;
3. regenerates JSON Schema bundle hashes for persisted records;
4. fails if generated package artifacts, JSON Schema outputs, or bundle hashes
   differ from committed outputs; and
5. publishes TS/Python packages only from the exact same protocol bundle.

This repo runs consumer-side checks that:

- the pinned lock names fusionkit as the origin;
- local fixture schema names match the persisted records listed in the lock;
- every bundled fixture uses the pinned schema bundle hash;
- `mlx_lm.openai_compat.MODEL_FUSION_SCHEMA_BUNDLE_HASH` comes from the lock;
  and
- import-safe tests can load protocol metadata without importing `mlx`.
