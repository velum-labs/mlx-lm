# Velum private release process

This fork keeps the upstream public PyPI workflow gated to
`ml-explore/mlx-lm`. Velum releases use
`.github/workflows/velum-private-release.yml`, which is guarded with:

```yaml
if: github.repository == 'velum-labs/mlx-lm'
```

It will not publish from forks, and it will not publish to public PyPI.

## Tag pattern

Publish Velum private releases by pushing an explicit tag:

```shell
git tag velum-mlx-lm-v<package-version>
git push origin velum-mlx-lm-v<package-version>
```

Example:

```shell
git tag velum-mlx-lm-v0.27.0
git push origin velum-mlx-lm-v0.27.0
```

The tag version must match `mlx_lm/_version.py`. The release validation step
fails before publishing if the tag does not match.

`workflow_dispatch` is available for dry-run build/test/package validation. It
does not publish; only `velum-mlx-lm-v*` tag pushes can run the publish job.

## Required secrets for private PyPI publishing

Configure these repository or environment secrets in `velum-labs/mlx-lm`:

- `PRIVATE_PYPI_URL`: upload endpoint for a private PyPI-compatible registry,
  such as Cloudsmith, AWS CodeArtifact, or Gemfury.
- `PRIVATE_PYPI_TOKEN`: preferred token credential. The workflow publishes with
  username `__token__` when this is set.

Alternatively, use username/password credentials:

- `PRIVATE_PYPI_USERNAME`
- `PRIVATE_PYPI_PASSWORD`

Do not point `PRIVATE_PYPI_URL` at public PyPI/TestPyPI or GitHub Packages.
GitHub Packages is not a PyPI-compatible Python package registry. The workflow
and `scripts/validate_velum_release.py` refuse those targets.

## Fallback when registry secrets are absent

If `PRIVATE_PYPI_URL` or credentials are absent, the publish job creates or
updates a **draft GitHub Release** for the tag and uploads the wheel/sdist files
there. This keeps releases private to repository access and avoids accidental
public PyPI publishing.

## Release validation

Before building and publishing, the workflow:

1. compiles the import-safe release modules;
2. runs the non-MLX import-safe test suite;
3. validates release metadata with `scripts/validate_velum_release.py`;
4. builds wheel and sdist artifacts with `python -m build`.

The validation script checks:

- tag pattern and package version match;
- `setup.py` package metadata includes the model-fusion protocol lock as package
  data;
- the pinned protocol lock still names fusionkit as origin;
- v1 protocol decisions remain JSON Schema + OpenAPI 3.1, with protobuf/Buf not
  required for v1;
- generated protocol package coordinates remain present;
- generated protocol package codegen choices remain pinned:
  `openapi-typescript` + `openapi-fetch` + `ajv` for TypeScript, and
  `openapi-python-client` + `datamodel-code-generator` + `pydantic` for Python;
- generated-code drift strategy remains `regenerate_and_fail_on_diff`;
- bundled fixture schema names and schema bundle hashes match the protocol lock;
  and
- private PyPI settings are safe, or GitHub Release fallback is explicitly
  allowed.

## TypeScript protocol package note

`mlx-lm` does not publish a TypeScript package. The TypeScript package
`@velum/model-fusion-protocol` should be published by fusionkit, the protocol
origin, using GitHub Packages/npm provenance from the JSON Schema/OpenAPI
contract bundle. It should expose generated OpenAPI client/types and generated
JSON Schema validators, not hand-written service interfaces. This repository
only pins the expected package coordinates and generator stack, and uses
Python-side import-safe validation until generated Python protocol models are
available.
