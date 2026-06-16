#!/usr/bin/env python3
"""Apple Silicon/MLX live smoke gate for model-fusion provider records."""

import datetime
import hashlib
import importlib.util
import json
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mlx_lm.openai_compat import (
    MODEL_FUSION_SCHEMA_BUNDLE_HASH,
    validate_model_call_record_fixture,
)


RUN_FLAG = "MLX_LM_RUN_LIVE_SMOKE"
MODEL_ENV = "MLX_LM_SMOKE_MODEL"
JSONL_ENV = "MLX_LM_SMOKE_JSONL"
HOST_ENV = "MLX_LM_SMOKE_HOST"
PORT_ENV = "MLX_LM_SMOKE_PORT"
TIMEOUT_ENV = "MLX_LM_SMOKE_TIMEOUT_SECONDS"

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
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


def main() -> int:
    if os.environ.get(RUN_FLAG) != "1":
        return skip(f"set {RUN_FLAG}=1 to run the live Apple Silicon/MLX smoke")
    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
        return skip("Apple Silicon is required")
    if importlib.util.find_spec("mlx.core") is None:
        return skip("mlx is not installed")

    model = os.environ.get(MODEL_ENV)
    if not model:
        return skip(f"set {MODEL_ENV} to a local MLX model path or Hugging Face repo")

    host = os.environ.get(HOST_ENV, "127.0.0.1")
    port = int(os.environ.get(PORT_ENV) or free_port(host))
    timeout_seconds = float(os.environ.get(TIMEOUT_ENV, "120"))
    jsonl_path = Path(
        os.environ.get(JSONL_ENV, "/tmp/mlx-lm-model-fusion-live-smoke.jsonl")
    )

    proc = start_server(host, port, model)
    try:
        wait_for_health(proc, host, port, timeout_seconds)
        capabilities = request_json("GET", f"http://{host}:{port}/v1/capabilities")
        if capabilities["capabilities"].get("tool_calls") != "supported":
            return skip("mlx-lm[structured] is required for forced tool-call smoke")

        record = run_tool_call_request(host, port, model, capabilities)
        write_and_validate_jsonl(jsonl_path, record)
        print(f"PASS: wrote validated model-call-record.v1 JSONL to {jsonl_path}")
        return 0
    finally:
        stop_server(proc)


def skip(message: str) -> int:
    print(f"SKIP: {message}")
    return 0


def utc_now() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def contract_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def producer_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return "0" * 40
    sha = result.stdout.strip().lower()
    if len(sha) == 40 and all(char in "0123456789abcdef" for char in sha):
        return sha
    return "0" * 40


def compact_dict(value: Dict[str, Any]) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def start_server(host: str, port: int, model: str) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.server",
        "--model",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--max-tokens",
        "128",
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


def wait_for_health(
    proc: subprocess.Popen, host: str, port: int, timeout_seconds: float
) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://{host}:{port}/v1/health"
    last_error: Optional[BaseException] = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = read_process_output(proc)
            raise RuntimeError(
                f"server exited before health check passed with code "
                f"{proc.returncode}\n{output}"
            )
        try:
            request_json("GET", url, timeout=5)
            return
        except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
            last_error = e
            time.sleep(0.5)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def read_process_output(proc: subprocess.Popen) -> str:
    if proc.stdout is None:
        return ""
    try:
        output, _ = proc.communicate(timeout=2)
    except subprocess.TimeoutExpired:
        return ""
    return output


def request_json(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30,
) -> Dict[str, Any]:
    data = None
    request_headers = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body}") from e
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"expected JSON object from {url}")
    return parsed


def run_tool_call_request(
    host: str, port: int, model: str, capabilities: Dict[str, Any]
) -> Dict[str, Any]:
    messages = [{"role": "user", "content": "What is the weather in Paris?"}]
    payload = {
        "model": "default_model",
        "messages": messages,
        "tools": [TOOL],
        "tool_choice": "required",
        "max_tokens": 64,
    }
    call_id = f"call_live_smoke_{uuid.uuid4().hex}"
    headers = {"x-velum-model-call-id": call_id}
    started_at = utc_now()
    start = time.perf_counter()
    response = request_json(
        "POST",
        f"http://{host}:{port}/v1/chat/completions",
        payload,
        headers=headers,
        timeout=120,
    )
    finished_at = utc_now()
    latency_ms = (time.perf_counter() - start) * 1000
    choice = response.get("choices", [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        raise RuntimeError("live smoke response did not include tool_calls")

    record = {
        "schema": "model-call-record.v1",
        "schema_version": "v1",
        "schema_bundle_hash": MODEL_FUSION_SCHEMA_BUNDLE_HASH,
        "producer": "mlx-lm-live-smoke",
        "producer_version": "local",
        "producer_git_sha": producer_git_sha(),
        "created_at": finished_at,
        "call_id": call_id,
        "endpoint_id": capabilities["endpoint"]["endpoint_id"],
        "provider_request_id": response.get("id"),
        "model": model,
        "request_hash": contract_sha256(payload),
        "response_hash": contract_sha256(response),
        "messages": messages,
        "status": "succeeded",
        "side_effects": "network",
        "started_at": started_at,
        "finished_at": finished_at,
        "latency_ms": latency_ms,
        "usage": response.get("usage"),
        "output_text": json.dumps(tool_calls, sort_keys=True),
        "metadata": compact_dict(
            {
                "platform": platform_metadata(),
                "tool_call_count": len(tool_calls),
                "valid_tool_call_count": count_valid_tool_calls(tool_calls),
                "finish_reason": choice.get("finish_reason"),
                "base_url": f"http://{host}:{port}",
                "response_model": response.get("model"),
            }
        ),
    }
    return validate_model_call_record_fixture(record)


def platform_metadata() -> Dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }


def count_valid_tool_calls(tool_calls: Any) -> int:
    if not isinstance(tool_calls, list):
        return 0
    return sum(1 for tool_call in tool_calls if is_valid_tool_call(tool_call))


def is_valid_tool_call(tool_call: Any) -> bool:
    if not isinstance(tool_call, dict):
        return False
    function = tool_call.get("function")
    if not isinstance(tool_call.get("id"), str) or not isinstance(function, dict):
        return False
    if not isinstance(function.get("name"), str):
        return False
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        return False
    try:
        parsed_arguments = json.loads(arguments)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed_arguments, dict)


def write_and_validate_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_model_call_record_fixture(record)
    path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise RuntimeError(f"expected exactly one JSONL line in {path}")
    validate_model_call_record_fixture(json.loads(lines[0]))


if __name__ == "__main__":
    raise SystemExit(main())
