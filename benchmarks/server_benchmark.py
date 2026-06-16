"""
Spin up the local server:

    mlx_lm.server

Then run the benchmark:

    python server_benchmark.py --concurrency 4
"""

import argparse
import asyncio
import datetime
import hashlib
import json
import math
import platform
import subprocess
import time
import uuid
from itertools import cycle
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from mlx_lm.openai_compat import (
    MODEL_FUSION_SCHEMA_BUNDLE_HASH,
    validate_model_call_record_fixture,
)
from tqdm import tqdm

try:
    import mlx.core as mx
except ImportError:
    mx = None

# Default prompts if no file is provided
DEFAULT_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What are the main differences between Python and JavaScript?",
    "Describe the process of photosynthesis in plants.",
    "How does a neural network learn from data?",
    "What is the significance of the Turing test in AI?",
    "Explain the concept of blockchain technology.",
    "What causes seasons on Earth?",
    "How do vaccines work in the human body?",
    "Describe the water cycle and its importance.",
    "What is the theory of relativity proposed by Einstein?",
    "How do electric cars help reduce carbon emissions?",
    "What are the key features of a market economy?",
    "Explain how DNA replication works in cells.",
    "What is machine learning and its real-world applications?",
    "Describe the structure and function of the human heart.",
]


def contract_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def compact_dict(value: Dict[str, Any]) -> Dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def platform_metadata() -> Dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }


def memory_peak_gb() -> Optional[float]:
    if mx is None:
        return None
    try:
        return mx.get_peak_memory() / 1e9
    except Exception:
        return None


def producer_git_sha() -> str:
    try:
        repo_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
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


def normalized_error(
    kind: str, message: str, *, retryable: Optional[bool] = None
) -> Dict[str, Any]:
    return compact_dict(
        {
            "kind": kind,
            "message": message[:1000],
            "retryable": retryable,
        }
    )


def response_error_message(status: int, body: str) -> str:
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:
        return body[:1000] or f"HTTP {status}"
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"][:1000]
    if isinstance(parsed, dict) and isinstance(parsed.get("message"), str):
        return parsed["message"][:1000]
    return f"HTTP {status}"


def make_model_call_record_fixture(
    *,
    call_id: str,
    endpoint_id: str,
    model: str,
    messages: List[Dict[str, str]],
    status: str,
    started_at: str,
    request_payload: Optional[Dict[str, Any]] = None,
    provider_request_id: Optional[str] = None,
    response_payload: Optional[Dict[str, Any]] = None,
    side_effects: str = "network",
    finished_at: Optional[str] = None,
    latency_ms: Optional[float] = None,
    usage: Optional[Dict[str, int]] = None,
    output_text: Optional[str] = None,
    error: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    created_at: Optional[str] = None,
    producer_git_sha: str = "0" * 40,
) -> Dict[str, Any]:
    """Build and validate a provider call fixture without writing benchmark JSONL."""
    timestamp = created_at or utc_now()
    hash_payload = (
        request_payload if request_payload is not None else {"messages": messages}
    )
    record: Dict[str, Any] = {
        "schema": "model-call-record.v1",
        "schema_version": "v1",
        "schema_bundle_hash": MODEL_FUSION_SCHEMA_BUNDLE_HASH,
        "producer": "mlx-lm-benchmark",
        "producer_version": "local",
        "producer_git_sha": producer_git_sha,
        "created_at": timestamp,
        "call_id": call_id,
        "endpoint_id": endpoint_id,
        "model": model,
        "request_hash": contract_sha256(hash_payload),
        "messages": messages,
        "status": status,
        "side_effects": side_effects,
        "started_at": started_at,
    }
    if provider_request_id is not None:
        record["provider_request_id"] = provider_request_id
    if response_payload is not None:
        record["response_hash"] = contract_sha256(response_payload)
    if finished_at is not None:
        record["finished_at"] = finished_at
    if latency_ms is not None:
        record["latency_ms"] = latency_ms
    if usage is not None:
        record["usage"] = usage
    if output_text is not None:
        record["output_text"] = output_text
    if error is not None:
        record["error"] = error
    if metadata is not None:
        record["metadata"] = metadata

    validate_model_call_record_fixture(record)
    return record


def make_model_call_record_from_benchmark_result(
    *,
    result: Dict[str, Any],
    endpoint_id: str,
    model: str,
    url: str,
    max_tokens: int,
    concurrency: int,
    total_requests: int,
    platform_info: Dict[str, Any],
    benchmark_task_id: str,
    schema_valid_rate: float,
    tool_call_valid_rate: Optional[float],
    producer_git_sha: str = "0" * 40,
) -> Dict[str, Any]:
    metadata = {
        "benchmark": {
            "task_id": benchmark_task_id,
            "url": url,
            "request_index": result["request_index"],
            "concurrency": concurrency,
            "total_requests": total_requests,
            "max_tokens": max_tokens,
            "stream": True,
            "schema_valid_rate": schema_valid_rate,
            "tool_call_valid_rate": tool_call_valid_rate,
        },
        "platform": platform_info,
    }
    metadata.update(result.get("metadata", {}))

    usage = result.get("usage")
    if usage is None and result.get("success"):
        usage = {"completion_tokens": len(result.get("tokens", []))}

    return make_model_call_record_fixture(
        call_id=result["call_id"],
        endpoint_id=endpoint_id,
        model=model,
        messages=result["messages"],
        status="succeeded" if result.get("success") else "failed",
        started_at=result["started_at"],
        request_payload=result["request_payload"],
        provider_request_id=result.get("provider_request_id"),
        response_payload=result.get("response_payload"),
        finished_at=result.get("finished_at"),
        latency_ms=result.get("latency_ms"),
        usage=usage,
        output_text=result.get("output_text"),
        error=result.get("error"),
        metadata=metadata,
        created_at=result.get("finished_at"),
        producer_git_sha=producer_git_sha,
    )


def write_model_call_records_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            validate_model_call_record_fixture(record)
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def is_valid_stream_tool_call(tool_call: Any) -> bool:
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


def tokens_per_second(tokens):
    start = math.floor(tokens[0])
    stop = math.ceil(tokens[-1])
    if stop <= start:
        return [tokens[0]], [0]
    n_bins = int(stop - start) * 10
    bins = [0] * n_bins
    for t in tokens:
        bins[int(n_bins * (t - start) / (stop - start))] += 1

    result = []

    ms = 0
    cnt = 0
    for i, b in enumerate(bins):
        ms += b
        if cnt == 10:
            ms -= bins[i - 10]
        else:
            cnt += 1

        result.append(10 * ms / cnt)

    times = [start]
    while times[-1] < stop:
        times.append(times[-1] + 0.1)

    return times, result


def plot_generation(times, tokens_per_sec, start=None, interval=1.0, width=50):
    c = "█"
    start = start or times[0]
    stop = times[-1]

    bar_times = [start]
    while bar_times[-1] < stop:
        bar_times.append(bar_times[-1] + interval)

    bar_values = [[] for _ in bar_times]
    bar_idx = 0

    for t, v in zip(times, tokens_per_sec):
        while t > bar_times[bar_idx] + interval:
            bar_idx += 1
        bar_values[bar_idx].append(v)

    bar_values = [sum(v) / len(v) if v else 0 for v in bar_values]
    m = max(bar_values)
    if m <= 0:
        print("Not enough token timing data to plot generation throughput.")
        return

    for t, v in zip(bar_times, bar_values):
        t = t - start
        b = c * int(v * width / m)
        print(f"{t:3.2f} {b} ({v})")


def percentile(data, percent):
    if not data:
        return 0
    data = sorted(data)
    k = (len(data) - 1) * percent / 100
    f = math.floor(k)
    c = math.ceil(k)
    return (
        data[int(f)]
        if f == c
        else data[int(f)] + (data[int(c)] - data[int(f)]) * (k - f)
    )


def median(data):
    return percentile(data, 50)


async def make_request(
    session: aiohttp.ClientSession,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_index: int,
) -> Dict[str, Any]:
    """
    Make a single streaming API request and return

        - whether the request succeeded
        - the request start time
        - the time of every generated token
    """
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    call_id = f"call_bench_{request_index:06d}_{uuid.uuid4().hex[:12]}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "x-velum-model-call-id": call_id,
    }

    start_time = time.perf_counter()
    started_at = utc_now()
    tokens = []
    tool_call_count = 0
    valid_tool_call_count = 0

    try:
        async with session.post(url, json=payload, headers=headers) as response:
            provider_request_id = None
            output_parts = []
            finish_reasons = []
            usage = None
            response_events = 0
            model_call_id_echo = response.headers.get("x-velum-model-call-id")

            if response.status != 200:
                error_body = await response.text()
                print(f"Error {response.status}: {error_body}")
                finished_at = utc_now()
                latency_ms = (time.perf_counter() - start_time) * 1000
                return {
                    "request_index": request_index,
                    "call_id": call_id,
                    "messages": messages,
                    "request_payload": payload,
                    "success": False,
                    "start_time": start_time,
                    "tokens": [],
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "latency_ms": latency_ms,
                    "response_payload": {
                        "http_status": response.status,
                        "body": error_body,
                    },
                    "error": normalized_error(
                        "rate_limited" if response.status == 429 else "provider_error",
                        response_error_message(response.status, error_body),
                        retryable=response.status == 429 or response.status >= 500,
                    ),
                    "metadata": compact_dict(
                        {
                            "http_status": response.status,
                            "model_call_id_echo": model_call_id_echo,
                            "memory_peak_gb": memory_peak_gb(),
                        }
                    ),
                }

            # Process streaming response
            async for chunk in response.content:
                if chunk:
                    chunk_str = chunk.decode("utf-8").strip()
                    for line in chunk_str.splitlines():
                        line = line.strip()
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            response_events += 1
                            if isinstance(data.get("id"), str):
                                provider_request_id = data["id"]
                            if isinstance(data.get("usage"), dict):
                                usage = data["usage"]
                            if choices := data.get("choices", False):
                                choice = choices[0]
                                delta = choice.get("delta") or {}
                                content = delta.get("content")
                                if isinstance(content, str):
                                    output_parts.append(content)
                                tool_calls = delta.get("tool_calls")
                                if isinstance(tool_calls, list):
                                    for tool_call in tool_calls:
                                        tool_call_count += 1
                                        if is_valid_stream_tool_call(tool_call):
                                            valid_tool_call_count += 1
                                finish_reason = choice.get("finish_reason")
                                if finish_reason is not None:
                                    finish_reasons.append(finish_reason)
                                if finish_reason != "length":
                                    tokens.append(time.perf_counter())
                        except json.JSONDecodeError:
                            continue

            finished_at = utc_now()
            latency_ms = (time.perf_counter() - start_time) * 1000
            output_text = "".join(output_parts)
            response_payload = compact_dict(
                {
                    "http_status": response.status,
                    "provider_request_id": provider_request_id,
                    "output_text": output_text,
                    "finish_reasons": finish_reasons,
                    "usage": usage,
                }
            )
            success = bool(tokens)
            return {
                "request_index": request_index,
                "call_id": call_id,
                "messages": messages,
                "request_payload": payload,
                "success": success,
                "start_time": start_time,
                "tokens": tokens,
                "started_at": started_at,
                "finished_at": finished_at,
                "latency_ms": latency_ms,
                "provider_request_id": provider_request_id,
                "response_payload": response_payload,
                "usage": usage,
                "output_text": output_text or None,
                "error": None
                if success
                else normalized_error(
                    "provider_error",
                    "Streaming response did not include generated tokens",
                    retryable=False,
                ),
                "metadata": compact_dict(
                    {
                        "http_status": response.status,
                        "model_call_id_echo": model_call_id_echo,
                        "response_events": response_events,
                        "finish_reasons": finish_reasons,
                        "tool_call_count": tool_call_count,
                        "valid_tool_call_count": valid_tool_call_count,
                        "memory_peak_gb": memory_peak_gb(),
                    }
                ),
            }

    except Exception as e:
        print(f"Request failed: {str(e)}")
        finished_at = utc_now()
        latency_ms = (time.perf_counter() - start_time) * 1000
        kind = "timeout" if isinstance(e, asyncio.TimeoutError) else "provider_error"
        return {
            "request_index": request_index,
            "call_id": call_id,
            "messages": messages,
            "request_payload": payload,
            "success": False,
            "start_time": start_time,
            "tokens": [],
            "started_at": started_at,
            "finished_at": finished_at,
            "latency_ms": latency_ms,
            "response_payload": {
                "exception_type": type(e).__name__,
                "message": str(e),
            },
            "error": normalized_error(kind, str(e), retryable=kind == "timeout"),
            "metadata": compact_dict(
                {
                    "exception_type": type(e).__name__,
                    "memory_peak_gb": memory_peak_gb(),
                }
            ),
        }


async def run_benchmark(
    url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    concurrency: int,
    total_requests: int,
    prompts: List[str],
    endpoint_id: str = "mlx-local-benchmark",
    jsonl_output: Optional[str] = None,
    benchmark_task_id: Optional[str] = None,
) -> Dict[str, Any]:
    prompt_cycle = cycle(prompts)
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    request_times = []
    bar = tqdm(total=total_requests)

    async def worker(request_index: int):
        async with semaphore:
            prompt = next(prompt_cycle)
            result = await make_request(
                session, url, api_key, model, prompt, max_tokens, request_index
            )
            bar.update(1)
            return result

    async with aiohttp.ClientSession() as session:
        tasks = []
        for request_index in range(1, total_requests + 1):
            task = asyncio.create_task(worker(request_index))
            tasks.append(task)
            await asyncio.sleep(0.01)  # Stagger requests slightly

        for task in tasks:
            result = await task
            results.append(result)
        bar.close()

    benchmark_task_id = benchmark_task_id or f"bench_{utc_now().replace(':', '-')}"
    tool_call_count = sum(
        result.get("metadata", {}).get("tool_call_count", 0) for result in results
    )
    valid_tool_call_count = sum(
        result.get("metadata", {}).get("valid_tool_call_count", 0)
        for result in results
    )
    schema_valid_rate = 1.0 if results else 0.0
    tool_call_valid_rate = (
        valid_tool_call_count / tool_call_count if tool_call_count else None
    )

    if jsonl_output:
        current_platform_metadata = platform_metadata()
        current_producer_git_sha = producer_git_sha()
        records = [
            make_model_call_record_from_benchmark_result(
                result=result,
                endpoint_id=endpoint_id,
                model=model,
                url=url,
                max_tokens=max_tokens,
                concurrency=concurrency,
                total_requests=total_requests,
                platform_info=current_platform_metadata,
                benchmark_task_id=benchmark_task_id,
                schema_valid_rate=schema_valid_rate,
                tool_call_valid_rate=tool_call_valid_rate,
                producer_git_sha=current_producer_git_sha,
            )
            for result in results
        ]
        write_model_call_records_jsonl(jsonl_output, records)

    successful_requests = [r for r in results if r["success"]]
    total_tokens = sum(len(r["tokens"]) for r in successful_requests)

    # Gather all the tokens generated with their corresponding timestamps
    all_tokens = []
    for r in successful_requests:
        all_tokens.extend(r["tokens"])
    all_tokens.sort()
    full_generation = tokens_per_second(all_tokens) if all_tokens else ([], [])
    start = min((r["start_time"] for r in successful_requests), default=0)
    total_time = all_tokens[-1] - start if all_tokens else 0

    # Aggregate metrics
    metrics = {
        "total_requests": total_requests,
        "successful_requests": len(successful_requests),
        "failed_requests": total_requests - len(successful_requests),
        "total_tokens": total_tokens,
        "total_time": total_time,
        "aggregate_tokens_per_sec": median(full_generation[1])
        if full_generation[1]
        else 0,
        "per_request": [],
        "start": start,
        "full_generation": full_generation,
        "benchmark_task_id": benchmark_task_id,
        "schema_valid_rate": schema_valid_rate,
        "tool_call_valid_rate": tool_call_valid_rate,
    }
    if jsonl_output:
        metrics["model_call_record_jsonl"] = jsonl_output

    # Per-request metrics
    for i, result in enumerate(successful_requests):
        request_start = result["start_time"]
        tokens = result["tokens"]
        metrics["per_request"].append(
            {
                "request_id": i + 1,
                "time_to_first_token": tokens[0] - request_start,
                "total_time": tokens[-1] - request_start,
                "tokens_received": len(tokens),
                "tokens_per_sec": median(tokens_per_second(tokens)[1]),
            }
        )

    # Calculate percentiles
    ttft_values = [m["time_to_first_token"] for m in metrics["per_request"]]
    tps_values = [m["tokens_per_sec"] for m in metrics["per_request"]]

    metrics["aggregate_metrics"] = {
        "time_to_first_token": {
            "min": min(ttft_values) if ttft_values else 0,
            "max": max(ttft_values) if ttft_values else 0,
            "avg": sum(ttft_values) / len(ttft_values) if ttft_values else 0,
            "p95": percentile(ttft_values, 95) if ttft_values else 0,
        },
        "tokens_per_sec": {
            "min": min(tps_values) if tps_values else 0,
            "max": max(tps_values) if tps_values else 0,
            "avg": sum(tps_values) / len(tps_values) if tps_values else 0,
            "p95": percentile(tps_values, 95) if tps_values else 0,
        },
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description="LLM API Benchmark Tool")
    parser.add_argument(
        "--url",
        default="http://localhost:8080/v1/chat/completions",
        help="Chat completions API endpoint URL",
    )
    parser.add_argument("--api-key", default="none", help="API key")
    parser.add_argument("--model", default="default_model", help="Model name")
    parser.add_argument(
        "--max-tokens", type=int, default=100, help="Max tokens to generate"
    )
    parser.add_argument(
        "--concurrency", type=int, default=1, help="Number of concurrent requests"
    )
    parser.add_argument(
        "--total-requests", type=int, default=10, help="Total requests to make"
    )
    parser.add_argument("--prompt-file", help="File containing prompts (one per line)")
    parser.add_argument("--output", help="Output file for results (JSON format)")
    parser.add_argument(
        "--jsonl-output",
        help="Output file for per-request model-call-record.v1 JSONL",
    )
    parser.add_argument(
        "--endpoint-id",
        default="mlx-local-benchmark",
        help="Endpoint ID to include in model-call-record.v1 JSONL",
    )
    parser.add_argument(
        "--benchmark-task-id",
        help="Benchmark task ID to include in model-call-record.v1 metadata",
    )

    args = parser.parse_args()

    # Load prompts
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    print(
        f"Starting benchmark with {args.concurrency} concurrency and {args.total_requests} total requests..."
    )
    start_time = time.perf_counter()

    # Run benchmark
    results = asyncio.run(
        run_benchmark(
            url=args.url,
            api_key=args.api_key,
            model=args.model,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            total_requests=args.total_requests,
            prompts=prompts,
            endpoint_id=args.endpoint_id,
            jsonl_output=args.jsonl_output,
            benchmark_task_id=args.benchmark_task_id,
        )
    )

    duration = time.perf_counter() - start_time
    print(f"\nBenchmark completed in {duration:.2f} seconds")
    print(
        f"Successful requests: {results['successful_requests']}/{args.total_requests}"
    )
    print(f"Total tokens generated: {results['total_tokens']}")
    print(f"Aggregate tokens/sec: {results['aggregate_tokens_per_sec']:.2f}")

    # Print summary
    if results["successful_requests"] > 0:
        ttft = results["aggregate_metrics"]["time_to_first_token"]
        tps = results["aggregate_metrics"]["tokens_per_sec"]

        print("\nTime to First Token (seconds):")
        print(
            f"  Min: {ttft['min']:.4f} | Max: {ttft['max']:.4f} | Avg: {ttft['avg']:.4f} | P95: {ttft['p95']:.4f}"
        )

        print("\nTokens per Second (per request):")
        print(
            f"  Min: {tps['min']:.2f} | Max: {tps['max']:.2f} | Avg: {tps['avg']:.2f} | P95: {tps['p95']:.2f}"
        )

        print()
        plot_generation(*results["full_generation"], results["start"])

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    if args.jsonl_output:
        print(f"Model call records saved to {args.jsonl_output}")


if __name__ == "__main__":
    main()
