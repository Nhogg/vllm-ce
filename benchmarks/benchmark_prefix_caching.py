# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark the efficiency of prefix caching.

This script allows you to benchmark the performance of
a model with and without prefix caching using either fixed prompts
or prompts sampled from the ShareGPT dataset.

Fixed example usage:
    python benchmark_prefix_caching.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --enable-prefix-caching \
        --num-prompts 1 \
        --repeat-count 100 \
        --input-length-range 128:256

ShareGPT example usage:
    # This command samples 20 prompts with input lengths
    # between 128 and 256 tokens from the ShareGPT dataset,
    # then replicates each prompt 5 times.
    python benchmark_prefix_caching.py \
        --model meta-llama/Llama-2-7b-chat-hf \
        --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
        --enable-prefix-caching \
        --num-prompts 20 \
        --repeat-count 5 \
        --input-length-range 128:256
"""

import dataclasses
import json
import os
import random
import shutil
import subprocess
import threading
import time
from statistics import mean, quantiles

from transformers import PreTrainedTokenizerBase

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

try:
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

try:
    import psutil
except ImportError:
    psutil = None

PROMPT = "You are a helpful assistant in recognizes the content of tables in markdown format. Here is a table as fellows. You need to answer my question about the table.\n# Table\n|Opening|Opening|Sl. No.|Film|Cast|Director|Music Director|Notes|\n|----|----|----|----|----|----|----|----|\n|J A N|9|1|Agni Pushpam|Jayabharathi, Kamalahasan|Jeassy|M. K. Arjunan||\n|J A N|16|2|Priyamvada|Mohan Sharma, Lakshmi, KPAC Lalitha|K. S. Sethumadhavan|V. Dakshinamoorthy||\n|J A N|23|3|Yakshagaanam|Madhu, Sheela|Sheela|M. S. Viswanathan||\n|J A N|30|4|Paalkkadal|Sheela, Sharada|T. K. Prasad|A. T. Ummer||\n|F E B|5|5|Amma|Madhu, Srividya|M. Krishnan Nair|M. K. Arjunan||\n|F E B|13|6|Appooppan|Thikkurissi Sukumaran Nair, Kamal Haasan|P. Bhaskaran|M. S. Baburaj||\n|F E B|20|7|Srishti|Chowalloor Krishnankutty, Ravi Alummoodu|K. T. Muhammad|M. S. Baburaj||\n|F E B|20|8|Vanadevatha|Prem Nazir, Madhubala|Yusufali Kechery|G. Devarajan||\n|F E B|27|9|Samasya|Madhu, Kamalahaasan|K. Thankappan|Shyam||\n|F E B|27|10|Yudhabhoomi|K. P. Ummer, Vidhubala|Crossbelt Mani|R. K. Shekhar||\n|M A R|5|11|Seemantha Puthran|Prem Nazir, Jayabharathi|A. B. Raj|M. K. Arjunan||\n|M A R|12|12|Swapnadanam|Rani Chandra, Dr. Mohandas|K. G. George|Bhaskar Chandavarkar||\n|M A R|19|13|Thulavarsham|Prem Nazir, sreedevi, Sudheer|N. Sankaran Nair|V. Dakshinamoorthy||\n|M A R|20|14|Aruthu|Kaviyoor Ponnamma, Kamalahasan|Ravi|G. Devarajan||\n|M A R|26|15|Swimming Pool|Kamal Haasan, M. G. Soman|J. Sasikumar|M. K. Arjunan||\n\n# Question\nWhat' s the content in the (1,1) cells\n"  # noqa: E501


def test_prefix(llm=None, sampling_params=None, prompts=None):
    start_time = time.time()

    llm.generate(prompts, sampling_params=sampling_params)

    end_time = time.time()
    print(f"cost time {end_time - start_time}")


@dataclasses.dataclass
class Request:
    prompt: str
    prompt_len: int
    output_len: int


def sample_tokens(tokenizer: PreTrainedTokenizerBase, length: int) -> list[int]:
    vocab = tokenizer.get_vocab()
    all_special_ids = set(tokenizer.all_special_ids)
    token_ids = sorted(v for v in vocab.values() if v not in all_special_ids)

    # Remove the special tokens.
    return random.choices(token_ids, k=length)


def sample_requests_from_dataset(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: tuple[int, int],
    fixed_output_len: int | None,
) -> list[Request]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]

    # Shuffle the dataset.
    random.shuffle(dataset)

    min_len, max_len = input_length_range
    assert min_len >= 0 and max_len >= min_len, "input_length_range too small"

    # Filter out sequences that are too long or too short
    filtered_requests: list[Request] = []

    for i in range(len(dataset)):
        if len(filtered_requests) == num_requests:
            break

        # Tokenize the prompts and completions.
        prompt_token_ids = tokenizer(dataset[i][0]).input_ids
        prompt = tokenizer.decode(prompt_token_ids)
        completion = dataset[i][1]
        completion_token_ids = tokenizer(completion).input_ids
        prompt_len = len(prompt_token_ids)
        output_len = (
            len(completion_token_ids) if fixed_output_len is None else fixed_output_len
        )
        if min_len <= prompt_len <= max_len:
            filtered_requests.append(Request(prompt, prompt_len, output_len))

    return filtered_requests


def sample_requests_from_random(
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    input_length_range: tuple[int, int],
    fixed_output_len: int | None,
    prefix_len: int,
) -> list[Request]:
    requests = []
    prefix_token_ids = sample_tokens(tokenizer, prefix_len)
    min_len, max_len = input_length_range

    for i in range(num_requests):
        unique_part_token_ids = sample_tokens(
            tokenizer, random.randint(min_len - prefix_len, max_len - prefix_len)
        )
        prompt_token_ids = prefix_token_ids + unique_part_token_ids
        prompt = tokenizer.decode(prompt_token_ids)
        prompt_len = len(prompt_token_ids)
        assert min_len <= prompt_len <= max_len, (
            f"prompt_len {prompt_len} out of range {min_len}:{max_len}"
        )
        requests.append(Request(prompt, prompt_len, fixed_output_len))
    return requests


def repeat_and_sort_requests(
    requests: list[Request], repeat_count: int, sort: bool = False
) -> list[str]:
    repeated_requests = requests * repeat_count
    if sort:
        repeated_requests.sort(key=lambda x: x[1])
    else:
        random.shuffle(repeated_requests)
    return [req.prompt for req in repeated_requests]


def collect_cache_hit_rate(llm) -> float | None:
    """Extract prefix cache hit rate from the engine's stats, if available."""
    try:
        engine = llm.llm_engine
        if hasattr(engine, "stat_loggers"):
            for logger in engine.stat_loggers.values():
                stats = getattr(logger, "_last_stats", None)
                if stats and hasattr(stats, "cache_config"):
                    return None
        return None
    except Exception:
        return None


def collect_gpu_kv_cache_util(llm) -> float | None:
    def collect_from_logger(logger) -> list[float]:
        stats = getattr(logger, "last_scheduler_stats", None)
        if stats is not None:
            return [stats.kv_cache_usage]

        per_engine_loggers = getattr(logger, "per_engine_stat_loggers", None)
        if per_engine_loggers is None:
            return []

        values = []
        for per_engine_logger in per_engine_loggers.values():
            values.extend(collect_from_logger(per_engine_logger))
        return values

    try:
        logger_manager = getattr(llm.llm_engine, "logger_manager", None)
        if logger_manager is None:
            return None

        values = []
        for logger in getattr(logger_manager, "stat_loggers", []):
            values.extend(collect_from_logger(logger))
        if values:
            return sum(values) / len(values)
    except Exception:
        return None
    return None


def run_requests_individually(llm, prompts, sampling_params):
    """Return per-request timing and vLLM request metrics."""
    results = []
    for prompt in prompts:
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], sampling_params=sampling_params)
        t1 = time.perf_counter()

        elapsed_ms = (t1 - t0) * 1000
        output = outputs[0]
        num_output = len(output.outputs[0].token_ids)

        metrics = output.metrics
        ttft_ms = None
        itl_ms = None
        tpot_ms = None

        if metrics is not None:
            ttft_ms = metrics.first_token_latency * 1000
            if metrics.num_generation_tokens > 1:
                decode_time_s = metrics.last_token_ts - metrics.first_token_ts
                itl_ms = decode_time_s * 1000 / (metrics.num_generation_tokens - 1)
                tpot_ms = itl_ms

        results.append(
            {
                "e2e_ms": elapsed_ms,
                "output_tokens": num_output,
                "cached_tokens": output.num_cached_tokens or 0,
                "ttft_ms": ttft_ms,
                "itl_ms": itl_ms,
                "tpot_ms": tpot_ms,
            }
        )
    return results


def compute_stats(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {}
    if len(latencies_ms) == 1:
        value = latencies_ms[0]
        return {
            "mean_ms": value,
            "p50_ms": value,
            "p95_ms": value,
            "p99_ms": value,
            "min_ms": value,
            "max_ms": value,
            "n": 1,
        }
    qs = quantiles(latencies_ms, n=100)
    return {
        "mean_ms": mean(latencies_ms),
        "p50_ms": qs[49],
        "p95_ms": qs[94],
        "p99_ms": qs[98],
        "min_ms": min(latencies_ms),
        "max_ms": max(latencies_ms),
        "n": len(latencies_ms),
    }


def compute_optional_stats(values: list[float | None]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {}
    return compute_stats(clean)


def compute_int_stats(values: list[int]) -> dict:
    if not values:
        return {}
    qs = quantiles(values, n=100)
    return {
        "mean": mean(values),
        "p50": qs[49],
        "p95": qs[94],
        "p99": qs[98],
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _max_or_none(values: list[float]) -> float | None:
    return max(values) if values else None


def summarize_resource_samples(samples: list[dict]) -> dict:
    if not samples:
        return {}

    cpu_percent = [
        sample["cpu_percent"]
        for sample in samples
        if sample.get("cpu_percent") is not None
    ]
    process_rss_mb = [
        sample["process_rss_mb"]
        for sample in samples
        if sample.get("process_rss_mb") is not None
    ]
    system_mem_percent = [
        sample["system_mem_percent"]
        for sample in samples
        if sample.get("system_mem_percent") is not None
    ]
    gpu_util_percent = [
        gpu["utilization_gpu_percent"]
        for sample in samples
        for gpu in sample.get("gpus", [])
        if gpu.get("utilization_gpu_percent") is not None
    ]
    gpu_mem_used_mb = [
        gpu["memory_used_mb"]
        for sample in samples
        for gpu in sample.get("gpus", [])
        if gpu.get("memory_used_mb") is not None
    ]

    return {
        "num_samples": len(samples),
        "cpu_percent_mean": _mean_or_none(cpu_percent),
        "cpu_percent_max": _max_or_none(cpu_percent),
        "process_rss_mb_mean": _mean_or_none(process_rss_mb),
        "process_rss_mb_max": _max_or_none(process_rss_mb),
        "system_mem_percent_mean": _mean_or_none(system_mem_percent),
        "system_mem_percent_max": _max_or_none(system_mem_percent),
        "gpu_util_percent_mean": _mean_or_none(gpu_util_percent),
        "gpu_util_percent_max": _max_or_none(gpu_util_percent),
        "gpu_mem_used_mb_mean": _mean_or_none(gpu_mem_used_mb),
        "gpu_mem_used_mb_max": _max_or_none(gpu_mem_used_mb),
    }


class ResourceMonitor:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.samples: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._process = psutil.Process(os.getpid()) if psutil is not None else None
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        self._visible_devices = {
            device.strip() for device in visible_devices.split(",") if device.strip()
        }

    def start(self) -> None:
        if self.interval_s <= 0:
            return
        if self._process is not None:
            self._process.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_s * 2, 1.0))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.samples.append(self._collect_sample())
            self._stop_event.wait(self.interval_s)

    def _collect_sample(self) -> dict:
        sample = {
            "timestamp_s": time.time(),
            "cpu_percent": None,
            "process_rss_mb": None,
            "process_vms_mb": None,
            "system_mem_percent": None,
            "system_mem_used_mb": None,
            "system_mem_total_mb": None,
            "gpus": [],
        }

        if psutil is not None and self._process is not None:
            try:
                processes = [self._process] + self._process.children(recursive=True)
                cpu_percent = 0.0
                rss_bytes = 0
                vms_bytes = 0
                for process in processes:
                    try:
                        cpu_percent += process.cpu_percent(interval=None)
                        memory = process.memory_info()
                        rss_bytes += memory.rss
                        vms_bytes += memory.vms
                    except psutil.Error:
                        continue

                system_mem = psutil.virtual_memory()
                sample.update(
                    {
                        "cpu_percent": cpu_percent,
                        "process_rss_mb": rss_bytes / (1024**2),
                        "process_vms_mb": vms_bytes / (1024**2),
                        "system_mem_percent": system_mem.percent,
                        "system_mem_used_mb": system_mem.used / (1024**2),
                        "system_mem_total_mb": system_mem.total / (1024**2),
                    }
                )
            except psutil.Error:
                pass

        if self._has_nvidia_smi:
            sample["gpus"] = self._collect_gpu_samples()

        return sample

    def _collect_gpu_samples(self) -> list[dict]:
        query = (
            "index,uuid,utilization.gpu,utilization.memory,"
            "memory.used,memory.total,power.draw"
        )
        cmd = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        try:
            output = subprocess.check_output(
                cmd,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return []

        gpus = []
        for line in output.strip().splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 7:
                continue
            if self._visible_devices and (
                fields[0] not in self._visible_devices
                and fields[1] not in self._visible_devices
            ):
                continue
            gpus.append(
                {
                    "index": _parse_int(fields[0]),
                    "uuid": fields[1],
                    "utilization_gpu_percent": _parse_float(fields[2]),
                    "utilization_memory_percent": _parse_float(fields[3]),
                    "memory_used_mb": _parse_float(fields[4]),
                    "memory_total_mb": _parse_float(fields[5]),
                    "power_draw_w": _parse_float(fields[6]),
                }
            )
        return gpus


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def main(args):
    t_script_start = time.perf_counter()
    phase_timings_s: dict[str, float] = {}

    t0 = time.perf_counter()
    tokenizer = get_tokenizer(args.model, trust_remote_code=True)
    input_length_range = tuple(map(int, args.input_length_range.split(":")))
    random.seed(args.seed)
    if args.dataset_path is not None:
        if args.prefix_len > 0:
            raise ValueError(
                "prefix-len is not supported when dataset-path is provided."
            )
        print(f"Start to sample {args.num_prompts} prompts from {args.dataset_path}")
        filtered_requests = sample_requests_from_dataset(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            input_length_range=input_length_range,
            fixed_output_len=args.output_len,
        )
    else:
        print(f"Start to sample {args.num_prompts} prompts from random")
        filtered_requests = sample_requests_from_random(
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
            input_length_range=input_length_range,
            fixed_output_len=args.output_len,
            prefix_len=args.prefix_len,
        )
    phase_timings_s["tokenizer_and_sampling_s"] = time.perf_counter() - t0

    prompt_lens = [req.prompt_len for req in filtered_requests]
    print(f"Sampled {len(filtered_requests)} requests.")
    print(f"  avg input len: {sum(prompt_lens) / len(prompt_lens):.0f}")
    print(f"  P50 input len: {sorted(prompt_lens)[len(prompt_lens) // 2]}")

    engine_args = EngineArgs.from_cli_args(args)
    t0 = time.perf_counter()
    llm = LLM.from_engine_args(engine_args)
    phase_timings_s["engine_init_s"] = time.perf_counter() - t0

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.output_len,
        ignore_eos=not args.respect_eos,
        detokenize=not args.disable_detokenize,
    )

    resource_monitor = ResourceMonitor(args.resource_sample_interval_s)
    resource_monitor.start()

    # Warm-up: send all prompts once to fill the cache (not measured).
    if args.warmup_rounds > 0:
        print(f"Warming up ({args.warmup_rounds} round(s))...")
        warmup_prompts = repeat_and_sort_requests(
            filtered_requests, repeat_count=args.warmup_rounds, sort=args.sort
        )
        t0 = time.perf_counter()
        llm.generate(warmup_prompts, sampling_params=sampling_params)
        phase_timings_s["warmup_s"] = time.perf_counter() - t0
        print("Warm-up done.")
    else:
        phase_timings_s["warmup_s"] = 0.0

    # Measurement phase: send one request at a time to get per-request timing.
    measure_prompts = repeat_and_sort_requests(
        filtered_requests, repeat_count=args.repeat_count, sort=args.sort
    )

    print(f"Measuring {len(measure_prompts)} requests one-by-one...")
    t_batch_start = time.perf_counter()
    per_request = run_requests_individually(llm, measure_prompts, sampling_params)
    t_batch_end = time.perf_counter()
    resource_monitor.stop()
    phase_timings_s["measurement_s"] = t_batch_end - t_batch_start
    phase_timings_s["script_total_s"] = time.perf_counter() - t_script_start

    latencies_ms = [r["e2e_ms"] for r in per_request]
    output_tokens_per_request = [r["output_tokens"] for r in per_request]
    cached_tokens_per_request = [r["cached_tokens"] for r in per_request]
    ttft_ms = [r["ttft_ms"] for r in per_request]
    itl_ms = [r["itl_ms"] for r in per_request]
    tpot_ms = [r["tpot_ms"] for r in per_request]
    total_input_tokens = sum(len(tokenizer(p).input_ids) for p in measure_prompts)
    total_output_tokens = sum(output_tokens_per_request)
    total_cached_tokens = sum(cached_tokens_per_request)
    wall_time_s = t_batch_end - t_batch_start
    throughput_tok_s = (total_input_tokens + total_output_tokens) / wall_time_s

    stats = compute_stats(latencies_ms)

    ttft_stats = compute_optional_stats(ttft_ms)
    itl_stats = compute_optional_stats(itl_ms)
    tpot_stats = compute_optional_stats(tpot_ms)
    prefix_cache_hit_rate = (
        total_cached_tokens / total_input_tokens if total_input_tokens else None
    )
    output_token_stats = compute_int_stats(output_tokens_per_request)
    resource_summary = summarize_resource_samples(resource_monitor.samples)

    print("\n=== Results ===")
    print(f"  Prefix eviction   : {args.eviction_policy}")
    print(f"  Active KV eviction: {args.active_kv_eviction_policy}")
    print(f"  Requests measured : {stats['n']}")
    print(f"  E2E latency P50   : {stats['p50_ms']:.1f} ms")
    print(f"  E2E latency P95   : {stats['p95_ms']:.1f} ms")
    print(f"  E2E latency P99   : {stats['p99_ms']:.1f} ms")
    print(f"  E2E latency mean  : {stats['mean_ms']:.1f} ms")
    print(f"  Throughput        : {throughput_tok_s:.1f} tok/s")
    if resource_summary:
        print(
            "  GPU util mean     : "
            f"{resource_summary.get('gpu_util_percent_mean')} %"
        )
        print(
            "  CPU util mean     : "
            f"{resource_summary.get('cpu_percent_mean')} %"
        )

    hit_latencies = [r["e2e_ms"] for r in per_request if r["cached_tokens"] > 0]
    miss_latencies = [r["e2e_ms"] for r in per_request if r["cached_tokens"] == 0]
    cache_miss_penalty_ms = None
    if hit_latencies and miss_latencies:
        cache_miss_penalty_ms = mean(miss_latencies) - mean(hit_latencies)

    gpu_kv_cache_util = collect_gpu_kv_cache_util(llm)

    if args.output_json:
        if args.active_kv_eviction_policy != "none":
            cache_eviction_type = "active_kv"
            cache_eviction_policy = args.active_kv_eviction_policy
            cache_eviction_label = args.active_kv_eviction_policy
            if args.active_kv_eviction_cache_budget_tokens is not None:
                cache_eviction_label = (
                    f"{cache_eviction_label}-"
                    f"{args.active_kv_eviction_cache_budget_tokens}"
                )
        else:
            cache_eviction_type = "prefix"
            cache_eviction_policy = args.eviction_policy
            cache_eviction_label = args.eviction_policy

        result = {
            "eviction_policy": args.eviction_policy,
            "active_kv_eviction_policy": args.active_kv_eviction_policy,
            "active_kv_eviction_cache_budget_tokens": (
                args.active_kv_eviction_cache_budget_tokens
            ),
            "cache_eviction_type": cache_eviction_type,
            "cache_eviction_policy": cache_eviction_policy,
            "cache_eviction_label": cache_eviction_label,
            "model": args.model,
            "num_prompts": args.num_prompts,
            "repeat_count": args.repeat_count,
            "warmup_rounds": args.warmup_rounds,
            "input_length_range": args.input_length_range,
            "output_len": args.output_len,
            "prefix_len": args.prefix_len,
            "dataset_path": args.dataset_path,
            "seed": args.seed,
            "sort": args.sort,
            "enable_prefix_caching": args.enable_prefix_caching,
            "num_gpu_blocks_override": args.num_gpu_blocks_override,
            "max_model_len": args.max_model_len,
            "latency_stats": stats,
            "output_token_stats": output_token_stats,
            "throughput_tok_s": throughput_tok_s,
            "total_output_tokens": total_output_tokens,
            "total_input_tokens": total_input_tokens,
            "wall_time_s": wall_time_s,
            "phase_timings_s": phase_timings_s,
            "per_request_latency_ms": latencies_ms,
            "per_request_output_tokens": output_tokens_per_request,
            "time_to_first_token_ms": ttft_stats.get("mean_ms"),
            "inter_token_latency_ms": itl_stats.get("mean_ms"),
            "time_per_output_token_ms": tpot_stats.get("mean_ms"),
            "prefix_cache_hit_rate": prefix_cache_hit_rate,
            "cache_miss_penalty_ms": cache_miss_penalty_ms,
            "recompute_cost_avoided": total_cached_tokens,
            "eviction_regret": None,
            "gpu_kv_cache_util": gpu_kv_cache_util,
            "ttft_stats": ttft_stats,
            "itl_stats": itl_stats,
            "tpot_stats": tpot_stats,
            "per_request_cached_tokens": cached_tokens_per_request,
            "per_request_ttft_ms": ttft_ms,
            "per_request_itl_ms": itl_ms,
            "per_request_tpot_ms": tpot_ms,
            "resource_sample_interval_s": args.resource_sample_interval_s,
            "resource_summary": resource_summary,
            "resource_samples": resource_monitor.samples,
        }
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Results saved to  : {args.output_json}")


def create_argument_parser():
    parser = FlexibleArgumentParser(
        description="Benchmark the performance with or without "
        "automatic prefix caching."
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None, help="Path to the dataset."
    )
    parser.add_argument("--output-len", type=int, default=10)
    parser.add_argument(
        "--num-prompts",
        type=int,
        required=True,
        help="Number of the prompts sampled from dataset",
    )
    parser.add_argument(
        "--repeat-count",
        type=int,
        default=1,
        help="Number of times to repeat each prompt",
    )
    parser.add_argument(
        "--sort", action="store_true", help="Sort prompts by input length"
    )
    parser.add_argument(
        "--input-length-range",
        type=str,
        required=True,
        help="Range of input lengths for sampling prompts,"
        'specified as "min:max" (e.g., "128:256").',
    )
    parser.add_argument(
        "--prefix-len",
        type=int,
        default=0,
        help="Specifies the length of a common prefix to be "
        "added to the input prompt. The input-length-range will "
        "subtract this length when filtering prompts. Only used "
        "when dataset-path is not provided.",
    )
    parser.add_argument(
        "--disable-detokenize",
        action="store_true",
        help=(
            "Do not detokenize responses (i.e. do not include "
            "detokenization time in the latency measurement)"
        ),
    )
    parser.add_argument(
        "--respect-eos",
        action="store_true",
        help=(
            "Allow EOS/stop tokens to end generation before output-len. By "
            "default this benchmark ignores EOS so all policies generate the "
            "same number of output tokens."
        ),
    )
    parser.add_argument(
        "--warmup-rounds",
        type=int,
        default=1,
        help="Number of warm-up rounds before measurement (fills the cache).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to write JSON results file.",
    )
    parser.add_argument(
        "--resource-sample-interval-s",
        type=float,
        default=1.0,
        help=(
            "Seconds between CPU, memory, and GPU utilization samples. "
            "Set to 0 to disable resource sampling."
        ),
    )

    parser = EngineArgs.add_cli_args(parser)

    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    main(args)
