#!/usr/bin/env python3
"""Compare delta-representation VRAM for fused LoRA vs dense task deltas.

This diagnostic does not train or load datasets. It reads LoRA adapter factors,
estimates the memory needed for SuperMerge's fused_lora_linear path, and compares
it with materializing full dense task deltas for the same modules. If CUDA is
available, it also performs allocation measurements and reports peak allocator
usage for each representation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from merging.runtime.utils import resolve_best_adapter


DEFAULT_TASKS = ["emotion", "intent", "kws", "langid", "speaker_ver", "asr", "vocalsound"]
DTYPES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def _bytes_to_gib(value: int | float) -> float:
    return float(value) / (1024**3)


def _fmt_gib(value: int | float) -> str:
    return f"{_bytes_to_gib(value):.3f} GiB"


def _resolve_device(raw: str) -> torch.device:
    value = str(raw).strip().lower()
    if value.isdigit():
        value = f"cuda:{value}"
    device = torch.device(value)
    if device.type != "cuda":
        raise ValueError(f"CUDA measurement requires a CUDA device, got '{raw}'.")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {device.index} is not available; "
            f"torch sees {torch.cuda.device_count()} CUDA device(s)."
        )
    return device


def _element_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _resolve_paths(tasks: Sequence[str], explicit_paths: Sequence[str]) -> Dict[str, Path]:
    if explicit_paths:
        if len(explicit_paths) != len(tasks):
            raise ValueError("--adapter-path must be supplied once per task when used.")
        return {task: Path(path).expanduser().resolve() for task, path in zip(tasks, explicit_paths)}
    paths: Dict[str, Path] = {}
    for task in tasks:
        path, _metadata = resolve_best_adapter(task)
        paths[task] = path.resolve()
    return paths


def _scale_for_key(pattern: Mapping[str, object], key: str, default: float) -> float:
    if key in pattern:
        return float(pattern[key])
    suffix_matches = [k for k in pattern if key.endswith(str(k))]
    if not suffix_matches:
        return float(default)
    return float(pattern[max(suffix_matches, key=len)])


def _load_lora(adapter_path: Path) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, float]]:
    config_path = adapter_path / "adapter_config.json"
    weight_path = adapter_path / "adapter_model.safetensors"
    if not config_path.exists() or not weight_path.exists():
        raise FileNotFoundError(f"Missing LoRA adapter files under {adapter_path}")

    with config_path.open("r") as handle:
        cfg = json.load(handle)
    default_rank = float(cfg["r"])
    default_alpha = float(cfg["lora_alpha"])
    rank_pattern = cfg.get("rank_pattern") or {}
    alpha_pattern = cfg.get("alpha_pattern") or {}
    if not isinstance(rank_pattern, Mapping) or not isinstance(alpha_pattern, Mapping):
        raise ValueError(f"rank_pattern/alpha_pattern must be mappings for {adapter_path}")

    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    with safe_open(weight_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if ".lora_A." in key:
                base_key = key.replace(".lora_A.weight", "")
                pairs.setdefault(base_key, {})["A"] = handle.get_tensor(key).clone()
            elif ".lora_B." in key:
                base_key = key.replace(".lora_B.weight", "")
                pairs.setdefault(base_key, {})["B"] = handle.get_tensor(key).clone()

    out: Dict[str, Tuple[torch.Tensor, torch.Tensor, float]] = {}
    for key, pair in pairs.items():
        if "A" not in pair or "B" not in pair:
            raise ValueError(f"Incomplete LoRA factor pair for {key} in {adapter_path}")
        rank = _scale_for_key(rank_pattern, key, default_rank)
        alpha = _scale_for_key(alpha_pattern, key, default_alpha)
        out[key] = (pair["A"], pair["B"], float(alpha / rank))
    return out


def _selected_keys(per_task: Mapping[str, Mapping[str, object]], mode: str) -> List[str]:
    key_sets = [set(values.keys()) for values in per_task.values()]
    if mode == "common":
        return sorted(set.intersection(*key_sets))
    if mode == "union":
        return sorted(set.union(*key_sets))
    raise ValueError("--module-mode must be common or union")


def _estimate_bytes(
    per_task: Mapping[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, float]]],
    keys: Iterable[str],
    dtype: torch.dtype,
) -> Tuple[int, int, int, int]:
    elem = _element_size(dtype)
    fused = 0
    dense = 0
    modules = 0
    pairs = 0
    for key in keys:
        modules += 1
        for task_factors in per_task.values():
            if key not in task_factors:
                continue
            a, b, _scale = task_factors[key]
            fused += (a.numel() + b.numel()) * elem
            dense += (int(b.shape[0]) * int(a.shape[1])) * elem
            pairs += 1
    return fused, dense, modules, pairs


def _measure_cuda(
    per_task: Mapping[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, float]]],
    keys: Sequence[str],
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, int]:
    results: Dict[str, int] = {}
    torch.cuda.set_device(device)
    device_index = torch.cuda.current_device()
    active_device = torch.device(f"cuda:{device_index}")

    def reset() -> None:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_index)
        torch.cuda.synchronize(device_index)

    with torch.no_grad():
        reset()
        fused_tensors = []
        for key in keys:
            for task_factors in per_task.values():
                if key not in task_factors:
                    continue
                a, b, _scale = task_factors[key]
                fused_tensors.append(a.to(device=active_device, dtype=dtype))
                fused_tensors.append(b.to(device=active_device, dtype=dtype))
        torch.cuda.synchronize(device_index)
        results["fused_lora_allocated"] = torch.cuda.memory_allocated(device_index)
        results["fused_lora_peak"] = torch.cuda.max_memory_allocated(device_index)
        del fused_tensors
        reset()

        dense_tensors = []
        for key in keys:
            for task_factors in per_task.values():
                if key not in task_factors:
                    continue
                a, b, scale = task_factors[key]
                a_gpu = a.to(device=active_device, dtype=dtype)
                b_gpu = b.to(device=active_device, dtype=dtype)
                dense_tensors.append((b_gpu @ a_gpu) * float(scale))
        torch.cuda.synchronize(device_index)
        results["dense_delta_allocated"] = torch.cuda.memory_allocated(device_index)
        results["dense_delta_peak"] = torch.cuda.max_memory_allocated(device_index)
        del dense_tensors
        reset()

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--adapter-path", action="append", default=[])
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--module-mode", choices=["common", "union"], default="common")
    parser.add_argument("--device", default="cuda:0", help="CUDA device, e.g. cuda:0 or 0.")
    parser.add_argument("--no-cuda-measure", action="store_true")
    args = parser.parse_args()

    dtype = DTYPES[args.dtype]
    paths = _resolve_paths(args.tasks, args.adapter_path)
    per_task = {task: _load_lora(path) for task, path in paths.items()}
    keys = _selected_keys(per_task, args.module_mode)
    fused_bytes, dense_bytes, modules, pairs = _estimate_bytes(per_task, keys, dtype)

    print("SuperMerge VRAM representation diagnostic")
    print(f"tasks: {', '.join(args.tasks)}")
    print(f"dtype: {args.dtype}")
    print(f"module_mode: {args.module_mode}")
    print(f"modules: {modules}")
    print(f"task/module pairs: {pairs}")
    print()
    print("Adapter paths:")
    for task, path in paths.items():
        print(f"  {task}: {path}")
    print()
    print("Estimated tensor residency:")
    print(f"  fused_lora_linear factors: {_fmt_gib(fused_bytes)} ({fused_bytes:,} bytes)")
    print(f"  materialized dense deltas: {_fmt_gib(dense_bytes)} ({dense_bytes:,} bytes)")
    if fused_bytes:
        print(f"  dense / fused ratio: {dense_bytes / fused_bytes:.2f}x")

    cuda_ok = torch.cuda.is_available()
    if args.no_cuda_measure or not cuda_ok:
        reason = "--no-cuda-measure set" if args.no_cuda_measure else "CUDA unavailable"
        print()
        print(f"CUDA allocation measurement skipped: {reason}.")
        return 0

    device = _resolve_device(args.device)
    measured = _measure_cuda(per_task, keys, dtype, device)
    print()
    print(f"Measured CUDA allocator usage on {device}:")
    print(f"  fused_lora_linear allocated: {_fmt_gib(measured['fused_lora_allocated'])}")
    print(f"  fused_lora_linear peak: {_fmt_gib(measured['fused_lora_peak'])}")
    print(f"  dense_delta allocated: {_fmt_gib(measured['dense_delta_allocated'])}")
    print(f"  dense_delta peak: {_fmt_gib(measured['dense_delta_peak'])}")
    if measured["fused_lora_peak"]:
        ratio = measured["dense_delta_peak"] / measured["fused_lora_peak"]
        print(f"  measured peak dense / fused ratio: {ratio:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
