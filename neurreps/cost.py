from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import nn

from .models import count_trainable_parameters


def _module_macs(model: nn.Module, sample: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[int, int]:
    macs = 0
    peak_activation_bytes = 0
    hooks = []

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any):
        nonlocal macs, peak_activation_bytes
        tensor = output[0] if isinstance(output, tuple) else output
        if isinstance(tensor, torch.Tensor):
            peak_activation_bytes = max(peak_activation_bytes, tensor.numel() * tensor.element_size())
        if isinstance(module, nn.Conv1d) and isinstance(tensor, torch.Tensor):
            out_elements = tensor.numel()
            kernel_work = (module.in_channels // module.groups) * module.kernel_size[0]
            macs += int(out_elements * kernel_work)
        elif isinstance(module, nn.Linear) and isinstance(tensor, torch.Tensor):
            batch_outputs = tensor.numel()
            macs += int(batch_outputs * module.in_features)

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook))
    with torch.no_grad():
        model(sample, mask=mask)
    for handle in hooks:
        handle.remove()
    return int(macs), int(peak_activation_bytes)


def benchmark_inference_latency(
    model: nn.Module,
    sample: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    warmup: int = 20,
    repeats: int = 100,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    sample = sample.to(device)
    mask = mask.to(device) if mask is not None else None
    with torch.no_grad():
        for _ in range(int(warmup)):
            model(sample, mask=mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings = []
        for _ in range(int(repeats)):
            start = time.perf_counter_ns()
            model(sample, mask=mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "latency_ms_median": float(np.median(timings)),
        "latency_ms_mean": float(np.mean(timings)),
        "latency_ms_p95": float(np.quantile(timings, 0.95)),
        "latency_device": device.type,
        "latency_batch_size": int(sample.shape[0]),
    }


def analyze_model_cost(
    model: nn.Module,
    X: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    representation: str,
    latency_warmup: int = 20,
    latency_repeats: int = 100,
) -> dict[str, float | int | str]:
    if len(X) == 0:
        raise ValueError("Cost analysis requires at least one input")
    device = next(model.parameters()).device
    sample_np = np.asarray(X[:1])
    sample = torch.from_numpy(sample_np).to(device)
    mask_tensor = None
    if mask is not None:
        mask_tensor = torch.from_numpy(np.asarray(mask[:1]).astype(np.float32)).to(device)
    macs, peak_activation_bytes = _module_macs(model, sample, mask_tensor)
    parameters = count_trainable_parameters(model)
    stored_input_bytes = int(sample_np.nbytes)
    if representation.upper() == "32C":
        theoretical_input_bits = int(np.prod(sample_np.shape[1:]))
    else:
        theoretical_input_bits = int(np.prod(sample_np.shape[1:]) * 8)
    result: dict[str, float | int | str] = {
        "representation": representation.upper(),
        "parameters": parameters,
        "macs_per_sample": int(macs),
        "weights_fp32_bytes": int(parameters * 4),
        "weights_int8_bytes": int(parameters),
        "input_stored_bytes": stored_input_bytes,
        "input_theoretical_bits": theoretical_input_bits,
        "input_theoretical_bytes": int(np.ceil(theoretical_input_bits / 8)),
        "peak_single_activation_bytes": int(peak_activation_bytes),
    }
    result.update(
        benchmark_inference_latency(
            model,
            sample,
            mask=mask_tensor,
            warmup=latency_warmup,
            repeats=latency_repeats,
        )
    )
    return result
