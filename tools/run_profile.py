"""
CVPR-style model profiling script.

Compute:
- Params
- Trainable Params
- GFLOPs (fvcore)
- Latency
- FPS
- GPU Memory

Usage:
python tools/profile.py -c configs/rtdetrv2/rtdetrv2_dinov3_vit_6x_coco.yml
"""

import os
import sys
import time
import argparse
import contextlib

import torch
from fvcore.nn import FlopCountAnalysis
from torch.profiler import profile, ProfilerActivity, record_function

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.core import YAMLConfig, yaml_utils


def _autocast_context(device: torch.device, enabled: bool = True, dtype=torch.float16):
    if enabled and device.type == "cuda":
        return torch.cuda.amp.autocast(dtype=dtype)
    return contextlib.nullcontext()


def _first_tensor(obj):
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = _first_tensor(v)
            if t is not None:
                return t
    return None


def compute_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "total_m": total / 1e6,
        "trainable_m": trainable / 1e6,
    }


def compute_flops(model, x):
    model.eval()

    device = x.device

    amp_enabled = device.type == "cuda"
    amp_dtype = torch.float16

    # fvcore will execute the forward to trace ops; run under inference + optional autocast.
    with torch.inference_mode(), _autocast_context(device, enabled=amp_enabled, dtype=amp_dtype):
        flops = FlopCountAnalysis(model, x)
        total_flops = flops.total()

    return {
        "flops": total_flops,
        "gflops": total_flops / 1e9,
        "mflops": total_flops / 1e6,
    }


@torch.inference_mode()
def profile_latency(
    model,
    x,
    warmup=20,
    iters=50,
    row_limit=30,
    sync_sections=False,
    postprocess_d2h=False,
):
    model.eval()

    device = x.device
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.float16

    # Match requested settings for CUDA inference.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    for _ in range(warmup):
        with _autocast_context(device, enabled=amp_enabled, dtype=amp_dtype):
            _ = model(x)

    if x.device.type == "cuda":
        torch.cuda.synchronize()
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
        sort_by = "cuda_time_total"
    else:
        activities = [ProfilerActivity.CPU]
        sort_by = "cpu_time_total"

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for _ in range(iters):
            if x.device.type == "cuda":
                x_cpu = x.detach().cpu()
            else:
                x_cpu = x

            with record_function("preprocess"):
                if x.device.type == "cuda":
                    x_iter = x_cpu.to(x.device, dtype=x.dtype, non_blocking=False)
                else:
                    x_iter = x_cpu
                if sync_sections and x.device.type == "cuda":
                    torch.cuda.synchronize()

            with record_function("forward"):
                with _autocast_context(device, enabled=amp_enabled, dtype=amp_dtype):
                    out = model(x_iter)
                if sync_sections and x.device.type == "cuda":
                    torch.cuda.synchronize()

            with record_function("postprocess"):
                if postprocess_d2h:
                    t = _first_tensor(out)
                    if t is not None:
                        # Minimal D2H touch to represent “after model” overhead.
                        if t.device.type == "cuda":
                            _ = t.reshape(-1)[:1].to("cpu", non_blocking=False)
                        else:
                            _ = t.reshape(-1)[:1].clone()
                if sync_sections and x.device.type == "cuda":
                    torch.cuda.synchronize()

    if x.device.type == "cuda":
        torch.cuda.synchronize()

    print("\n===== Torch Profiler (Top Ops) =====")
    print(
        prof.key_averages().table(
            sort_by=sort_by,
            row_limit=row_limit,
        )
    )
    print("===================================\n")


@torch.inference_mode()
def compute_fps(
    model,
    x,
    warmup=50,
    iters=200,
):
    model.eval()

    device = x.device

    amp_enabled = device.type == "cuda"
    amp_dtype = torch.float16

    # Match requested settings for CUDA inference.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    # warmup
    for _ in range(warmup):
        with _autocast_context(device, enabled=amp_enabled, dtype=amp_dtype):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        starter.record()

        for _ in range(iters):
            with _autocast_context(device, enabled=amp_enabled, dtype=amp_dtype):
                _ = model(x)

        ender.record()
        torch.cuda.synchronize()

        elapsed = starter.elapsed_time(ender)

    else:
        start = time.time()

        for _ in range(iters):
            with _autocast_context(device, enabled=amp_enabled, dtype=amp_dtype):
                _ = model(x)

        elapsed = (time.time() - start) * 1000

    latency = elapsed / iters
    fps = 1000 / latency

    return {
        "latency_ms": latency,
        "fps": fps,
    }


def compute_gpu_mem():
    if torch.cuda.is_available():
        mem = torch.cuda.max_memory_allocated() / 1024**2
        return mem
    return 0


def profile_model(
    config,
    device="cuda:0",
    shape=(1, 3, 640, 640),
    warmup=50,
    iters=200,
    update=None,
    do_profiler=False,
    profiler_warmup=20,
    profiler_iters=50,
    profiler_row_limit=30,
    profiler_sync_sections=False,
    profiler_postprocess_d2h=False,
):

    update_dict = yaml_utils.parse_cli(update) if update else {}

    cfg = YAMLConfig(config, **update_dict)

    model = cfg.model.to(device)

    # Match requested settings for CUDA inference.
    try:
        device_obj = torch.device(device)
    except Exception:
        device_obj = None
    if device_obj is None or device_obj.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    dtype = next(model.parameters()).dtype

    x = torch.randn(*shape, device=device, dtype=dtype)

    print("Building model...")

    params = compute_params(model)

    print("Computing FLOPs...")
    flops = compute_flops(model, x)

    print("Benchmarking speed...")
    speed = compute_fps(model, x, warmup, iters)

    if do_profiler:
        profile_latency(
            model,
            x,
            warmup=profiler_warmup,
            iters=profiler_iters,
            row_limit=profiler_row_limit,
            sync_sections=profiler_sync_sections,
            postprocess_d2h=profiler_postprocess_d2h,
        )

    mem = compute_gpu_mem()

    return {
        **params,
        **flops,
        **speed,
        "gpu_mem": mem,
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="config yaml",
    )

    parser.add_argument(
        "-d",
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--shape",
        nargs=4,
        type=int,
        default=[1, 3, 640, 640],
    )

    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)

    parser.add_argument(
        "-u",
        "--update",
        nargs="+",
        help="yaml override",
    )

    parser.add_argument(
        "--profiler",
        action="store_true",
        help="run torch.profiler to show per-op latency breakdown",
    )
    parser.add_argument("--profiler-warmup", type=int, default=20)
    parser.add_argument("--profiler-iters", type=int, default=50)
    parser.add_argument("--profiler-row-limit", type=int, default=30)
    parser.add_argument(
        "--profiler-sync",
        action="store_true",
        help="synchronize at section boundaries to approximate wall-time per section",
    )
    parser.add_argument(
        "--profiler-d2h",
        action="store_true",
        help="in postprocess, do a minimal GPU->CPU touch of outputs",
    )

    args = parser.parse_args()

    stats = profile_model(
        config=args.config,
        device=args.device,
        shape=tuple(args.shape),
        warmup=args.warmup,
        iters=args.iters,
        update=args.update,
        do_profiler=args.profiler,
        profiler_warmup=args.profiler_warmup,
        profiler_iters=args.profiler_iters,
        profiler_row_limit=args.profiler_row_limit,
        profiler_sync_sections=args.profiler_sync,
        profiler_postprocess_d2h=args.profiler_d2h,
    )

    print("\n===== Model Profile =====")

    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print(f"Input: {args.shape}")

    print(
        f"Params (Total / Trainable): "
        f"{stats['total_m']:.2f}M / {stats['trainable_m']:.2f}M"
    )

    print(
        f"FLOPs: {stats['gflops']:.2f} GFLOPs"
    )

    print(
        f"Latency: {stats['latency_ms']:.2f} ms"
    )

    print(
        f"FPS: {stats['fps']:.2f}"
    )

    if stats["gpu_mem"] > 0:
        print(
            f"Max GPU Memory: {stats['gpu_mem']:.1f} MB"
        )

    print("=========================")


if __name__ == "__main__":
    main()