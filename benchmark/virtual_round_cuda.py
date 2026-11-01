"""GPU benchmark for lof.virtual_round with a FloatFormatDescriptor.

CUDA counterpart of virtual_round.py. Requires the extension built with
USE_CUDA=1 (in that build the CPU path of virtual_round is compiled out).

Timing uses CUDA events around each call, after a warmup, so it measures the
kernel plus per-call launch overhead -- not just the asynchronous launch. Note
that the descriptor overload clones the input on the device and then rounds the
copy in place, so every call includes one device-to-device copy.

Run (pick the GPU with CUDA_VISIBLE_DEVICES):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python3 benchmark/virtual_round_cuda.py
"""
import argparse
import csv
import math
import os

import torch

import LoFloat as lof
from LoFloat.formats import create_p3109_params, create_half_params


def sanitize(s):
    return ''.join(c if c.isalnum() else '_' for c in s)


def logspace_int(nmin, nmax, points):
    if points == 1:
        return [nmin]

    a = math.log10(nmin)
    b = math.log10(nmax)

    v = []
    for i in range(points):
        t = i / (points - 1)
        xi_int = round(10.0 ** (a + (b - a) * t))
        xi_int = max(nmin, min(nmax, xi_int))
        if not v or xi_int > v[-1]:
            v.append(xi_int)

    if v[-1] != nmax:
        v.append(nmax)

    return v


def check_half(params, round_mode, device):
    """Sanity check: rounding to binary16 with RNE must match torch's float->half
    cast. Fails loudly if the extension is returning garbage (e.g. a CPU-only
    build or a broken kernel) instead of silently benchmarking it."""
    x = torch.randn(1 << 20, device=device) * 100.0
    got = lof.virtual_round(x, params, round_mode=round_mode, stoch_len=0)
    ref = x.half().float()
    if not torch.equal(got, ref):
        bad = (got != ref).nonzero()[0].item()
        raise RuntimeError(
            f"virtual_round(binary16) mismatch vs torch half cast at index {bad}: "
            f"x={x[bad].item()!r} got={got[bad].item()!r} ref={ref[bad].item()!r}")
    print("Sanity check: binary16 RNE matches torch float->half cast (1M values).")


def bench_virtual_round(name, params, sizes, iters, warmup_iters, round_mode,
                        csv_writer, device):
    print(f"\nBenchmarking {name}:")
    print("-" * 100)

    type_name = sanitize(name)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    sink = 0.0

    for n in sizes:
        arr = torch.rand(n, dtype=torch.float32, device=device)

        for _ in range(warmup_iters):
            lof.virtual_round(arr, params, round_mode=round_mode, stoch_len=0)
        torch.cuda.synchronize(device)

        iters_used = iters
        if n >= 10_000_000:
            iters_used = min(iters_used, 5)
        if n >= 50_000_000:
            iters_used = min(iters_used, 3)
        iters_used = max(1, iters_used)

        samples_us = []
        for _ in range(iters_used):
            start.record()
            result = lof.virtual_round(arr, params, round_mode=round_mode, stoch_len=0)
            end.record()
            end.synchronize()
            samples_us.append(start.elapsed_time(end) * 1e3)  # ms -> us

        # Consume the result so the work can't be considered dead.
        sink += result[n // 2].item()

        avg_us = sum(samples_us) / len(samples_us)
        min_us = min(samples_us)
        max_us = max(samples_us)
        # Logical traffic: read n floats + write n floats (the internal clone
        # adds another read+write that is not counted here).
        gbps = (2 * 4 * n) / (min_us * 1e-6) / 1e9

        print(f"{name:24s}  n={n:>10d}  avg={avg_us:>10.2f} us  min={min_us:>10.2f} us  "
              f"max={max_us:>10.2f} us  best={gbps:>7.1f} GB/s  iters={iters_used}")

        csv_writer.writerow([type_name, n, avg_us, min_us, max_us, gbps,
                             iters_used, warmup_iters])

        del arr, result

    return sink


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--nmin", type=int, default=10)
    ap.add_argument("--nmax", type=int, default=100_000_000)
    ap.add_argument("--points", type=int, default=20)
    ap.add_argument("--csv", default="virtual_round_bench_cuda.csv")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available to torch.")

    device = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(12345)
    round_mode = lof.RoundingMode.RoundToNearestEven

    print("=" * 100)
    print("Virtual Round Benchmark (CUDA)")
    print(f"Device: {torch.cuda.get_device_name(device)}  "
          f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')})")
    print("=" * 100)

    formats = [
        ("binary8p4 (P3109<8,4>)", create_p3109_params(k=8, p=4, is_signed=True, saturating=True)),
        ("binary8p3 (P3109<8,3>)", create_p3109_params(k=8, p=3, is_signed=True, saturating=True)),
        ("half (binary16)",        create_half_params()),
    ]

    check_half(formats[2][1], round_mode, device)

    sizes = logspace_int(args.nmin, args.nmax, args.points)

    need_header = not os.path.exists(args.csv) or os.path.getsize(args.csv) == 0
    sink = 0.0
    with open(args.csv, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["type", "n", "avg_us", "min_us", "max_us", "best_GBps",
                        "iters_used", "warmup_iters"])
        for name, params in formats:
            sink += bench_virtual_round(name, params, sizes, args.iters, args.warmup,
                                        round_mode, w, device)
            f.flush()

    print("\n" + "=" * 100)
    print(f"Benchmark complete! Results appended to {args.csv}  (sink={sink:.4g})")
    print("=" * 100)


if __name__ == "__main__":
    main()
