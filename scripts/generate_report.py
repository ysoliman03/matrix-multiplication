#!/usr/bin/env python3
"""
generate_report.py

Reads benchmark CSV output (from stdin or a file) and produces a markdown
report at results/benchmark_results.md.

Usage:
    ./matmul_benchmark | python3 scripts/generate_report.py
    python3 scripts/generate_report.py results.csv
"""

import sys
import os
import re
from datetime import date
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
# Parse input
# ─────────────────────────────────────────────────────────────

KERNEL_DISPLAY = {
    "naive":             "1. Naive",
    "global_coalesce":   "2. Global Coalescing",
    "shared_mem_tiling": "3. Shared Memory Tiling",
    "1d_blocktiling":    "4. 1D Block Tiling",
    "2d_blocktiling":    "5. 2D Block Tiling",
    "vectorized":        "6. Vectorized (float4)",
    "cublas":            "cuBLAS (reference)",
}

KERNEL_ORDER = [
    "naive",
    "global_coalesce",
    "shared_mem_tiling",
    "1d_blocktiling",
    "2d_blocktiling",
    "vectorized",
    "cublas",
]

STAGE_DESCRIPTIONS = [
    (
        "naive -> global_coalesce",
        "**Global Memory Coalescing**: Remaps `threadIdx.x` to the column dimension "
        "so that threads in a warp access adjacent columns of B and C — contiguous "
        "in memory — instead of adjacent rows (which are `N` floats apart). "
        "This converts uncoalesced 32-transaction reads into single 128-byte "
        "transactions, yielding a significant bandwidth improvement."
    ),
    (
        "global_coalesce -> shared_mem_tiling",
        "**Shared Memory Tiling**: Divides the K dimension into 32-element tiles. "
        "Each thread block cooperatively loads a tile of A and B into on-chip "
        "shared memory (≈1-2 cycle latency vs ≈200 cycles for global memory), "
        "then all threads read from shared memory to compute their partial sums. "
        "Each global memory element is loaded once but reused 32 times, reducing "
        "global memory traffic by 32×."
    ),
    (
        "shared_mem_tiling -> 1d_blocktiling",
        "**1D Thread Coarsening**: Each thread now computes 8 output elements "
        "(a vertical strip of 8 rows in the same column) instead of just one. "
        "This increases arithmetic intensity — 8 accumulators share the cost of "
        "loading B, amortizing memory latency — and provides more independent "
        "instructions for the GPU's out-of-order execution to overlap."
    ),
    (
        "1d_blocktiling -> 2d_blocktiling",
        "**2D Block Tiling (Outer Product)**: Extends thread coarsening to 2D: "
        "each thread computes an 8×8 = 64-element sub-block of C. Per inner-loop "
        "iteration, 8 A-elements and 8 B-elements are loaded into registers and "
        "their outer product (64 MACs) is accumulated. Arithmetic intensity is "
        "8×8/(8+8) = 4× better than 1D tiling, pushing closer to peak FLOP/s."
    ),
    (
        "2d_blocktiling -> vectorized",
        "**Vectorized float4 Loads**: Replaces scalar `float` global memory loads "
        "with 128-bit `float4` vector loads, fetching 4 floats per memory "
        "instruction instead of 1. This reduces load-instruction count by 4×, "
        "decreases instruction-issue pressure, and allows the memory subsystem "
        "to operate more efficiently — particularly helpful at large matrix sizes "
        "where the kernel is memory-bandwidth bound."
    ),
]


def parse_input(lines):
    gpu_name = "Unknown GPU"
    data = defaultdict(dict)   # data[kernel][size] = (time_ms, gflops, pct)
    sizes = []

    for line in lines:
        line = line.strip()

        # Pick up GPU name from the info block
        m = re.match(r'\s*Device:\s+(.+)', line)
        if m:
            gpu_name = m.group(1).strip()
            continue

        # CSV data rows
        parts = line.split(',')
        if len(parts) != 5:
            continue
        kernel, size_s, time_s, gflops_s, pct_s = parts
        if kernel == 'kernel':  # header guard
            continue
        try:
            size   = int(size_s)
            time_ms = float(time_s) if time_s != 'NaN' else None
            gflops  = float(gflops_s) if gflops_s != 'NaN' else None
            pct_str = pct_s.replace('%', '').strip()
            pct     = float(pct_str) if pct_str != 'NaN' else None
        except ValueError:
            continue

        data[kernel][size] = (time_ms, gflops, pct)
        if size not in sizes:
            sizes.append(size)

    sizes.sort()
    return gpu_name, data, sizes


# ─────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────

def cell(data, kernel, size):
    if kernel not in data or size not in data[kernel]:
        return "—"
    time_ms, gflops, pct = data[kernel][size]
    if gflops is None:
        return "FAIL"
    if kernel == "cublas":
        return f"{gflops:.1f} GFLOPS"
    return f"{gflops:.1f} GFLOPS ({pct:.1f}%)"


def generate_report(gpu_name, data, sizes):
    lines = []

    # ── Header ──────────────────────────────────────────────
    lines.append("# CUDA Matrix Multiplication: Performance Report\n")
    lines.append(f"**GPU**: {gpu_name}  ")
    lines.append(f"**Date**: {date.today().isoformat()}  ")
    lines.append(f"**Sizes tested**: {', '.join(str(s)+'×'+str(s) for s in sizes)}\n")

    # ── Performance Table ────────────────────────────────────
    lines.append("## Performance Summary\n")
    col_header = " | ".join(f"{s}×{s}" for s in sizes)
    lines.append(f"| Kernel | {col_header} |")
    lines.append("|" + "|".join(["-" * 25] + ["-" * 20] * len(sizes)) + "|")

    for k in KERNEL_ORDER:
        display = KERNEL_DISPLAY.get(k, k)
        cells = " | ".join(cell(data, k, s) for s in sizes)
        lines.append(f"| {display} | {cells} |")

    lines.append("")

    # ── Stage-by-stage analysis ──────────────────────────────
    lines.append("## Stage-by-Stage Analysis\n")
    largest = sizes[-1] if sizes else None

    stage_keys = [
        ("naive",             "global_coalesce"),
        ("global_coalesce",   "shared_mem_tiling"),
        ("shared_mem_tiling", "1d_blocktiling"),
        ("1d_blocktiling",    "2d_blocktiling"),
        ("2d_blocktiling",    "vectorized"),
    ]

    for (prev_k, next_k), (label, desc) in zip(stage_keys, STAGE_DESCRIPTIONS):
        lines.append(f"### {label}\n")
        lines.append(desc)

        if largest and prev_k in data and next_k in data:
            prev_entry = data[prev_k].get(largest)
            next_entry = data[next_k].get(largest)
            if prev_entry and next_entry and prev_entry[1] and next_entry[1]:
                speedup = next_entry[1] / prev_entry[1]
                lines.append(
                    f"\n**Speedup at {largest}×{largest}**: "
                    f"{prev_entry[1]:.1f} -> {next_entry[1]:.1f} GFLOPS "
                    f"(**{speedup:.2f}×**)"
                )
        lines.append("")

    # ── Summary ─────────────────────────────────────────────
    lines.append("## Summary\n")

    if largest and "vectorized" in data and "cublas" in data:
        vec_entry = data["vectorized"].get(largest)
        cub_entry = data["cublas"].get(largest)
        if vec_entry and vec_entry[1] and cub_entry and cub_entry[1]:
            pct = vec_entry[1] / cub_entry[1] * 100.0
            lines.append(
                f"The final vectorized kernel achieves **{vec_entry[1]:.1f} GFLOPS** "
                f"at {largest}×{largest}, reaching **{pct:.1f}%** of cuBLAS performance."
            )

    # Biggest single speedup
    if largest:
        best_label = None
        best_mult = 0.0
        for (prev_k, next_k), (label, _) in zip(stage_keys, STAGE_DESCRIPTIONS):
            prev_entry = data.get(prev_k, {}).get(largest)
            next_entry = data.get(next_k, {}).get(largest)
            if prev_entry and next_entry and prev_entry[1] and next_entry[1]:
                mult = next_entry[1] / prev_entry[1]
                if mult > best_mult:
                    best_mult = mult
                    best_label = label
        if best_label:
            lines.append(
                f"\nThe largest single performance jump was **{best_label}** "
                f"({best_mult:.2f}× speedup), demonstrating the most impactful "
                f"optimization in this journey."
            )

    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    gpu_name, data, sizes = parse_input(lines)

    if not sizes:
        print("No benchmark data found in input.", file=sys.stderr)
        sys.exit(1)

    report = generate_report(gpu_name, data, sizes)

    os.makedirs("results", exist_ok=True)
    out_path = "results/benchmark_results.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report written to {out_path}", file=sys.stderr)
    sys.stdout.buffer.write(report.encode("utf-8"))


if __name__ == "__main__":
    main()
