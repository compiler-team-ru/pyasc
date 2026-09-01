#!/usr/bin/env python3
# Copyright (c) 2026 AISS Group, Harbin Institute of Technology.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
import csv
import glob
import os
import shutil
import subprocess
import time
from pathlib import Path

# (B, H, S, D). Cover common Llama-1 prefill configurations:
# 7B uses 32 heads, 13B uses 40 heads, and 65B uses 64 heads; D is 128.
SHAPES = [
    (1, 32, 128, 128),
    (1, 32, 512, 128),
    (1, 32, 1024, 128),
    (2, 32, 512, 128),
    (1, 40, 512, 128),
    (1, 64, 512, 128),
]
BACKENDS = ("pyasc", "ascendc", "torch_npu")
WARMUP, ITERS = 5, 10


# Run one subprocess command in cwd and raise on failure.
# Inputs: command tokens and the working directory.
# Output: None; subprocess failures are propagated.
def run(cmd, cwd):
    print("[RUN]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)


# Return the most recently modified path matching p.
# Inputs: a recursive glob pattern.
# Output: the newest matching path.
def latest(p):
    matches = glob.glob(p, recursive=True)
    if not matches:
        raise FileNotFoundError(p)
    return max(matches, key=os.path.getmtime)


# Convert a CSV value to float, returning zero for missing values.
# Inputs: one optional CSV field value.
# Output: the parsed float or 0.0.
def to_float(v):
    return 0.0 if v in (None, "", "NA", "N/A") else float(str(v).strip().replace("\t", ""))


# Read target-kernel duration from OpBasicInfo.csv.
# Inputs: profile directory and accepted operator-name prefix or prefixes.
# Output: matching CSV rows and the source CSV path.
def read_basic(d, n):
    try:
        p = Path(latest(str(d / "OPPROF_*" / "OpBasicInfo.csv")))
    except FileNotFoundError:
        p = Path(latest(str(d / "OPPROF_*" / "**" / "OpBasicInfo*.csv")))
    rows = list(csv.DictReader(p.open(newline="")))
    m = [r for r in rows if r.get("Op Name", "").startswith(n)]
    if not m:
        raise RuntimeError(f"expected {n}*, got {[r.get('Op Name') for r in rows]}")
    return m, p


# Read pipeline counters from the newest PipeUtilization.csv.
# Inputs: one msprof output directory.
# Output: parsed pipeline rows and the source CSV path.
def read_pipe(prof_dir):
    path = Path(latest(str(prof_dir / "OPPROF_*" / "PipeUtilization.csv")))
    rows = list(csv.DictReader(path.open(newline="")))
    return rows, path


# Return the largest numeric value of key across parsed rows.
# Inputs: parsed CSV rows and a field name.
# Output: the maximum field value, or 0.0 for no rows.
def max_field(rows, key):
    values = [to_float(row.get(key)) for row in rows]
    return max(values) if values else 0.0


# Profile one backend/shape and return its summary and pipeline metrics.
# Inputs: sample paths, operator names, backend, and BNSD shape.
# Output: one dictionary of duration, block, pipeline, and source metadata.
def profile_one(ed, out, bench, demo, ot, backend, shape):
    batch, heads, seq_len, head_dim = shape
    o = out / f"{backend}_{batch}x{heads}x{seq_len}x{head_dim}"
    shutil.rmtree(o, ignore_errors=True)
    if backend == "pyasc":
        cmd = [
            "msprof", "op", "--kernel-name=fused_infer_attention_kernel", f"--output={o}", "python3",
            str(bench), "--backend", "pyasc", "--batch",
            str(batch), "--seq",
            str(seq_len), "--heads",
            str(heads), "--head_dim",
            str(head_dim), "--warmup",
            str(WARMUP), "--iters",
            str(ITERS), "-r", "NPU", "-v", "Ascend910B4"
        ]
    elif backend == "ascendc":
        cmd = [
            "msprof", "op", f"--output={o}",
            str(demo), "--batch",
            str(batch), "--seq",
            str(seq_len), "--heads",
            str(heads), "--head_dim",
            str(head_dim), "--warmup",
            str(WARMUP), "--iters",
            str(ITERS)
        ]
    else:  # torch_npu
        cmd = [
            "msprof", "op", "--kernel-name=FusedInferAttentionScore", f"--output={o}", "python3",
            str(bench), "--backend", "torch_npu", "--batch",
            str(batch), "--seq",
            str(seq_len), "--heads",
            str(heads), "--head_dim",
            str(head_dim), "--warmup",
            str(WARMUP), "--iters",
            str(ITERS)
        ]
    run(cmd, ed)
    rows, bp = read_basic(o, ot[backend])
    ds = [to_float(r.get("Task Duration(us)")) for r in rows]
    td = ds[WARMUP:] if len(ds) > WARMUP else ds
    pipe_rows, pipe_path = read_pipe(o)
    return {
        "backend": backend,
        "shape": shape,
        "elements": batch * heads * seq_len * seq_len,
        "task_avg_us": sum(td) / len(td),
        "block_dim": int(to_float(rows[0].get("Block Dim"))),
        "mix_block_dim": int(to_float(rows[0].get("Mix Block Dim"))),
        "aic_cube_us": max_field(pipe_rows, "aic_cube_time(us)"),
        "aic_scalar_us": max_field(pipe_rows, "aic_scalar_time(us)"),
        "aic_mte1_us": max_field(pipe_rows, "aic_mte1_time(us)"),
        "aic_mte2_us": max_field(pipe_rows, "aic_mte2_time(us)"),
        "aic_mte3_us": max_field(pipe_rows, "aic_mte3_time(us)"),
        "aic_fixpipe_us": max_field(pipe_rows, "aic_fixpipe_time(us)"),
        "aiv_vec_us": max_field(pipe_rows, "aiv_vec_time(us)"),
        "aiv_scalar_us": max_field(pipe_rows, "aiv_scalar_time(us)"),
        "aiv_mte2_us": max_field(pipe_rows, "aiv_mte2_time(us)"),
        "aiv_mte3_us": max_field(pipe_rows, "aiv_mte3_time(us)"),
        "source_csv": str(bp),
        "pipe_source_csv": str(pipe_path),
    }


# Write one summary row per Attention shape.
# Inputs: destination path and profile-result rows.
# Output: None; writes summary.csv.
def write_summary(path, rows):
    by = {(r["backend"], r["shape"]): r for r in rows}
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "shape",
            "elements",
            "pyasc_us",
            "ascendc_us",
            "torch_npu_us",
            "pyasc_over_ascendc",
            "pyasc_over_torch_npu",
        ])
        for s in SHAPES:
            p = by.get(("pyasc", s))
            a = by.get(("ascendc", s))
            t = by.get(("torch_npu", s))
            w.writerow([
                f"({s[0]},{s[1]},{s[2]},{s[3]})",
                p["elements"] if p else (a["elements"] if a else (t["elements"] if t else 0)),
                f"{(p['task_avg_us'] if p else 0):.6f}",
                f"{(a['task_avg_us'] if a else 0):.6f}",
                f"{(t['task_avg_us'] if t else 0):.6f}",
                f"{((p['task_avg_us'] / a['task_avg_us']) if p and a else 0):.6f}",
                f"{((p['task_avg_us'] / t['task_avg_us']) if p and t else 0):.6f}",
            ])


# Convert one profile row into pipeline-detail CSV fields.
# Inputs: one profile dictionary and the ordered output fields.
# Output: a CSV-ready dictionary.
def format_detail_row(row, fields):
    output = dict(row)
    output["shape"] = "(" + ",".join(map(str, row["shape"])) + ")"
    for key, value in output.items():
        if isinstance(value, float):
            output[key] = f"{value:.6f}"
    return {field: output[field] for field in fields}


# Write per-backend pipeline counters to pipeline_detail.csv.
# Inputs: destination path and profile-result rows.
# Output: None; writes pipeline_detail.csv.
def write_detail(path, rows):
    fields = [
        "backend",
        "shape",
        "elements",
        "task_avg_us",
        "block_dim",
        "mix_block_dim",
        "aic_cube_us",
        "aic_scalar_us",
        "aic_mte1_us",
        "aic_mte2_us",
        "aic_mte3_us",
        "aic_fixpipe_us",
        "aiv_vec_us",
        "aiv_scalar_us",
        "aiv_mte2_us",
        "aiv_mte3_us",
        "source_csv",
        "pipe_source_csv",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_detail_row(row, fields))


# Profile the selected backends and write both CSV files.
# Inputs: command-line output and backend options.
# Output: None; writes summary.csv and pipeline_detail.csv.
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--backends",
        default=",".join(BACKENDS),
        help="Comma-separated backends: pyasc, ascendc, torch_npu.",
    )
    a = p.parse_args()
    backends = tuple(item.strip() for item in a.backends.split(",") if item.strip())
    unknown = set(backends) - set(BACKENDS)
    if unknown:
        p.error(f"unknown backends: {','.join(sorted(unknown))}")
    ed = Path(__file__).resolve().parent
    bench = ed / "bench_fused_infer_attention.py"
    ot = {
        "pyasc": "fused_infer_attention_kernel",
        "ascendc": "fused_infer_attention_kernel",
        "torch_npu": "FusedInferAttentionScore",
    }
    demo = ed / "ascendc" / "build" / "demo"
    out = a.output or (ed / "prof_results" / time.strftime("msprof_op_%Y%m%d_%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in SHAPES:
        for b in backends:
            try:
                rows.append(profile_one(ed, out, bench, demo, ot, b, s))
            except Exception as e:
                print(f"[SKIP] {b} {s}: {e}", flush=True)
    write_summary(out / "summary.csv", rows)
    write_detail(out / "pipeline_detail.csv", rows)
    print(f"[DONE] {out / 'summary.csv'}")
    print(f"[DONE] {out / 'pipeline_detail.csv'}")


if __name__ == "__main__":
    main()
