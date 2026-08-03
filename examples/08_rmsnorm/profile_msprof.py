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


SHAPES = [
    (2, 64, 256),
    (2, 64, 512),
    (2, 128, 512),
    (2, 256, 512),
    (2, 512, 512),
    (2, 1024, 512),
    (2, 1024, 1024),
    (2, 2048, 512),
    (2, 2048, 1024),
]
BACKENDS = ("pyasc", "ascendc", "torch_npu")
WARMUP = 5
ITERS = 10


def run(cmd, cwd):
    print("[RUN]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)


def latest(pattern):
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(pattern)
    return max(matches, key=os.path.getmtime)


def to_float(value):
    if value in (None, "", "NA", "N/A"):
        return 0.0
    return float(str(value).strip().replace("\t", ""))


def read_basic(prof_dir, expected_name):
    path = Path(latest(str(prof_dir / "OPPROF_*" / "OpBasicInfo.csv")))
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"empty OpBasicInfo.csv: {path}")
    matched = [r for r in rows if r.get("Op Name", "").startswith(expected_name)]
    if not matched:
        raise RuntimeError(f"expected {expected_name}*, got {[r.get('Op Name') for r in rows]} in {path}")
    return matched, path


def read_pipe(prof_dir):
    path = Path(latest(str(prof_dir / "OPPROF_*" / "PipeUtilization.csv")))
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows, path


def max_field(rows, key):
    vals = [to_float(row.get(key)) for row in rows]
    return max(vals) if vals else 0.0


def profile_one(example_dir, out_root, bench_script, ascendc_demo, op_types, backend, shape):
    b, s, h = shape
    out = out_root / f"{backend}_{b}x{s}x{h}"
    shutil.rmtree(out, ignore_errors=True)
    if backend == "pyasc":
        cmd = [
            "msprof", "op", f"--output={out}", "python3", str(bench_script),
            "--backend", "pyasc", "--batch", str(b), "--seq", str(s),
            "--hidden", str(h), "--warmup", str(WARMUP), "--iters", str(ITERS),
            "-r", "NPU", "-v", "Ascend910B4",
        ]
    elif backend == "ascendc":
        cmd = [
            "msprof", "op", f"--output={out}", str(ascendc_demo),
            "--batch", str(b), "--seq", str(s), "--hidden", str(h),
            "--warmup", str(WARMUP), "--iters", str(ITERS),
        ]
    else:
        cmd = [
            "msprof", "op", f"--output={out}", "python3", str(bench_script),
            "--backend", "torch_npu",
            "--batch", str(b), "--seq", str(s),
            "--hidden", str(h), "--warmup", str(WARMUP), "--iters", str(ITERS),
        ]
    run(cmd, example_dir)
    all_rows, basic_path = read_basic(out, op_types[backend])
    durations = [to_float(r.get("Task Duration(us)")) for r in all_rows]
    # Skip warmup launches if msprof outputs per-launch rows;
    # if msprof aggregates, use all rows.
    timed = durations[WARMUP:] if len(durations) > WARMUP else durations
    task_us = sum(timed) / len(timed)
    pipe_rows, pipe_path = read_pipe(out)
    row = {
        "backend": backend,
        "shape": shape,
        "elements": shape[0] * shape[1] * shape[2],
        "task_avg_us": task_us,
        "task_min_us": task_us,
        "task_max_us": task_us,
        "task_std_us": 0.0,
        "block_dim": int(to_float(all_rows[0].get("Block Dim"))),
        "vec_avg_us": max_field(pipe_rows, "aiv_vec_time(us)"),
        "scalar_avg_us": max_field(pipe_rows, "aiv_scalar_time(us)"),
        "mte2_avg_us": max_field(pipe_rows, "aiv_mte2_time(us)"),
        "mte3_avg_us": max_field(pipe_rows, "aiv_mte3_time(us)"),
        "used_rows": 1,
        "matched_rows": 1,
        "source_csv": str(basic_path),
        "pipe_source_csv": str(pipe_path),
    }
    print(f"[PARSE] {backend} {shape}: {row['task_avg_us']:.3f} us")
    return row


def write_summary(path, rows):
    by_key = {(row["backend"], row["shape"]): row for row in rows}
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "shape", "elements", "pyasc_us", "ascendc_us", "torch_npu_us",
            "pyasc_over_ascendc", "pyasc_over_torch_npu", "pyasc_source", "ascendc_source", "torch_npu_source",
        ])
        for shape in SHAPES:
            p = by_key.get(("pyasc", shape))
            a = by_key.get(("ascendc", shape))
            t = by_key.get(("torch_npu", shape))
            writer.writerow([
                f"({shape[0]},{shape[1]},{shape[2]})", p["elements"] if p else 0,
                f"{(p['task_avg_us'] if p else 0):.6f}",
                f"{(a['task_avg_us'] if a else 0):.6f}",
                f"{(t['task_avg_us'] if t else 0):.6f}",
                f"{((p['task_avg_us'] / a['task_avg_us']) if p and a else 0):.6f}",
                f"{((p['task_avg_us'] / t['task_avg_us']) if p and t else 0):.6f}",
                p["source_csv"] if p else "N/A", a["source_csv"] if a else "N/A", t["source_csv"] if t else "N/A",
            ])


def format_detail_row(row):
    out = dict(row)
    shape = row["shape"]
    out["shape"] = f"({shape[0]},{shape[1]},{shape[2]})"
    for key, value in list(out.items()):
        if isinstance(value, float):
            out[key] = f"{value:.6f}"
    return out


def write_detail(path, rows):
    fields = [
        "backend", "shape", "elements", "task_avg_us", "task_min_us",
        "task_max_us", "task_std_us", "block_dim", "vec_avg_us",
        "scalar_avg_us", "mte2_avg_us", "mte3_avg_us", "used_rows",
        "matched_rows", "source_csv", "pipe_source_csv",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fields)
        writer.writeheader()
        for row in rows:
            out = format_detail_row(row)
            writer.writerow({field: out[field] for field in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    example_dir = Path(__file__).resolve().parent
    bench_script = example_dir / "bench_rmsnorm.py"
    op_types = {"pyasc": "rmsnorm_kernel", "ascendc": "rms_norm_kernel", "torch_npu": "RmsNorm"}
    ascendc_demo = example_dir / "ascendc" / "build" / "demo"
    out_dir = args.output or (example_dir / "prof_results" / time.strftime("msprof_op_%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for shape in SHAPES:
        for backend in BACKENDS:
            try:
                rows.append(profile_one(example_dir, out_dir, bench_script, ascendc_demo, op_types, backend, shape))
            except Exception as e:
                print(f"[SKIP] {backend} {shape}: {e}", flush=True)

    write_summary(out_dir / "summary.csv", rows)
    write_detail(out_dir / "pipeline_detail.csv", rows)
    print(f"[DONE] {out_dir / 'summary.csv'}")
    print(f"[DONE] {out_dir / 'pipeline_detail.csv'}")


if __name__ == "__main__":
    main()
