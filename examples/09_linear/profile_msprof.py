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
import shlex
import shutil
import subprocess
import time
from pathlib import Path

# Llama projection shapes: x[M,K] @ weight[N,K].T -> output[M,N].
CASES = [
    ("hidden-short", (128, 512, 512)),
    ("qkv-short", (128, 512, 1536)),
    ("ffn-down-short", (128, 1024, 512)),
    ("hidden-medium", (1024, 512, 512)),
    ("qkv-medium", (1024, 512, 1536)),
    ("ffn-down-medium", (1024, 1024, 512)),
    ("hidden-long", (4096, 512, 512)),
    ("qkv-long", (4096, 512, 1536)),
    ("ffn-down-long", (4096, 1024, 512)),
]
ALL_SHAPES = [shape for _, shape in CASES]
SCENARIOS = {shape: scenario for scenario, shape in CASES}
BACKENDS = ("pyasc", "ascendc", "torch_npu")
WARMUP = 5
ITERS = 10


# Run one subprocess command in cwd and raise on failure.
# Inputs: command tokens and the working directory.
# Output: None; subprocess failures are propagated.
def run(cmd, cwd):
    print("[RUN]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, check=True)


# Return the most recently modified path matching pattern.
# Inputs: a recursive glob pattern.
# Output: the newest matching path.
def latest(pattern):
    return max(glob.glob(pattern, recursive=True), key=os.path.getmtime)


# Convert a CSV value to float, returning zero for missing values.
# Inputs: one optional CSV field value.
# Output: the parsed float or 0.0.
def to_float(value):
    if value in (None, "", "NA", "N/A"):
        return 0.0
    return float(str(value).strip().replace("\t", ""))


# Read target-kernel duration and block metadata from OpBasicInfo.csv.
# Inputs: profile directory and accepted operator-name prefix or prefixes.
# Output: matching CSV rows and the source CSV path.
def read_basic(directory, names):
    path = Path(latest(str(directory / "OPPROF_*" / "OpBasicInfo.csv")))
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    prefixes = (names, ) if isinstance(names, str) else tuple(names)
    matches = [row for row in rows if any(row.get("Op Name", "").startswith(name) for name in prefixes)]
    if not matches:
        op_names = [row.get("Op Name") for row in rows]
        raise RuntimeError(f"expected {prefixes}*, got {op_names}")
    return matches, path


# Read pipeline counters from the newest PipeUtilization.csv.
# Inputs: one msprof output directory.
# Output: parsed pipeline rows and the source CSV path.
def read_pipe(directory):
    path = Path(latest(str(directory / "OPPROF_*" / "PipeUtilization.csv")))
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    return rows, path


# Return the largest numeric value of key across parsed rows.
# Inputs: parsed CSV rows and a field name.
# Output: the maximum field value, or 0.0 for no rows.
def max_field(rows, key):
    values = [to_float(row.get(key)) for row in rows]
    return max(values) if values else 0.0


# Build the msprof command for one backend and Linear shape.
# Inputs: output paths, executables, backend, M/K/N shape, and demo options.
# Output: the command token list passed to subprocess.
def build_profile_command(output, bench, demo, backend, shape, extra_args):
    m, k, n = shape
    common = ["--m", str(m), "--k", str(k), "--n", str(n)]
    common += ["--warmup", str(WARMUP), "--iters", str(ITERS)]
    command = ["msprof", "op", f"--output={output}"]
    if backend == "ascendc":
        return command + [str(demo)] + common + extra_args
    command += ["python3", str(bench), "--backend", backend] + common
    if backend == "pyasc":
        command += ["-r", "NPU", "-v", "Ascend910B4"]
    return command


# Profile one backend/shape and return its summary and pipeline metrics.
# Inputs: sample paths, operator names, backend, shape, and demo options.
# Output: one dictionary of duration, block, pipeline, and source metadata.
def profile_one(example_dir, out_dir, bench, demo, op_types, backend, shape, ascendc_extra_args):
    m, k, _ = shape
    output = out_dir / f"{backend}_{shape[0]}x{shape[1]}x{shape[2]}"
    shutil.rmtree(output, ignore_errors=True)
    command = build_profile_command(output, bench, demo, backend, shape, ascendc_extra_args)
    run(command, example_dir)
    basic_rows, basic_path = read_basic(output, op_types[backend])
    durations = [to_float(row.get("Task Duration(us)")) for row in basic_rows]
    measured = durations[WARMUP:] if len(durations) > WARMUP else durations
    pipe_rows, pipe_path = read_pipe(output)
    return {
        "backend": backend,
        "shape": shape,
        "elements": m * k,
        "task_avg_us": sum(measured) / len(measured),
        "block_dim": int(to_float(basic_rows[0].get("Block Dim"))),
        "cube_avg_us": max_field(pipe_rows, "aic_cube_time(us)"),
        "scalar_avg_us": max_field(pipe_rows, "aic_scalar_time(us)"),
        "mte1_avg_us": max_field(pipe_rows, "aic_mte1_time(us)"),
        "mte2_avg_us": max_field(pipe_rows, "aic_mte2_time(us)"),
        "fixpipe_avg_us": max_field(pipe_rows, "aic_fixpipe_time(us)"),
        "source_csv": str(basic_path),
        "pipe_source_csv": str(pipe_path),
    }


# Format one optional duration value for CSV output.
# Inputs: one optional profile-result dictionary.
# Output: a six-decimal duration string or an empty string.
def format_duration(row):
    return f"{row['task_avg_us']:.6f}" if row else ""


# Format a duration ratio, leaving unavailable values empty.
# Inputs: numerator and denominator profile-result dictionaries.
# Output: a six-decimal ratio string or an empty string.
def format_ratio(left, right):
    if not left or not right:
        return ""
    return f"{left['task_avg_us'] / right['task_avg_us']:.6f}"


# Write one summary row per requested Linear shape.
# Inputs: destination path, profile rows, and ordered shapes.
# Output: None; writes summary.csv.
def write_summary(path, rows, shapes):
    by_key = {(row["backend"], row["shape"]): row for row in rows}
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "scenario",
            "x_shape",
            "weight_shape",
            "elements",
            "pyasc_us",
            "ascendc_us",
            "torch_npu_us",
            "pyasc_over_ascendc",
            "pyasc_over_torch_npu",
            "ascendc_over_torch_npu",
        ])
        for shape in shapes:
            pyasc = by_key.get(("pyasc", shape))
            ascendc = by_key.get(("ascendc", shape))
            torch_npu = by_key.get(("torch_npu", shape))
            writer.writerow([
                SCENARIOS[shape],
                f"[{shape[0]},{shape[1]}]",
                f"[{shape[2]},{shape[1]}]",
                shape[0] * shape[1],
                format_duration(pyasc),
                format_duration(ascendc),
                format_duration(torch_npu),
                format_ratio(pyasc, ascendc),
                format_ratio(pyasc, torch_npu),
                format_ratio(ascendc, torch_npu),
            ])


# Convert one profile row into pipeline-detail CSV fields.
# Inputs: one profile dictionary and the ordered output fields.
# Output: a CSV-ready dictionary.
def format_detail_row(row, fields):
    output = dict(row)
    m, k, n = output.pop("shape")
    output["x_shape"] = f"[{m},{k}]"
    output["weight_shape"] = f"[{n},{k}]"
    output["scenario"] = SCENARIOS[(m, k, n)]
    for key, value in output.items():
        if isinstance(value, float):
            output[key] = f"{value:.6f}"
    return {field: output[field] for field in fields}


# Write per-backend pipeline counters to pipeline_detail.csv.
# Inputs: destination path and profile-result rows.
# Output: None; writes pipeline_detail.csv.
def write_detail(path, rows):
    fields = [
        "scenario",
        "backend",
        "x_shape",
        "weight_shape",
        "elements",
        "task_avg_us",
        "block_dim",
        "cube_avg_us",
        "scalar_avg_us",
        "mte1_avg_us",
        "mte2_avg_us",
        "fixpipe_avg_us",
        "source_csv",
        "pipe_source_csv",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_detail_row(row, fields))


# Parse output, backend, and Ascend C demo options.
# Inputs: process command-line arguments.
# Output: parsed arguments and their ArgumentParser.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--ascendc-extra-args", default="",
                        help="Extra command-line arguments passed only to the Ascend C demo.")
    parser.add_argument("--backends", default=",".join(BACKENDS),
                        help="Comma-separated backends to profile: pyasc,ascendc,torch_npu.")
    return parser.parse_args(), parser


# Profile the selected backends and write both CSV files.
# Inputs: command-line output, backend, and demo options.
# Output: None; writes summary.csv and pipeline_detail.csv.
def main():
    args, parser = parse_args()
    backends = tuple(item.strip() for item in args.backends.split(",") if item.strip())
    unknown = set(backends) - set(BACKENDS)
    if unknown:
        parser.error(f"unknown backends: {','.join(sorted(unknown))}")
    example_dir = Path(__file__).resolve().parent
    bench = example_dir / "bench_linear.py"
    demo = example_dir / "ascendc" / "build" / "demo"
    op_types = {
        "pyasc": "linear_kernel",
        "ascendc": (
            "linear_kernel",
            "linear_kernel_direct",
            "_Z20linear_kernel_direct",
            "_Z28linear_kernel",
        ),
        "torch_npu": "MatMulV2",
    }
    timestamp = time.strftime("msprof_op_%Y%m%d_%H%M%S")
    out_dir = args.output or (example_dir / "prof_results" / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    shapes = ALL_SHAPES
    extra_args = shlex.split(args.ascendc_extra_args)
    rows = []
    for shape in shapes:
        for backend in backends:
            try:
                rows.append(profile_one(example_dir, out_dir, bench, demo, op_types, backend, shape, extra_args))
            except Exception as error:
                print(f"[SKIP] {backend} {shape}: {error}", flush=True)
    write_summary(out_dir / "summary.csv", rows, shapes)
    write_detail(out_dir / "pipeline_detail.csv", rows)
    print(f"[DONE] {out_dir / 'summary.csv'}")
    print(f"[DONE] {out_dir / 'pipeline_detail.csv'}")


if __name__ == "__main__":
    main()
