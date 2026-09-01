"""
Parses raw `ollama-benchmark-*.json` files (produced by the user's own
benchmarking script, run on real hardware) into a clean CSV of real
measurements for use in build_dataset.py.

IMPORTANT DATA CAVEAT, read before trusting these numbers blindly:
In both source JSON files, every run's `prompt_eval_count`/`prompt_eval_duration`
field is numerically IDENTICAL to `eval_count`/`eval_duration`. That's not
plausible for real prefill vs decode measurements (prefill and decode have
very different costs), so this is almost certainly a bug in the benchmarking
script itself (both values likely parsed from the same regex capture in
`ollama run --verbose` output). Practical effect: we cannot trust these files
for a separate prefill (time-to-first-token) measurement -- only the combined
per-token rate is usable. This script therefore only emits DECODE-side rows;
prefill training stays synthetic-only (see build_dataset.py). If the
benchmark script is fixed to capture prompt_eval separately from eval in a
future run, re-run this script and prefill can be added the same way.

Usage:
    python parse_real_benchmarks.py <json1> <json2> ... --out real_benchmarks.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# param counts (billions) for the specific Ollama tags used in the benchmark.
# Sourced from each model's published card; "b" tag suffix is the vendor's
# own rounding, not always exact (e.g. llama3.2:1b is actually ~1.24B).
TEXT_MODEL_PARAMS_B = {
    "qwen2.5:0.5b": 0.49,
    "tinyllama:1.1b": 1.1,
    "gemma3:1b": 1.0,
    "qwen2.5:1.5b": 1.5,
    "llama3.2:1b": 1.24,
    "phi3:mini": 3.8,
    "llama3.2:3b": 3.21,
}
VISION_MODEL_PARAMS_B = {
    "qwen3-vl:2b": 2.0,
    "qwen2.5vl:3b": 3.75,
    "gemma3:4b": 4.3,
}

# All models pulled via plain `ollama pull <tag>` without an explicit
# quantization suffix resolve to Ollama's default GGUF quant, which is
# Q4_K_M for essentially every small instruct model on the Ollama library
# as of 2026. Not verifiable per-tag from this JSON, so treated as an
# assumption applied uniformly -- flagged in DATASET_NOTES.md.
ASSUMED_PRECISION = "Q4_K_M"

DUR_RE = re.compile(r"^(?:(\d+)m)?([\d.]+)s$")


def parse_duration_ms(s):
    if not s:
        return None
    s = s.strip()
    if s.endswith("ms"):
        return float(s[:-2])
    m = DUR_RE.match(s)
    if m:
        minutes = float(m.group(1)) if m.group(1) else 0.0
        seconds = float(m.group(2))
        return (minutes * 60 + seconds) * 1000.0
    if s.endswith("s"):
        return float(s[:-1]) * 1000.0
    return None


def load_benchmark(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def rows_from_file(path, available_ram_gb):
    data = load_benchmark(path)
    hw = data["hardware"]
    machine = hw.get("machine_name", Path(path).stem)
    cpu_name = hw.get("cpu_name", "")
    total_ram_gb = hw.get("total_ram_gb")
    gpus = hw.get("gpus", [])
    gpu_name = gpus[0]["name"] if gpus and len(gpus) == 1 else (
        gpus[-1]["name"] if gpus else None  # prefer the discrete one if >1 listed
    )
    # crude "has a real discrete GPU" check -- Iris Xe iGPUs are listed but
    # aren't meaningfully used by Ollama's default backend on these machines
    has_discrete_gpu = any("Iris" not in g.get("name", "") for g in gpus)

    rows = []
    for section, param_table, category in (
        ("text_runs", TEXT_MODEL_PARAMS_B, "text"),
        ("vision_runs", VISION_MODEL_PARAMS_B, "vision"),
    ):
        for run in data.get(section, []):
            if not run.get("success"):
                continue
            model = run["model"]
            param_b = param_table.get(model)
            if param_b is None:
                continue

            eval_count = run.get("eval_count")
            eval_dur_ms = parse_duration_ms(run.get("eval_duration"))
            if not eval_count or not eval_dur_ms:
                continue
            eval_count = float(eval_count)
            ms_per_token = eval_dur_ms / eval_count

            bytes_per_param = {"Q4_K_M": 0.63}[ASSUMED_PRECISION]
            model_size_gb = param_b * bytes_per_param

            rows.append({
                "machine": machine,
                "cpu_name": cpu_name,
                "gpu_name": gpu_name or "",
                "has_discrete_gpu": has_discrete_gpu,
                "total_ram_gb": total_ram_gb,
                "available_ram_gb": available_ram_gb,
                "category": category,
                "model": model,
                "param_count_b": param_b,
                "precision": ASSUMED_PRECISION,
                "model_size_gb": round(model_size_gb, 4),
                "eval_count_tokens": eval_count,
                "decode_ms_per_token": round(ms_per_token, 4),
                "tokens_per_sec": round(1000.0 / ms_per_token, 2),
                "load_duration_ms": parse_duration_ms(run.get("load_duration")),
                "pull_seconds": run.get("pull_seconds"),
                "wall_clock_seconds": run.get("wall_clock_seconds"),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_files", nargs="+")
    ap.add_argument("--out", default="real_benchmarks.csv")
    ap.add_argument(
        "--available-ram", action="append", default=[],
        help="machine_name:available_ram_gb, e.g. LAPTOP-S5E2D231:4.3. "
             "Repeatable. Falls back to total_ram_gb from the JSON if a "
             "machine isn't given here.",
    )
    args = ap.parse_args()

    ram_overrides = {}
    for entry in args.available_ram:
        name, val = entry.split(":")
        ram_overrides[name] = float(val)

    all_rows = []
    for jf in args.json_files:
        data = load_benchmark(jf)
        machine = data["hardware"].get("machine_name")
        available = ram_overrides.get(machine, data["hardware"].get("total_ram_gb"))
        all_rows.extend(rows_from_file(jf, available))

    if not all_rows:
        print("No usable rows extracted.", file=sys.stderr)
        sys.exit(1)

    fieldnames = list(all_rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print(f"Wrote {len(all_rows)} real benchmark rows to {args.out}")


if __name__ == "__main__":
    main()
