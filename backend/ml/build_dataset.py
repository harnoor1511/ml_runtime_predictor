"""
Rebuilds the training data and retrains the LLM (decode/prefill) and vision
latency models, blending:
  1. A synthetic grid generated from the SAME roofline physics formulas used
     at inference time in predict.py (copied here verbatim -- see the
     "MUST mirror" comments in predict.py). This gives broad coverage across
     hardware classes / model sizes / precisions that we have no real
     measurements for.
  2. Real measurements from data/real_benchmarks.csv (see
     parse_real_benchmarks.py), upweighted so the model actually listens to
     them instead of being drowned out by the much larger synthetic grid.

NEW FEATURE this run adds: `ram_pressure_ratio` = available_ram_gb / model_size_gb.
This wasn't in the original feature set, and IS a real, generally-useful signal
for large models where the model genuinely approaches available memory. It is
included here as a legitimate feature to have for that case.

HOWEVER -- an important honest correction: it does NOT explain the strangest
pattern in this specific benchmark data, and this file's docstring originally
claimed it did before a closer look. On DESKTOP-OHTB214, qwen2.5:0.5b ran at
1.57 tok/s while qwen2.5:1.5b (a LARGER model, same family, same run) ran at
555 tok/s -- a 350x difference that has nothing to do with RAM headroom (both
models are tiny relative to the 7GB free). tinyllama:1.1b, gemma3:1b, and
phi3:mini were all similarly slow; qwen2.5:1.5b, llama3.2:1b, and llama3.2:3b
were all fast -- with no clean split by parameter count, pull time, or any
other field captured in the benchmark JSON. The most likely explanation is
inconsistent per-model GPU-vs-CPU placement by Ollama (some model formats
offload to the RTX 4060 cleanly, others silently fall back to CPU) -- but the
benchmark script didn't capture `ollama ps` per-run to confirm this, so it's
a plausible hypothesis, not a verified one. This is flagged here rather than
silently modeled, since presenting a wrong causal story as fact would be
worse than admitting the real driver is unknown from the available data. If
you re-run the benchmark script with `ollama ps` captured per-model, that
would settle it and could be added as a proper `gpu_offloaded` feature.


Usage:
    python build_dataset.py
(reads data/real_benchmarks.csv, data/hardware_specs_lookup.csv;
 writes data/llm_training_data.csv, data/vision_training_data.csv,
 and overwrites models/*.joblib)
"""

import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Roofline formulas -- copied verbatim from predict.py. If you change one,
# change both, or inference and training will silently drift apart.
# ---------------------------------------------------------------------------
PRECISION_BYTES = {
    "fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0,
    "Q8_0": 1.06, "Q6_K": 0.82, "Q5_K_M": 0.73, "Q4_K_M": 0.63,
    "Q4_0": 0.59, "Q3_K_M": 0.49,
}
EFFICIENCY = {
    "consumer_low": 0.55, "consumer_mid": 0.65, "consumer_high": 0.68,
    "consumer_flagship": 0.70, "datacenter": 0.75, "datacenter_low": 0.60,
    "apple_base": 0.65, "apple_pro": 0.70, "apple_max": 0.75, "apple_ultra": 0.80,
    "cpu_modern": 0.75, "cpu_older": 0.65, "cpu_low": 0.55, "cpu_apple": 0.70,
}
PREFILL_EFFICIENCY = {k: max(v - 0.25, 0.20) for k, v in EFFICIENCY.items()}

HW_GRID = [
    # (hw_class, device_type, tflops, bandwidth_gbs, vram_gb)
    ("cpu_older", "cpu", 0.4, 45, 16),
    ("cpu_modern", "cpu", 1.0, 76.8, 16),
    ("cpu_low", "cpu", 0.15, 38.4, 8),
    ("consumer_low", "gpu", 9, 200, 6),
    ("consumer_mid", "gpu", 25, 350, 8),
    ("consumer_high", "gpu", 45, 500, 12),
    ("consumer_flagship", "gpu", 65, 700, 24),
    ("datacenter_low", "gpu", 150, 900, 40),
    ("datacenter", "gpu", 700, 2500, 80),
    ("apple_base", "apple", 3, 68, 16),
    ("apple_pro", "apple", 8, 150, 32),
    ("apple_max", "apple", 15, 400, 64),
    ("apple_ultra", "apple", 27, 800, 128),
]
PARAM_COUNTS_B = [0.5, 1, 1.5, 2, 3, 3.8, 7, 8, 13, 30, 70]
PRECISIONS = list(PRECISION_BYTES.keys())
CONTEXT_LENGTHS = [512, 2048, 4096, 8192, 16384]
FRAMEWORKS = ["ollama", "llama.cpp", "transformers", "vllm", "mlx"]


def _kv_cache_gb(param_count_b, context_len):
    return 0.0000164 * param_count_b * context_len


def roofline_decode_ms(param_count_b, precision, hw_class, bandwidth, context_len):
    bytes_per_param = PRECISION_BYTES[precision]
    model_bytes = param_count_b * 1e9 * bytes_per_param
    kv_bytes = _kv_cache_gb(param_count_b, context_len) * 1e9
    eff = EFFICIENCY.get(hw_class, 0.5)
    return ((model_bytes + kv_bytes) / (bandwidth * 1e9 * eff)) * 1000.0


def roofline_prefill_ms(param_count_b, hw_class, tflops):
    prefill_eff = PREFILL_EFFICIENCY.get(hw_class, 0.3)
    flops_per_token = 2 * param_count_b * 1e9
    return (flops_per_token / (tflops * 1e12 * prefill_eff)) * 1000.0


def build_synthetic_llm_rows(n_samples=2600):
    rows = []
    for _ in range(n_samples):
        hw_class, device_type, tflops, bandwidth, vram = random.choice(HW_GRID)
        param_b = random.choice(PARAM_COUNTS_B)
        precision = random.choice(PRECISIONS)
        context_len = random.choice(CONTEXT_LENGTHS)
        framework = random.choice(FRAMEWORKS)
        architecture_type = random.choices(["dense", "moe"], weights=[0.85, 0.15])[0]

        # jitter hardware numbers +/-15% so the grid isn't just 13 exact points
        tflops_j = tflops * np.random.uniform(0.85, 1.15)
        bandwidth_j = bandwidth * np.random.uniform(0.85, 1.15)

        decode_roof = roofline_decode_ms(param_b, precision, hw_class, bandwidth_j, context_len)
        prefill_roof = roofline_prefill_ms(param_b, hw_class, tflops_j)

        # Target = roofline * lognormal noise. Real systems deviate from the
        # idealized roofline due to kernel launch overhead, memory
        # fragmentation, thermal throttling, etc. -- represented here as
        # multiplicative noise, not additive, since these effects scale with
        # the base latency rather than being a fixed offset.
        decode_target = decode_roof * np.random.lognormal(mean=0.0, sigma=0.25)
        prefill_target = prefill_roof * np.random.lognormal(mean=0.0, sigma=0.30)

        model_bytes_gb = param_b * PRECISION_BYTES[precision]
        # Assume a comfortably-provisioned synthetic system for the grid
        # (2-4x model size free) -- ram_pressure only becomes interesting/
        # low in the real rows appended below.
        available_ram_gb = model_bytes_gb * np.random.uniform(2.0, 5.0)
        ram_pressure_ratio = min(available_ram_gb / max(model_bytes_gb, 0.01), 10.0)

        rows.append(dict(
            param_count_b=param_b,
            architecture_type=architecture_type,
            precision=precision,
            model_size_gb=model_bytes_gb,
            input_tokens=128,
            output_tokens=256,
            context_length_used=context_len,
            batch_size=1,
            kv_cache_gb=_kv_cache_gb(param_b, context_len),
            device_type=device_type,
            compute_tflops=tflops_j,
            memory_bandwidth_gbs=bandwidth_j,
            vram_or_ram_gb=vram,
            hw_class=hw_class,
            framework=framework,
            roofline_prefill_ms_per_token=prefill_roof,
            roofline_decode_ms_per_token=decode_roof,
            available_ram_gb=available_ram_gb,
            ram_pressure_ratio=ram_pressure_ratio,
            decode_ms_per_token=decode_target,
            prefill_ms_per_token=prefill_target,
            sample_weight=1.0,
            source="synthetic",
        ))
    return rows


def _hw_class_for_llm_row(device_type, vendor, tflops):
    if device_type == "cpu":
        return "cpu_modern" if tflops > 1.0 else "cpu_older"
    if tflops < 20:
        return "consumer_low"
    elif tflops < 40:
        return "consumer_mid"
    return "consumer_high"


def build_real_llm_rows(real_weight=25.0):
    """Loads data/real_benchmarks.csv (text rows only -- decode side).
    Prefill is intentionally NOT populated from real data; see the caveat
    in parse_real_benchmarks.py and DATASET_NOTES.md."""
    path = DATA_DIR / "real_benchmarks.csv"
    if not path.exists():
        return []

    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["category"] != "text":
                continue
            param_b = float(r["param_count_b"])
            precision = r["precision"]
            model_size_gb = float(r["model_size_gb"])
            available_ram_gb = float(r["available_ram_gb"])
            ram_pressure_ratio = min(available_ram_gb / max(model_size_gb, 0.01), 10.0)
            has_gpu = r["has_discrete_gpu"] == "True"
            device_type = "gpu" if has_gpu else "cpu"

            # Use whatever hw_class this exact machine's CPU/GPU maps to,
            # so the trained model directly calibrates predictions for the
            # hardware class these two real machines fall into.
            if has_gpu:
                # RTX 4060 Laptop: 11.6 TFLOPS fp16 (see hardware_specs_lookup.csv)
                tflops, bandwidth, vram = 11.6, 232, 8
            else:
                # i5-12500H / i7-12700H laptop CPUs (see hardware_specs_lookup.csv)
                tflops, bandwidth, vram = 0.9, 76.8, available_ram_gb
            hw_class = _hw_class_for_llm_row(device_type, None, tflops)

            context_len = int(float(r["eval_count_tokens"]))
            decode_roof = roofline_decode_ms(param_b, precision, hw_class, bandwidth, context_len)
            prefill_roof = roofline_prefill_ms(param_b, hw_class, tflops)

            rows.append(dict(
                param_count_b=param_b,
                architecture_type="dense",
                precision=precision,
                model_size_gb=model_size_gb,
                input_tokens=context_len,
                output_tokens=context_len,
                context_length_used=context_len,
                batch_size=1,
                kv_cache_gb=_kv_cache_gb(param_b, context_len),
                device_type=device_type,
                compute_tflops=tflops,
                memory_bandwidth_gbs=bandwidth,
                vram_or_ram_gb=vram,
                hw_class=hw_class,
                framework="ollama",
                roofline_prefill_ms_per_token=prefill_roof,
                roofline_decode_ms_per_token=decode_roof,
                available_ram_gb=available_ram_gb,
                ram_pressure_ratio=ram_pressure_ratio,
                decode_ms_per_token=float(r["decode_ms_per_token"]),
                prefill_ms_per_token=None,  # not trustworthy -- see caveat above
                sample_weight=real_weight,
                source=f"real:{r['machine']}",
            ))
    return rows


LLM_FEATURE_COLS = [
    "param_count_b", "architecture_type", "precision", "model_size_gb",
    "input_tokens", "output_tokens", "context_length_used", "batch_size",
    "kv_cache_gb", "device_type", "compute_tflops", "memory_bandwidth_gbs",
    "vram_or_ram_gb", "hw_class", "framework",
    "roofline_prefill_ms_per_token", "roofline_decode_ms_per_token",
    "ram_pressure_ratio",
]
LLM_CATEGORICAL_COLS = ["architecture_type", "precision", "device_type", "hw_class", "framework"]


def train_llm_models():
    synthetic = build_synthetic_llm_rows()
    real = build_real_llm_rows()
    all_rows = synthetic + real
    df = pd.DataFrame(all_rows)

    df.to_csv(DATA_DIR / "llm_training_data.csv", index=False)

    encoders = {}
    for col in LLM_CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[LLM_FEATURE_COLS]
    sw = df["sample_weight"].values

    # Decode: trained on synthetic + real (real rows carry ~25x weight)
    y_decode = df["decode_ms_per_token"].values
    decode_model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
    )
    decode_model.fit(X, y_decode, sample_weight=sw)

    # Prefill: real rows have no trustworthy prefill target -> train on
    # synthetic-only subset (source == "synthetic")
    synth_mask = df["source"] == "synthetic"
    y_prefill = df.loc[synth_mask, "prefill_ms_per_token"].values
    prefill_model = GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42
    )
    prefill_model.fit(X[synth_mask], y_prefill)

    joblib.dump(decode_model, MODEL_DIR / "decode_model.joblib")
    joblib.dump(prefill_model, MODEL_DIR / "prefill_model.joblib")
    joblib.dump(encoders, MODEL_DIR / "llm_categorical_encoders.joblib")
    joblib.dump(LLM_FEATURE_COLS, MODEL_DIR / "llm_feature_columns.joblib")

    print(f"LLM: trained on {len(synthetic)} synthetic + {len(real)} real rows")
    return len(synthetic), len(real)


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
LINEAR_PARAMS = {
    ("datacenter_low", "tensorrt"): (1.07, 0.211),
    ("cpu_1thread", "onnxruntime"): (3.92, 2.803),
    ("consumer_flagship", "pytorch"): (3.0, 0.3095),
    ("consumer_mid", "pytorch"): (5.0, 4.209),
    ("datacenter_low", "pytorch"): (1.5, 0.8544),
    ("cpu_1thread", "pytorch"): (3.49, 178.46),
    ("cpu_4thread", "pytorch"): (11.51, 45.70),
    ("cpu_mobile", "tflite"): (3.0, 25.70),
}
VISION_HW_GRID = [
    ("cpu_1thread", "cpu", 0.15), ("cpu_4thread", "cpu", 0.6), ("cpu_modern_mt", "cpu", 1.0),
    ("consumer_low", "gpu", 9), ("consumer_mid", "gpu", 25),
    ("consumer_high", "gpu", 45), ("consumer_flagship", "gpu", 65),
    ("datacenter_low", "gpu", 150), ("datacenter", "gpu", 700),
]
TASKS = ["classification", "detection", "segmentation"]
ARCH_FAMILIES = ["cnn", "transformer", "hybrid"]
FLOPS_PER_PARAM_M = {"classification": 0.35, "detection": 3.2, "segmentation": 2.6}
RESOLUTIONS = {"classification": [224, 256, 384], "detection": [416, 640, 832], "segmentation": [512, 768]}
VISION_FRAMEWORKS = ["pytorch", "onnxruntime", "tensorrt", "tflite", "coreml"]

VISION_FEATURE_COLS = [
    "param_count_m", "architecture_family", "task_type", "model_flops_g",
    "input_resolution", "precision", "batch_size", "device_type",
    "compute_tflops", "memory_bandwidth_gbs", "vram_or_ram_gb", "hw_class",
    "framework", "roofline_ms_per_image", "ram_pressure_ratio",
]
VISION_CATEGORICAL_COLS = ["architecture_family", "task_type", "precision", "device_type", "hw_class", "framework"]


def build_synthetic_vision_rows(n_samples=1800):
    rows = []
    for _ in range(n_samples):
        hw_class, device_type, tflops = random.choice(VISION_HW_GRID)
        task = random.choice(TASKS)
        arch = random.choice(ARCH_FAMILIES)
        param_m = random.choice([5, 11, 25, 44, 68, 100, 150])
        resolution = random.choice(RESOLUTIONS[task])
        precision = random.choice(["fp32", "fp16", "int8"])
        framework = random.choice(VISION_FRAMEWORKS)
        flops_g = param_m * FLOPS_PER_PARAM_M[task] * np.random.uniform(0.8, 1.2)

        overhead_ms, k = LINEAR_PARAMS.get((hw_class, framework), (3.0, 5.0))
        roofline_ms = overhead_ms + k * flops_g
        target_ms = roofline_ms * np.random.lognormal(mean=0.0, sigma=0.30)

        model_size_gb = (param_m * 1e6 * {"fp32": 4, "fp16": 2, "int8": 1}[precision]) / 1e9
        available_ram_gb = model_size_gb * np.random.uniform(2.0, 5.0)
        ram_pressure_ratio = min(available_ram_gb / max(model_size_gb, 0.01), 10.0)

        rows.append(dict(
            param_count_m=param_m, architecture_family=arch, task_type=task,
            model_flops_g=flops_g, input_resolution=resolution, precision=precision,
            batch_size=1, device_type=device_type, compute_tflops=tflops,
            memory_bandwidth_gbs=300, vram_or_ram_gb=8, hw_class=hw_class,
            framework=framework, roofline_ms_per_image=roofline_ms,
            ram_pressure_ratio=ram_pressure_ratio,
            ms_per_image=target_ms, sample_weight=1.0, source="synthetic",
        ))
    return rows


def build_real_vision_rows(real_weight=25.0):
    path = DATA_DIR / "real_benchmarks.csv"
    if not path.exists():
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["category"] != "vision":
                continue
            param_b = float(r["param_count_b"])
            param_m = param_b * 1000
            model_size_gb = float(r["model_size_gb"])
            available_ram_gb = float(r["available_ram_gb"])
            ram_pressure_ratio = min(available_ram_gb / max(model_size_gb, 0.01), 10.0)
            has_gpu = r["has_discrete_gpu"] == "True"
            device_type = "gpu" if has_gpu else "cpu"
            hw_class = "consumer_low" if has_gpu else "cpu_4thread"
            tflops = 11.6 if has_gpu else 0.9
            framework = "pytorch"  # Ollama's backend, closest available category
            flops_g = param_m * FLOPS_PER_PARAM_M["classification"]  # VLMs -> nearest available task bucket
            overhead_ms, k = LINEAR_PARAMS.get((hw_class, framework), (3.0, 5.0))
            roofline_ms = overhead_ms + k * flops_g

            # decode_ms_per_token here is really "ms per output token" for a
            # vision-language model, not ms/image -- treated as the closest
            # available proxy for this latency model's target unit.
            rows.append(dict(
                param_count_m=param_m, architecture_family="transformer", task_type="classification",
                model_flops_g=flops_g, input_resolution=336, precision="Q4_K_M",
                batch_size=1, device_type=device_type, compute_tflops=tflops,
                memory_bandwidth_gbs=232 if has_gpu else 76.8, vram_or_ram_gb=8 if has_gpu else available_ram_gb,
                hw_class=hw_class, framework=framework, roofline_ms_per_image=roofline_ms,
                ram_pressure_ratio=ram_pressure_ratio,
                ms_per_image=float(r["decode_ms_per_token"]),
                sample_weight=real_weight, source=f"real:{r['machine']}",
            ))
    return rows


def train_vision_model():
    synthetic = build_synthetic_vision_rows()
    real = build_real_vision_rows()
    all_rows = synthetic + real
    df = pd.DataFrame(all_rows)
    df.to_csv(DATA_DIR / "vision_training_data.csv", index=False)

    encoders = {}
    for col in VISION_CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[VISION_FEATURE_COLS]
    y = df["ms_per_image"].values
    sw = df["sample_weight"].values

    model = GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42)
    model.fit(X, y, sample_weight=sw)

    joblib.dump(model, MODEL_DIR / "vision_latency_model.joblib")
    joblib.dump(encoders, MODEL_DIR / "vision_categorical_encoders.joblib")
    joblib.dump(VISION_FEATURE_COLS, MODEL_DIR / "vision_feature_columns.joblib")

    print(f"Vision: trained on {len(synthetic)} synthetic + {len(real)} real rows")
    return len(synthetic), len(real)


if __name__ == "__main__":
    train_llm_models()
    train_vision_model()
    print("Done. Models + training CSVs written to ml/models/ and ml/data/.")
