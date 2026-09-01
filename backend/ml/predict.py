"""
Loads the trained joblib models (one pair for LLM: decode + prefill, one for
vision) and turns (analyzed repo info + resolved hardware + workload presets)
into predictions.

Every LabelEncoder.transform() call is wrapped with a safe fallback: if a
category value at prediction time wasn't seen during training (a new
framework string, a device_type not in the training data, etc.), it's mapped
to the most common training-time value for that column instead of raising --
this WILL happen once real users hit hardware/frameworks outside the training
grid, and a 500 error there is worse than a slightly-off estimate with a flag.
"""

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).parent / "models"

# ---------------------------------------------------------------------------
# Model load time -- how long it takes to read the weights from disk into
# RAM/VRAM the FIRST time, before any inference happens. This is NOT an ML
# prediction: it's dominated by disk I/O speed and model file size, so a
# simple formula is the right tool (see the earlier design discussion).
#
# psutil doesn't expose actual sequential disk read speed (only free-space),
# so this uses a fixed assumption rather than a measurement. SATA SSD
# (~500 MB/s) is used as a conservative middle-of-the-road default -- real
# NVMe drives are often 2000-3500 MB/s (much faster load), and old HDDs are
# ~100-150 MB/s (much slower). This is flagged as an assumption in the
# result, not presented as a measured fact.
ASSUMED_DISK_READ_MBPS = 500.0


def estimate_load_time_sec(total_size_gb: float, disk_read_mbps: float = ASSUMED_DISK_READ_MBPS) -> dict:
    seconds = (total_size_gb * 1024) / disk_read_mbps
    return {
        "seconds": round(seconds, 2),
        "assumed_disk_read_mbps": disk_read_mbps,
        "desc": "One-time cost of reading the model's weights off disk into memory, before it can respond to anything. Only happens once per session (or once per app restart), not per request.",
        "note": (
            "Formula-based estimate (model size / assumed disk read speed), not "
            "measured on this machine. Assumes a SATA SSD (~500 MB/s); an NVMe "
            "drive will load significantly faster, an HDD significantly slower."
        ),
    }

# ---------------------------------------------------------------------------
# Load once at import time (these are read-only after training, safe to share
# across requests)
# ---------------------------------------------------------------------------
_decode_model = joblib.load(MODEL_DIR / "decode_model.joblib")
_prefill_model = joblib.load(MODEL_DIR / "prefill_model.joblib")
_llm_encoders = joblib.load(MODEL_DIR / "llm_categorical_encoders.joblib")
_llm_feature_cols = joblib.load(MODEL_DIR / "llm_feature_columns.joblib")

_vision_model = joblib.load(MODEL_DIR / "vision_latency_model.joblib")
_vision_encoders = joblib.load(MODEL_DIR / "vision_categorical_encoders.joblib")
_vision_feature_cols = joblib.load(MODEL_DIR / "vision_feature_columns.joblib")


def _safe_encode(encoder, value):
    """LabelEncoder.transform raises on unseen labels. Fall back to the most
    frequent class seen during training (index 0 of classes_ is arbitrary
    alphabetical order, not frequency -- but LabelEncoder doesn't retain
    frequency, so we fall back to classes_[0] as a deterministic, documented
    choice rather than raising a 500 on the user's request)."""
    try:
        return int(encoder.transform([value])[0])
    except ValueError:
        fallback = encoder.classes_[0]
        return int(encoder.transform([fallback])[0]), fallback
    except Exception:
        return 0, None


def _encode_row(row: dict, encoders: dict, feature_cols: list):
    """Builds a single-row DataFrame in the exact column order/encoding the
    model was trained on. Returns (dataframe, warnings) where warnings lists
    any category that had to fall back to a default because it wasn't seen
    during training."""
    warnings = []
    encoded = dict(row)
    for col, encoder in encoders.items():
        if col not in encoded:
            continue
        result = _safe_encode(encoder, encoded[col])
        if isinstance(result, tuple):
            encoded_val, fallback_used = result
            warnings.append(
                f"'{encoded[col]}' not seen during training for '{col}' -- "
                f"used fallback '{fallback_used}' instead"
            )
            encoded[col] = encoded_val
        else:
            encoded[col] = result

    missing = [c for c in feature_cols if c not in encoded]
    if missing:
        raise ValueError(f"Missing required feature(s) for prediction: {missing}")

    df = pd.DataFrame([{c: encoded[c] for c in feature_cols}])
    return df, warnings


# ---------------------------------------------------------------------------
# LLM prediction
# ---------------------------------------------------------------------------
# Roofline formulas -- MUST mirror build_dataset.py exactly, since the trained
# model was given the roofline prediction as an input feature (formula-as-
# feature design) and expects the same computation at inference time.
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

# Workload presets shown to the user without asking them to specify tokens
# (see the earlier design discussion -- token counts aren't something a
# typical user can supply, so we show 3 canonical profiles instead)
WORKLOAD_PRESETS = {
    "short_qa": {"label": "Short Q&A", "input_tokens": 128, "output_tokens": 256},
    "summarization": {"label": "Summarization", "input_tokens": 2000, "output_tokens": 200},
    "long_form": {"label": "Long-form generation", "input_tokens": 200, "output_tokens": 1000},
}


def _kv_cache_gb(param_count_b, context_len):
    return 0.0000164 * param_count_b * context_len


def _hw_class_for_llm(hardware_row: dict) -> str:
    """Maps a hardware_specs_lookup.csv row (device-agnostic categories) onto
    the coarser hw_class buckets the LLM model was trained on. Best-effort --
    exact mapping table would be large, this covers the common cases."""
    category = hardware_row.get("category", "")
    vendor = hardware_row.get("vendor", "")
    device_type = hardware_row.get("device_type", "")
    tflops = float(hardware_row.get("compute_fp16_tflops") or hardware_row.get("compute_fp32_tflops") or 0)

    if device_type == "cpu":
        return "cpu_modern" if tflops > 1.0 else "cpu_older"
    if vendor == "Apple":
        if tflops < 5:
            return "apple_base"
        elif tflops < 10:
            return "apple_pro"
        elif tflops < 20:
            return "apple_max"
        return "apple_ultra"
    if category == "datacenter":
        return "datacenter"
    if tflops < 20:
        return "consumer_low"
    elif tflops < 40:
        return "consumer_mid"
    elif tflops < 60:
        return "consumer_high"
    return "consumer_flagship"


def predict_llm(model_info: dict, hardware_row: dict, framework: str = "ollama", available_ram_gb: float = None):
    """model_info expected keys (best-effort, with defaults if missing):
        param_count_b, precision, context_length_used, architecture_type
    Returns per-preset totals plus the headline rate metrics.

    available_ram_gb: live free RAM on this machine right now (from
    system_specs.py), NOT the static hardware spec-sheet capacity. Real
    benchmark data showed this matters a lot -- see build_dataset.py's
    module docstring for the specific example (a stronger machine with low
    free RAM measured 10-100x slower than a weaker machine with headroom).
    Falls back to the static hardware_row capacity if not supplied."""
    param_count_b = model_info.get("param_count_b") or 1.0
    precision = model_info.get("precision") or "Q4_K_M"
    if precision not in PRECISION_BYTES:
        precision = "fp16"  # unknown precision string -> safe default
    architecture_type = model_info.get("architecture_type") or "dense"
    context_length_used = model_info.get("context_length_used") or 4096

    tflops = float(hardware_row.get("compute_fp16_tflops") or hardware_row.get("compute_fp32_tflops") or 1.0)
    bandwidth = float(hardware_row.get("memory_bandwidth_gbs") or 20.0)
    vram = float(hardware_row.get("memory_gb") or 8.0)
    device_type = hardware_row.get("device_type", "cpu")
    hw_class = _hw_class_for_llm(hardware_row)

    bytes_per_param = PRECISION_BYTES[precision]
    model_bytes = param_count_b * 1e9 * bytes_per_param
    model_size_gb = model_bytes / 1e9
    kv_bytes = _kv_cache_gb(param_count_b, context_length_used) * 1e9
    eff = EFFICIENCY.get(hw_class, 0.5)
    roofline_decode_ms = ((model_bytes + kv_bytes) / (bandwidth * 1e9 * eff)) * 1000.0

    prefill_eff = PREFILL_EFFICIENCY.get(hw_class, 0.3)
    flops_per_token = 2 * param_count_b * 1e9
    roofline_prefill_ms = (flops_per_token / (tflops * 1e12 * prefill_eff)) * 1000.0

    ram_gb = available_ram_gb if available_ram_gb is not None else vram
    ram_pressure_ratio = min(ram_gb / max(model_size_gb, 0.01), 10.0)

    row = dict(
        param_count_b=param_count_b,
        architecture_type=architecture_type,
        precision=precision,
        model_size_gb=model_size_gb,
        input_tokens=128,  # placeholder row values for the fixed hw/model features;
        output_tokens=256,  # actual per-preset totals are computed via formula below,
        context_length_used=context_length_used,  # not re-predicted per preset
        batch_size=1,
        kv_cache_gb=kv_bytes / 1e9,
        device_type=device_type,
        compute_tflops=tflops,
        memory_bandwidth_gbs=bandwidth,
        vram_or_ram_gb=vram,
        hw_class=hw_class,
        framework=framework,
        roofline_prefill_ms_per_token=roofline_prefill_ms,
        roofline_decode_ms_per_token=roofline_decode_ms,
        ram_pressure_ratio=ram_pressure_ratio,
    )

    warnings = []
    if ram_pressure_ratio < 1.5:
        warnings.append(
            f"Free RAM is close to this model's size ({ram_gb:.1f}GB free vs "
            f"~{model_size_gb:.1f}GB model) -- when a model's memory footprint "
            f"approaches available RAM, paging/swapping can slow things down "
            f"substantially. Close other applications for a more realistic result."
        )
    if hw_class == "consumer_low" and device_type == "gpu":
        warnings.append(
            "Real lab benchmarks on a similar laptop-GPU configuration showed "
            "large, model-to-model variance not explained by size or hardware "
            "alone (a smaller model ran ~350x slower than a larger one in the "
            "same session) -- likely inconsistent GPU-vs-CPU placement by the "
            "inference backend for certain model formats. Treat this prediction "
            "as a rough midpoint, not a tight estimate, on this hardware class."
        )

    ml_decode_ms = ml_prefill_ms = None
    try:
        df, w = _encode_row(row, _llm_encoders, _llm_feature_cols)
        warnings.extend(w)
        ml_decode_ms = float(_decode_model.predict(df)[0])
        ml_prefill_ms = float(_prefill_model.predict(df)[0])
        decode_ms, prefill_ms = ml_decode_ms, ml_prefill_ms
    except Exception as e:
        # model-level failure -- fall back to the pure roofline formula rather
        # than failing the whole request
        decode_ms, prefill_ms = roofline_decode_ms, roofline_prefill_ms
        warnings.append(f"ML model prediction failed ({e}); used roofline formula only as fallback")

    # SANITY GUARD: the trained model was fit on a specific range of inputs
    # (see the dataset README) and, like any tree model, can produce wildly
    # wrong values on out-of-distribution inputs -- most dangerously a
    # near-zero prediction that would silently claim impossible speeds (e.g.
    # 100,000 tok/s on a laptop CPU). If the ML prediction diverges from the
    # physics-based roofline baseline by more than 5x in either direction,
    # trust the roofline instead and say so, rather than show a bad number.
    DIVERGENCE_FACTOR = 5.0
    for label, ml_val, roofline_val in [
        ("decode", decode_ms, roofline_decode_ms),
        ("prefill", prefill_ms, roofline_prefill_ms),
    ]:
        if ml_val is None or roofline_val <= 0:
            continue
        ratio = ml_val / roofline_val if roofline_val else float("inf")
        if ratio < 1 / DIVERGENCE_FACTOR or ratio > DIVERGENCE_FACTOR:
            warnings.append(
                f"ML {label} prediction ({ml_val:.3f}ms) diverged more than "
                f"{DIVERGENCE_FACTOR:.0f}x from the physics-based estimate "
                f"({roofline_val:.3f}ms) -- used the formula-only estimate "
                f"instead for this value, since the input is likely outside "
                f"what the model was trained on."
            )
            if label == "decode":
                decode_ms = roofline_decode_ms
            else:
                prefill_ms = roofline_prefill_ms

    decode_ms = max(decode_ms, 0.01)
    prefill_ms = max(prefill_ms, 0.01)
    tokens_per_sec = 1000.0 / decode_ms

    presets = {}
    for key, preset in WORKLOAD_PRESETS.items():
        ttft_ms = prefill_ms * preset["input_tokens"]
        total_ms = ttft_ms + decode_ms * preset["output_tokens"]
        presets[key] = {
            "label": preset["label"],
            "input_tokens": preset["input_tokens"],
            "output_tokens": preset["output_tokens"],
            "ttft_ms": round(ttft_ms, 1),
            "total_latency_ms": round(total_ms, 1),
            "total_latency_sec": round(total_ms / 1000.0, 2),
            "desc": (
                f"Time to get a complete response for a {preset['label'].lower()} "
                f"workload (~{preset['input_tokens']} words in, ~{preset['output_tokens']} "
                f"words out) -- includes both the wait for the first word and the "
                f"time to generate the rest."
            ),
        }

    load_time = estimate_load_time_sec(model_bytes / 1e9)

    return {
        "model_class": "llm",
        "decode_ms_per_token": round(decode_ms, 3),
        "decode_desc": "How long each word takes to generate once the model has started responding -- lower is faster.",
        "prefill_ms_per_token": round(prefill_ms, 3),
        "prefill_desc": "How long it takes to read each word of your prompt before the model starts replying -- this adds to the wait before you see anything.",
        "tokens_per_sec": round(tokens_per_sec, 1),
        "tokens_per_sec_desc": "Generation speed once the model is warmed up and replying -- higher is faster.",
        "presets": presets,
        "load_time_estimate_sec": load_time,
        "ram_pressure_ratio": round(ram_pressure_ratio, 2),
        "warnings": warnings,
        "_debug": {
            "roofline_decode_ms_per_token": round(roofline_decode_ms, 4),
            "roofline_prefill_ms_per_token": round(roofline_prefill_ms, 4),
            "raw_ml_decode_ms_per_token": round(ml_decode_ms, 4) if ml_decode_ms is not None else None,
            "raw_ml_prefill_ms_per_token": round(ml_prefill_ms, 4) if ml_prefill_ms is not None else None,
            "feature_row": row,
        },
    }


# ---------------------------------------------------------------------------
# Vision prediction
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
_TASK_DEFAULT_RESOLUTION = {
    "classification": 224, "detection": 640, "segmentation": 512,
}
# Rough params -> FLOPs ratio by task, used ONLY when model_flops_g isn't
# available from the model card/config (HF rarely publishes FLOPs directly).
# Calibrated loosely against the MODEL_GRID reference points used to build
# the vision dataset -- this is the weakest part of the vision pipeline and
# should be replaced with a real FLOPs computation (e.g. via `fvcore`/
# `torchinfo` if the model can be loaded) when you extend this further.
_FLOPS_PER_PARAM_M = {"classification": 0.35, "detection": 3.2, "segmentation": 2.6}


def _hw_class_for_vision(hardware_row: dict) -> str:
    device_type = hardware_row.get("device_type", "")
    category = hardware_row.get("category", "")
    tflops = float(hardware_row.get("compute_fp16_tflops") or hardware_row.get("compute_fp32_tflops") or 0)
    if device_type == "cpu":
        return "cpu_modern_mt" if tflops > 0.5 else "cpu_1thread"
    if category == "datacenter":
        return "datacenter_low" if tflops < 200 else "datacenter"
    if tflops < 20:
        return "consumer_low"
    elif tflops < 40:
        return "consumer_mid"
    elif tflops < 60:
        return "consumer_high"
    return "consumer_flagship"


def _default_framework_for_device(device_type: str) -> str:
    return {"gpu": "pytorch", "cpu": "onnxruntime", "apple": "coreml"}.get(device_type, "pytorch")


def predict_vision(model_info: dict, hardware_row: dict, framework: str = None, available_ram_gb: float = None):
    param_count_m = model_info.get("param_count_m") or 25.0
    task_type = model_info.get("task_type") or "classification"
    architecture_family = model_info.get("architecture_family") or "cnn"
    precision = model_info.get("precision") or "fp32"
    resolution = model_info.get("input_resolution") or _TASK_DEFAULT_RESOLUTION.get(task_type, 224)
    flops_g = model_info.get("model_flops_g")
    flops_estimated = flops_g is None
    if flops_g is None:
        flops_g = param_count_m * _FLOPS_PER_PARAM_M.get(task_type, 0.5)

    device_type = hardware_row.get("device_type", "cpu")
    tflops = float(hardware_row.get("compute_fp16_tflops") or hardware_row.get("compute_fp32_tflops") or 1.0)
    bandwidth = float(hardware_row.get("memory_bandwidth_gbs") or 20.0)
    vram = float(hardware_row.get("memory_gb") or 8.0)
    hw_class = _hw_class_for_vision(hardware_row)
    fw = framework or _default_framework_for_device(device_type)

    overhead_ms, k = LINEAR_PARAMS.get((hw_class, fw), (3.0, 5.0))
    roofline_ms = overhead_ms + k * flops_g

    model_size_gb_for_ratio = (param_count_m * 1e6 * {"fp32": 4, "fp16": 2, "int8": 1}.get(precision, 4)) / 1e9
    ram_gb = available_ram_gb if available_ram_gb is not None else vram
    ram_pressure_ratio = min(ram_gb / max(model_size_gb_for_ratio, 0.01), 10.0)

    row = dict(
        param_count_m=param_count_m,
        architecture_family=architecture_family,
        task_type=task_type,
        model_flops_g=flops_g,
        input_resolution=resolution,
        precision=precision,
        batch_size=1,
        device_type=device_type,
        compute_tflops=tflops,
        memory_bandwidth_gbs=bandwidth,
        vram_or_ram_gb=vram,
        hw_class=hw_class,
        framework=fw,
        roofline_ms_per_image=roofline_ms,
        ram_pressure_ratio=ram_pressure_ratio,
    )

    warnings = []
    if flops_estimated:
        warnings.append(
            "model_flops_g not available from the model card -- estimated from "
            "param_count_m using a rough per-task ratio. Treat this prediction "
            "as lower-confidence than one with a real FLOPs figure."
        )
    if ram_pressure_ratio < 1.5:
        warnings.append(
            f"Free RAM is close to this model's size ({ram_gb:.1f}GB free vs "
            f"~{model_size_gb_for_ratio:.1f}GB model) -- when a model's memory "
            f"footprint approaches available RAM, paging/swapping can slow "
            f"things down substantially. Close other applications for a more "
            f"realistic result."
        )

    ml_ms_per_image = None
    try:
        df, w = _encode_row(row, _vision_encoders, _vision_feature_cols)
        warnings.extend(w)
        ml_ms_per_image = float(_vision_model.predict(df)[0])
        ms_per_image = ml_ms_per_image
    except Exception as e:
        ms_per_image = roofline_ms
        warnings.append(f"ML model prediction failed ({e}); used fitted formula only as fallback")

    # Same sanity guard as the LLM path -- see predict_llm for the full
    # rationale. Vision's fitted-formula baseline is itself lower-confidence
    # than the LLM roofline (see the vision dataset README), so this is a
    # coarser safety net, not a precise check.
    DIVERGENCE_FACTOR = 8.0
    if ml_ms_per_image is not None and roofline_ms > 0:
        ratio = ml_ms_per_image / roofline_ms
        if ratio < 1 / DIVERGENCE_FACTOR or ratio > DIVERGENCE_FACTOR:
            warnings.append(
                f"ML prediction ({ml_ms_per_image:.3f}ms) diverged more than "
                f"{DIVERGENCE_FACTOR:.0f}x from the fitted-formula estimate "
                f"({roofline_ms:.3f}ms) -- used the formula-only estimate "
                f"instead, since the input is likely outside what the model "
                f"was trained on."
            )
            ms_per_image = roofline_ms

    ms_per_image = max(ms_per_image, 0.01)
    fps = 1000.0 / ms_per_image

    model_size_gb = (param_count_m * 1e6 * {"fp32": 4, "fp16": 2, "int8": 1}.get(precision, 4)) / 1e9
    load_time = estimate_load_time_sec(model_size_gb)

    return {
        "model_class": "vision",
        "ms_per_image": round(ms_per_image, 3),
        "ms_per_image_desc": "How long it takes to process one image/frame through the model, start to finish.",
        "fps": round(fps, 2),
        "fps_desc": "Frames per second this hardware can sustain -- higher is faster, useful for judging real-time feasibility.",
        "load_time_estimate_sec": load_time,
        "ram_pressure_ratio": round(ram_pressure_ratio, 2),
        "warnings": warnings,
        "_debug": {
            "roofline_ms_per_image": round(roofline_ms, 4),
            "raw_ml_ms_per_image": round(ml_ms_per_image, 4) if ml_ms_per_image is not None else None,
            "feature_row": row,
        },
    }
