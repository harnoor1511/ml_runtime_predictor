"""
Turns the `info` dict already produced by hf_analyzer.py / github_analyzer.py
into the flat model_info dict predict.py expects. Kept separate from
model_type.py (which only decides LLM vs vision) and predict.py (which only
knows how to run the models) so each has one job.
"""


def _bytes_per_param_from_dtype(dtype):
    return {
        "float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "float64": 8,
    }.get(dtype)


def extract_llm_attributes(info: dict) -> dict:
    param_count_b = None
    if info.get("estimated_param_count"):
        param_count_b = info["estimated_param_count"] / 1e9

    precision = None
    dtype = info.get("torch_dtype")
    if dtype:
        precision = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}.get(dtype, "fp16")

    architecture_type = "dense"
    archs = info.get("architecture")
    arch_str = " ".join(archs).lower() if isinstance(archs, list) else str(archs or "").lower()
    if "moe" in arch_str or "mixtral" in arch_str or "switch" in arch_str:
        architecture_type = "moe"

    context_length_used = info.get("max_position_embeddings")

    return {
        "param_count_b": param_count_b,
        "precision": precision,
        "architecture_type": architecture_type,
        "context_length_used": context_length_used,
        # confidence flags so the API response can tell the frontend which
        # values were real vs defaulted
        "_had_param_count": param_count_b is not None,
        "_had_precision": precision is not None,
        "_had_context_length": context_length_used is not None,
    }


def extract_vision_attributes(info: dict) -> dict:
    param_count_m = None
    if info.get("estimated_param_count"):
        param_count_m = info["estimated_param_count"] / 1e6

    pipeline_tag = (info.get("pipeline_tag") or "").lower()
    task_type = "classification"
    if "detection" in pipeline_tag:
        task_type = "detection"
    elif "segmentation" in pipeline_tag:
        task_type = "segmentation"

    archs = info.get("architecture")
    arch_str = " ".join(archs).lower() if isinstance(archs, list) else str(archs or "").lower()
    architecture_family = "cnn"
    if any(k in arch_str for k in ("vit", "vision_transformer", "swin", "deit", "beit")):
        architecture_family = "transformer"
    elif any(k in arch_str for k in ("repvit", "efficientformer")):
        architecture_family = "hybrid"

    dtype = info.get("torch_dtype")
    precision = {"float32": "fp32", "float16": "fp16"}.get(dtype, "fp32")

    return {
        "param_count_m": param_count_m,
        "task_type": task_type,
        "architecture_family": architecture_family,
        "precision": precision,
        "model_flops_g": None,  # not available from HF/GitHub metadata directly;
                                  # predict_vision() estimates it from param count
        "_had_param_count": param_count_m is not None,
    }
