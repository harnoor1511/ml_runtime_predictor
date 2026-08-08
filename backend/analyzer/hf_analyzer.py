"""Extracts structured info from a Hugging Face model repo without downloading/running it."""

from datetime import datetime, timezone
import requests

HF_API = "https://huggingface.co/api/models"


def human_bytes(n):
    if n is None:
        return None
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def safe_get(url, headers=None, params=None, timeout=15):
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        return r if r.status_code == 200 else None
    except requests.RequestException:
        return None


def infer_modality(pipeline_tag, tags):
    tag_str = " ".join([pipeline_tag or ""] + (tags or [])).lower()
    if any(k in tag_str for k in ["image", "vision", "detection", "segmentation"]):
        return "image"
    if any(k in tag_str for k in ["audio", "speech", "asr", "tts"]):
        return "audio"
    if "video" in tag_str:
        return "video"
    if any(k in tag_str for k in ["text", "nlp", "translation", "summarization", "generation", "question-answering"]):
        return "text"
    if any(k in tag_str for k in ["tabular", "regression", "classification"]) and "image" not in tag_str:
        return "tabular"
    return "unknown"


def get_hf_file_size(model_id, filename):
    url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"
    try:
        r = requests.head(url, allow_redirects=True, timeout=15)
        size = r.headers.get("Content-Length") or r.headers.get("x-linked-size")
        return int(size) if size else None
    except requests.RequestException:
        return None


def fetch_hf_json_file(model_id, filename):
    r = safe_get(f"https://huggingface.co/{model_id}/resolve/main/{filename}")
    if not r:
        return None
    try:
        return r.json()
    except ValueError:
        return None


def fetch_hf_raw_file(model_id, filename):
    r = safe_get(f"https://huggingface.co/{model_id}/resolve/main/{filename}")
    return r.text if r else None


def analyze_hf_model(model_id):
    info = {
        "source_type": "huggingface",
        "model_id": model_id,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }

    r = safe_get(f"{HF_API}/{model_id}")
    if not r:
        info["error"] = "Could not fetch model metadata from Hugging Face API. Check the model id."
        return info

    meta = r.json()
    if not isinstance(meta, dict):
        info["error"] = (
            f"Hugging Face API did not return model data for '{model_id}'. "
            "Check that the model id/URL is correct (e.g. owner/model-name)."
        )
        return info

    info["pipeline_tag"] = meta.get("pipeline_tag")
    info["library_name"] = meta.get("library_name")
    info["tags"] = meta.get("tags", [])
    info["downloads"] = meta.get("downloads")
    info["likes"] = meta.get("likes")
    info["last_modified"] = meta.get("lastModified")
    info["gated"] = meta.get("gated", False)

    siblings = meta.get("siblings", [])
    weight_exts = (".bin", ".safetensors", ".pt", ".onnx", ".h5", ".ckpt", ".gguf")
    all_files = [s.get("rfilename", "") for s in siblings]
    weight_files = [f for f in all_files if f.lower().endswith(weight_exts)]

    info["file_list"] = all_files
    info["weight_files"] = weight_files

    total_weight_bytes = 0
    for fname in weight_files:
        size = get_hf_file_size(model_id, fname)
        if size:
            total_weight_bytes += size
    info["total_weight_bytes"] = total_weight_bytes or None
    info["total_weight_size_human"] = human_bytes(total_weight_bytes) if total_weight_bytes else None

    config = fetch_hf_json_file(model_id, "config.json")
    info["config"] = config
    if config:
        info["architecture"] = config.get("architectures") or config.get("model_type")
        info["hidden_size"] = config.get("hidden_size") or config.get("d_model")
        info["num_layers"] = (
            config.get("num_hidden_layers") or config.get("n_layer") or config.get("num_layers")
        )
        info["num_attention_heads"] = config.get("num_attention_heads") or config.get("n_head")
        info["vocab_size"] = config.get("vocab_size")
        info["max_position_embeddings"] = config.get("max_position_embeddings")
        info["torch_dtype"] = config.get("torch_dtype")

    if info.get("total_weight_bytes"):
        bytes_per_param = {
            "float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "float64": 8,
        }.get(info.get("torch_dtype"), 2)
        info["estimated_param_count"] = int(info["total_weight_bytes"] / bytes_per_param)
        if not info.get("torch_dtype"):
            info["param_count_estimation_note"] = "dtype unknown, assumed fp16 (2 bytes/param)"

    readme = fetch_hf_raw_file(model_id, "README.md")
    info["readme_excerpt"] = (readme[:1500] + "...") if readme and len(readme) > 1500 else readme

    info["inferred_modality"] = infer_modality(info.get("pipeline_tag"), info.get("tags", []))
    info["mode"] = "model"  # HF links are always treated as "model" mode

    return info
