"""
Decides which trained latency model applies to a given analyzed repo/model.

Currently recognizes: "llm", "vision". Everything else returns "unknown" with
a reason, rather than guessing -- the caller should surface that plainly
rather than silently defaulting to one category.

Designed to be extended: adding a new category (audio/embedding/diffusion)
means adding a new branch here plus a new predict_<category>() in predict.py
-- this function is the single place that routes to one or the other.
"""

_LLM_ARCH_KEYWORDS = {
    "llama", "mistral", "mixtral", "gpt2", "gptj", "gpt_neox", "falcon",
    "qwen", "gemma", "phi", "bloom", "opt", "mpt", "baichuan", "yi", "deepseek",
    "causallm", "gpt_bigcode", "starcoder", "codegen",
}
_LLM_PIPELINE_TAGS = {
    "text-generation", "text2text-generation", "conversational",
}
_VISION_PIPELINE_TAGS = {
    "image-classification", "object-detection", "image-segmentation",
    "zero-shot-image-classification", "image-to-image",
}
_VISION_ARCH_KEYWORDS = {
    "resnet", "vit", "vision_transformer", "swin", "convnext", "efficientnet",
    "mobilenet", "yolo", "detr", "mask2former", "deeplab", "regnet", "densenet",
    "vgg", "squeezenet", "deit", "beit",
}


def _arch_strings(info: dict):
    arch = info.get("architecture")
    if isinstance(arch, list):
        return [a.lower() for a in arch]
    if isinstance(arch, str):
        return [arch.lower()]
    return []


def detect_model_class(info: dict):
    """Returns (model_class, confidence, reason). model_class is one of
    'llm', 'vision', 'unknown'."""
    source = info.get("source_type")

    if source == "huggingface":
        pipeline_tag = (info.get("pipeline_tag") or "").lower()
        modality = (info.get("inferred_modality") or "").lower()
        archs = _arch_strings(info)
        tags = [t.lower() for t in (info.get("tags") or [])]

        if pipeline_tag in _LLM_PIPELINE_TAGS or any(
            k in a for a in archs for k in _LLM_ARCH_KEYWORDS
        ) or any(k in " ".join(tags) for k in _LLM_ARCH_KEYWORDS):
            return "llm", "high", f"HF pipeline_tag/architecture matched LLM signals ({pipeline_tag or archs})"

        if pipeline_tag in _VISION_PIPELINE_TAGS or modality == "image" or any(
            k in a for a in archs for k in _VISION_ARCH_KEYWORDS
        ):
            return "vision", "high", f"HF pipeline_tag/architecture matched vision signals ({pipeline_tag or archs})"

        if modality == "text":
            # text modality but not clearly a generation architecture -- could
            # be an encoder-only/embedding model, which we don't have a
            # trained predictor for yet
            return "unknown", "low", "text modality but not a recognized causal-LM architecture (possibly an embedding/encoder model -- not yet supported)"

        return "unknown", "low", f"pipeline_tag '{pipeline_tag}' / modality '{modality}' not yet mapped to a supported model class"

    # GitHub repos: weaker signal, based on detected frameworks/topics/description
    text_blob = " ".join([
        info.get("description") or "",
        " ".join(info.get("topics") or []),
        " ".join(info.get("detected_frameworks") or []),
    ]).lower()

    if any(k in text_blob for k in _LLM_ARCH_KEYWORDS) or "llm" in text_blob or "language model" in text_blob:
        return "llm", "medium", "GitHub description/topics/frameworks matched LLM keywords"
    if any(k in text_blob for k in _VISION_ARCH_KEYWORDS) or "object detection" in text_blob or "image classification" in text_blob:
        return "vision", "medium", "GitHub description/topics/frameworks matched vision keywords"

    return "unknown", "low", "no confident LLM or vision signal found in repo metadata"
