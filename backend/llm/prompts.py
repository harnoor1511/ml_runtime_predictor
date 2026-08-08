def build_model_mock_prompt(info):
    """Prompt for repos/models that ARE an ML model -> mock the model's EXPECTED OUTPUT
    only (what it is, sample outputs, dashboard). Latency/resource info is intentionally
    NOT part of this — that's handled separately by the raw, non-LLM "run system
    analysis" feature, and eventually a trained latency predictor. Keeping it out of the
    LLM's job avoids it inventing param counts/RAM numbers it has no real basis for."""
    source = info.get("model_id") or info.get("repo")
    source_type = info.get("source_type")  # "huggingface" or "github"
    description = info.get("description") or "No description available."
    topics = ", ".join(info.get("topics", [])) or "none"
    frameworks = info.get("detected_frameworks") or info.get("library_name") or "unknown"
    readme = info.get("readme_excerpt") or "No README available."

    # On Hugging Face, architecture/param-count/modality come from real config.json /
    # model-card metadata — trustworthy facts. On GitHub "model" mode, our modality/
    # framework guess is a crude keyword match with no config file behind it — a HINT
    # to be checked against the actual description/topics/README, not asserted as fact.
    is_hf = source_type == "huggingface"
    modality_hint = info.get("inferred_modality", "unknown")
    architecture = info.get("architecture")
    param_count = info.get("estimated_param_count")
    weight_size = info.get("total_weight_size_human")
    param_str = f"{param_count:,}" if isinstance(param_count, (int, float)) else None

    if is_hf:
        hard_facts = f"""Modality: {modality_hint}
Architecture: {architecture or 'unknown'}
Estimated parameter count: {param_str or 'unknown'}
Total weight size: {weight_size or 'unknown'}
Framework(s): {frameworks}"""
    else:
        hard_facts = f"""Our heuristic keyword-matcher's rough guess at modality: {modality_hint}
   (this is a crude keyword match over description/topics/deps — it is frequently wrong,
    especially for multi-stage tools; treat it as a hint to check, not a fact to repeat)
Detected framework keywords: {frameworks}
Architecture / parameter count: not available for GitHub repos (no config.json)."""

    return f"""You are simulating what a machine learning model's or ML-powered tool's output
would look like, WITHOUT actually running it. Base your answer only on the metadata below.

Source: {source}
Description: {description}
Topics/tags: {topics}
{hard_facts}
README excerpt:
\"\"\"{readme}\"\"\"

FIRST, before anything else: figure out what this actually is and does, using the
description, topics, and README as your primary evidence — they are real and specific;
any modality/framework guess above is only a hint and may be wrong or incomplete (e.g. a
repo tagged with an image-related dependency can still be a document-conversion pipeline,
not a vision classifier — read what it says it does, don't pattern-match on one keyword).
If it's a multi-stage pipeline/tool wrapping several sub-models rather than one single
model with one I/O contract, say so and describe it as that, not as a single classifier.

Then return a structured response with these sections — expected output only, do NOT
include anything about latency, speed, RAM/VRAM, or hardware requirements; that is
handled elsewhere by a separate, non-LLM system-specs feature:

1. What this actually is and does (1-2 sentences), and the likely output type — be precise
   and grounded in the description/README (e.g. "structured document JSON with page layout,
   tables, and text blocks" or "classification over N labels with confidence scores" or
   "autoregressive token-by-token text generation") — not a generic guess.

2. Two realistic SAMPLE outputs (fabricated but plausible, clearly labeled MOCK EXAMPLE) —
   one for a short/simple input and one for a longer/harder input, so the difference in
   output shape is visible. Match the sample's shape to what you concluded in section 1 —
   e.g. a document-conversion tool's sample should look like converted document JSON/
   markdown, not bounding boxes. Use real-looking sample JSON, labels, or text — no lorem
   ipsum. If conversational/text-generation, write an actual multi-turn exchange (user
   message + model reply), not a description of one.

3. What a results dashboard for this would display — list 4-6 concrete panels/widgets
   with what data each would show, tailored to the actual output type from section 1.
   Do not include any latency/performance/resource-usage panel here.

Keep it concise but do not skimp on section 2. Use clear headers, and always frame
outputs as simulated/mock — never claim this is a real run.
"""


def build_output_only_prompt(info):
    """Prompt for non-model repos (tools/dashboards/libraries) -> mock the UI/output it would produce."""
    repo = info.get("repo")
    description = info.get("description") or "No description available."
    language = info.get("language") or "unknown"
    topics = ", ".join(info.get("topics", [])) or "none"
    readme = info.get("readme_excerpt") or "No README available."
    entrypoints = ", ".join(info.get("entrypoint_candidates", [])) or "none detected"

    return f"""You are simulating what a software repository's output or dashboard would look like
WITHOUT actually installing or running it. Base your answer only on the metadata below —
the description, topics, and README are real and specific; use them as your primary
evidence for what this project actually does, rather than guessing from the repo name alone.

Repository: {repo}
Description: {description}
Primary language: {language}
Topics: {topics}
Detected entrypoints: {entrypoints}
README excerpt:
\"\"\"{readme}\"\"\"

Return a structured response with these sections:
1. What this project most likely does, in one or two sentences — grounded in the
   description/README, not a generic guess from the repo name.
2. Whether this project even HAS a user-facing UI/dashboard at all — check first. Many
   repos are pure libraries, CLIs, or backend-only services with NO visual interface of
   their own; if the README/description points to that (e.g. "install via pip and call
   in code", no mention of a web UI/GUI/dashboard anywhere, entrypoints look like library
   modules not servers), say plainly "this is a library/CLI with no built-in UI" and
   describe the CLI/API surface instead — do NOT invent a dashboard, chart area, or
   screen that isn't evidenced by the README. Only describe a web dashboard/GUI layout
   if the README/topics actually indicate one exists (mentions of a UI, a screenshot, a
   web app, a GUI, "playground", etc.). If it's a CLI, describe what running it prints
   instead (e.g. "a CLI tool that prints a formatted table" or "writes a converted output
   file to disk").
3. A realistic MOCK EXAMPLE of the output — e.g. a sample CLI output block, a sample JSON
   API response, a sample output file's content, or (only if a UI was confirmed in
   section 2) a description of a sample screen with example data values.
4. Key inputs a user would need to provide to use this project.

Keep it concise, use clear headers, and clearly label the example as a MOCK/SIMULATED preview, not a real run.
"""
