"""Extracts structured info from a GitHub repo without cloning/running it."""

import base64
import os
import re
from datetime import datetime, timezone

import requests

GITHUB_API = "https://api.github.com/repos"


def safe_get(url, headers=None, params=None, timeout=15):
    try:
        r = requests.get(url, headers=headers, params=params, timeout=timeout)
        return r if r.status_code == 200 else None
    except requests.RequestException:
        return None


def parse_github_url(url):
    m = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url.strip())
    if not m:
        raise ValueError("Could not parse GitHub URL. Expected format: https://github.com/owner/repo")
    return m.group(1), m.group(2)


def fetch_github_raw_file(owner, repo, branch, path, headers):
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"
    r = safe_get(url, headers=headers)
    return r.text if r else None


def fetch_github_readme(owner, repo, headers):
    r = safe_get(f"{GITHUB_API}/{owner}/{repo}/readme", headers=headers)
    if not r:
        return None
    data = r.json()
    try:
        return base64.b64decode(data.get("content", "")).decode("utf-8", errors="ignore")
    except Exception:
        return None


def detect_frameworks(deps_text, file_paths):
    frameworks = set()
    text = deps_text.lower()
    checks = {
        "pytorch": ["torch", "pytorch"],
        "tensorflow": ["tensorflow", "keras"],
        "jax": ["jax", "flax"],
        "onnx": ["onnx"],
        "sklearn": ["scikit-learn", "sklearn"],
        "transformers": ["transformers"],
        "diffusers": ["diffusers"],
        "opencv": ["opencv", "cv2"],
        "xgboost": ["xgboost"],
        "lightgbm": ["lightgbm"],
    }
    for fw, keywords in checks.items():
        if any(k in text for k in keywords):
            frameworks.add(fw)
    if any(f.endswith(".ipynb") for f in file_paths):
        frameworks.add("jupyter-notebooks-present")
    return sorted(frameworks)


def infer_modality(text):
    text = (text or "").lower()
    if any(k in text for k in ["image", "vision", "detection", "segmentation"]):
        return "image"
    if any(k in text for k in ["audio", "speech", "asr", "tts"]):
        return "audio"
    if "video" in text:
        return "video"
    if any(k in text for k in ["text", "nlp", "translation", "summarization", "generation", "question-answering"]):
        return "text"
    if any(k in text for k in ["tabular", "regression", "classification"]):
        return "tabular"
    return "unknown"


def guess_mode(frameworks, weight_files, description, topics):
    """Auto-suggest whether this repo is an ML 'model' repo or an 'output_only' (dashboard/tool) repo."""
    ml_frameworks = {"pytorch", "tensorflow", "jax", "onnx", "sklearn", "transformers", "diffusers", "xgboost", "lightgbm"}
    has_ml_framework = bool(ml_frameworks & set(frameworks))
    has_weights = len(weight_files) > 0
    text = f"{description or ''} {' '.join(topics or [])}".lower()
    dashboard_keywords = ["dashboard", "visualiz", "cli tool", "library", "framework", "sdk", "web app", "utility"]
    looks_like_dashboard = any(k in text for k in dashboard_keywords)

    if has_ml_framework or has_weights:
        return "model"
    if looks_like_dashboard:
        return "output_only"
    return "output_only"  # safe default: no ML signals found


def extract_frontend_source(owner, repo, branch, file_paths, headers, max_files=6, per_file_chars=1800):
    """For output_only mode: find and fetch the most relevant frontend component
    source files, so the LLM can reproduce a close visual approximation instead
    of guessing from a README description."""
    component_exts = (".jsx", ".tsx", ".vue", ".svelte")
    candidates = [f for f in file_paths if f.lower().endswith(component_exts)]

    # skip obvious non-screen files
    skip_names = {"index.jsx", "index.tsx", "main.jsx", "main.tsx", "vite-env.d.ts"}
    skip_dirs = ("test", "tests", "__tests__", "stories", "node_modules")
    candidates = [
        f for f in candidates
        if os.path.basename(f) not in skip_names
        and not any(seg in skip_dirs for seg in f.lower().split("/"))
    ]

    def priority(path):
        base = os.path.basename(path).lower()
        p = path.lower()
        if base.startswith("app."):
            return 0
        if "/pages/" in p or "/screens/" in p or "/views/" in p:
            return 1
        if "dashboard" in base or "home" in base:
            return 2
        if "/components/" in p:
            return 3
        return 4

    candidates.sort(key=priority)
    picked = candidates[:max_files]

    files = []
    for path in picked:
        content = fetch_github_raw_file(owner, repo, branch, path, headers)
        if content:
            files.append({"path": path, "content": content[:per_file_chars]})

    # grab a theme/config file if present — gives real colors/spacing to work with
    theme_candidates = [f for f in file_paths if os.path.basename(f) in (
        "tailwind.config.js", "tailwind.config.ts", "theme.js", "theme.ts"
    )]
    theme_content = None
    if theme_candidates:
        theme_content = fetch_github_raw_file(owner, repo, branch, theme_candidates[0], headers)
        if theme_content:
            theme_content = theme_content[:1500]

    return {
        "has_frontend_components": len(files) > 0,
        "files": files,
        "theme_file": theme_content,
        "theme_file_path": theme_candidates[0] if theme_candidates else None,
    }


def analyze_github_repo(url):
    owner, repo = parse_github_url(url)
    info = {
        "source_type": "github",
        "repo": f"{owner}/{repo}",
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }

    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    r = safe_get(f"{GITHUB_API}/{owner}/{repo}", headers=headers)
    if not r:
        info["error"] = "Could not fetch repo metadata from GitHub API (rate limit or repo not found)."
        return info

    meta = r.json()
    info["description"] = meta.get("description")
    info["language"] = meta.get("language")
    info["stars"] = meta.get("stargazers_count")
    info["forks"] = meta.get("forks_count")
    info["size_kb"] = meta.get("size")
    info["default_branch"] = meta.get("default_branch")
    info["topics"] = meta.get("topics", [])
    info["license"] = (meta.get("license") or {}).get("spdx_id")
    info["last_pushed"] = meta.get("pushed_at")

    tree_r = safe_get(
        f"{GITHUB_API}/{owner}/{repo}/git/trees/{info['default_branch']}",
        headers=headers, params={"recursive": "1"},
    )
    file_paths = []
    if tree_r:
        tree = tree_r.json()
        file_paths = [t["path"] for t in tree.get("tree", []) if t.get("type") == "blob"]
    info["file_count"] = len(file_paths)
    info["file_list_sample"] = file_paths[:200]

    dep_files = [f for f in file_paths if os.path.basename(f) in (
        "requirements.txt", "environment.yml", "pyproject.toml", "setup.py", "Pipfile"
    )]
    info["dependency_files"] = dep_files
    deps_text = ""
    for f in dep_files[:3]:
        content = fetch_github_raw_file(owner, repo, info["default_branch"], f, headers)
        if content:
            deps_text += f"\n# {f}\n" + content[:2000]
    info["dependencies_excerpt"] = deps_text.strip() or None
    info["detected_frameworks"] = detect_frameworks(deps_text, file_paths)

    weight_exts = (".pt", ".pth", ".onnx", ".h5", ".ckpt", ".safetensors", ".pb", ".tflite", ".gguf")
    info["weight_files_in_repo"] = [f for f in file_paths if f.lower().endswith(weight_exts)]

    entry_candidates = [f for f in file_paths if os.path.basename(f) in (
        "main.py", "run.py", "train.py", "inference.py", "app.py", "predict.py", "server.py"
    )]
    info["entrypoint_candidates"] = entry_candidates

    readme = fetch_github_readme(owner, repo, headers)
    info["readme_excerpt"] = (readme[:1500] + "...") if readme and len(readme) > 1500 else readme

    combined_text = f"{info.get('description','')} {' '.join(info.get('topics', []))} {deps_text}"
    info["inferred_modality"] = infer_modality(combined_text)
    info["suggested_mode"] = guess_mode(
        info["detected_frameworks"], info["weight_files_in_repo"], info["description"], info["topics"]
    )
    # Explicit strong-signal flag: mirrors the SAME narrow ml_frameworks set used
    # inside guess_mode (not the broader detected_frameworks list, which also
    # includes non-ML signals like opencv or "a notebook exists"). The frontend
    # uses this — not detected_frameworks/weight_files_in_repo directly — to
    # decide whether to ask the user model/output_only, so the two stay in sync.
    ml_frameworks = {"pytorch", "tensorflow", "jax", "onnx", "sklearn", "transformers", "diffusers", "xgboost", "lightgbm"}
    info["has_strong_ml_signal"] = bool(
        ml_frameworks & set(info["detected_frameworks"])
    ) or len(info["weight_files_in_repo"]) > 0

    # Only worth the extra fetches if this repo has a frontend-shaped structure at all
    component_exts = (".jsx", ".tsx", ".vue", ".svelte")
    if any(f.lower().endswith(component_exts) for f in file_paths):
        info["frontend_source"] = extract_frontend_source(
            owner, repo, info["default_branch"], file_paths, headers
        )
    else:
        info["frontend_source"] = {"has_frontend_components": False, "files": [], "theme_file": None}

    return info
