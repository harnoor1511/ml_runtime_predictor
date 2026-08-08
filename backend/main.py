from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from analyzer.hf_analyzer import analyze_hf_model
from analyzer.github_analyzer import analyze_github_repo
from analyzer.jsx_mockup import build_fast_mockup_html
from llm.ollama_client import call_ollama, DEFAULT_MODEL
from llm.prompts import build_model_mock_prompt, build_output_only_prompt
from system_specs import collect_system_specs

app = FastAPI(title="ML Runtime Predictor & Repo Visualizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# Collected once at server start — hardware doesn't change mid-session, and
# this keeps every /api/mock-output call from re-shelling to nvidia-smi/ollama.
_SYSTEM_SPECS = collect_system_specs()


@app.get("/api/system-specs")
def system_specs():
    return _SYSTEM_SPECS


class AnalyzeRequest(BaseModel):
    url: str
    mode: str | None = None  # "model" | "output_only" | None (auto-detect for github)


class MockRequest(BaseModel):
    info: dict
    mode: str  # "model" | "output_only"
    ollama_model: str | None = None
    speed: str = "quality"  # "quality" (LLM) | "fast" (no LLM, parsed template — output_only only)


def detect_source(url: str) -> str:
    url = url.strip()
    if "huggingface.co" in url:
        return "huggingface"
    if "github.com" in url:
        return "github"
    raise HTTPException(status_code=400, detail="URL must be a huggingface.co or github.com link.")


def extract_hf_id(url: str) -> str:
    # Accepts either a raw model id ("bert-base-uncased") or a full URL
    url = url.strip()
    if "huggingface.co" not in url:
        model_id = url
    else:
        model_id = url.rstrip("/").split("huggingface.co/")[-1]

    # Strip query string / fragment (?, #) and any trailing path segments
    # beyond owner/name (e.g. /tree/main, /blob/main/config.json)
    model_id = model_id.split("?")[0].split("#")[0]
    segments = [s for s in model_id.split("/") if s]
    known_suffixes = {"tree", "blob", "resolve", "commit", "discussions"}
    cleaned = []
    for seg in segments:
        if seg in known_suffixes:
            break
        cleaned.append(seg)
    model_id = "/".join(cleaned)

    if not model_id or model_id in ("models", "models/"):
        raise HTTPException(
            status_code=400,
            detail="Could not find a valid model id in that URL. "
                   "Expected something like https://huggingface.co/owner/model-name",
        )
    return model_id


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    source = detect_source(req.url)

    if source == "huggingface":
        model_id = extract_hf_id(req.url)
        info = analyze_hf_model(model_id)
        if info.get("error"):
            raise HTTPException(status_code=404, detail=info["error"])
        return info

    # GitHub
    try:
        info = analyze_github_repo(req.url)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if info.get("error"):
        raise HTTPException(status_code=404, detail=info["error"])

    # honor explicit user-selected mode; otherwise use the auto-suggested one
    info["mode"] = req.mode if req.mode in ("model", "output_only") else info.get("suggested_mode", "output_only")
    return info


def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # drop opening fence (with optional language tag) and closing fence
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


@app.post("/api/mock-output")
def mock_output(req: MockRequest):
    fs = (req.info or {}).get("frontend_source") or {}
    use_frontend_repro = req.mode == "output_only" and fs.get("has_frontend_components")

    # Frontend reproduction is always the fast, LLM-free path: parse the source
    # with regex and template an HTML mockup directly. Runs in milliseconds,
    # less faithful to the real UI, but always available and never times out.
    # (The old LLM-based HTML reproduction path has been removed — it was too
    # slow on CPU for the size of prompt it needed.)
    if use_frontend_repro:
        html_out = build_fast_mockup_html(req.info)
        return {"mock_output": html_out, "render_type": "html", "generation": "fast"}

    if req.mode == "model":
        prompt = build_model_mock_prompt(req.info)
        render_type = "text"
        num_predict = 900
    else:
        prompt = build_output_only_prompt(req.info)
        render_type = "text"
        num_predict = 600

    result = call_ollama(prompt, model=req.ollama_model or DEFAULT_MODEL, num_predict=num_predict)
    if not result["ok"]:
        raise HTTPException(status_code=503, detail=result["error"])

    text = result["text"]
    if render_type == "html":
        text = strip_code_fences(text)
        # sanity check: if it doesn't look like HTML, fall back to text rendering
        if "<html" not in text.lower() and "<!doctype" not in text.lower():
            render_type = "text"

    return {"mock_output": text, "render_type": render_type, "generation": "llm"}


# --- Serve frontend ---
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
