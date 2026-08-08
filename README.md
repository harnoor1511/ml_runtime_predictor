# probe — ML Runtime Predictor & Repo Output Visualizer (Phase 1)

Paste a Hugging Face model link or a GitHub repo URL. The system reads
metadata only (no clone, no install, no execution), shows you what it found,
and asks a local LLM (`llama3.2:3b` via Ollama) to imagine what the
output/dashboard would look like.

Two modes, decided automatically where possible, or by you when it's ambiguous:

- **Model mode** — HF models, or GitHub repos with a clear ML framework/weight
  signal. Shows model info (architecture, params, size, dtype) + a mock
  prediction output.
- **Output-only mode** — GitHub repos that are tools/dashboards/libraries, not
  models. Shows repo info + a mock UI/output preview.

Latency/RAM/VRAM prediction (personalized to your hardware) is **not** part of
this phase — it plugs in after the mock-output step, using the same info
payload as its feature source.

## Project structure

```
ml_runtime_predictor/
├── backend/
│   ├── main.py                 FastAPI app: /api/analyze, /api/mock-output
│   ├── requirements.txt
│   ├── analyzer/
│   │   ├── hf_analyzer.py      Hugging Face metadata extraction
│   │   └── github_analyzer.py  GitHub metadata extraction + mode guessing
│   └── llm/
│       ├── ollama_client.py    Calls local Ollama HTTP API
│       └── prompts.py          Prompt templates (model vs output-only)
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── README.md
```

## Setup

### 1. Install and start Ollama (for the mock-output step)

```bash
# https://ollama.com for install instructions, then:
ollama pull llama3.2:3b
ollama serve        # usually starts automatically after install
```

### 2. Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

This also serves the frontend — open **http://localhost:8000** in your browser.

### 3. (Optional) GitHub API rate limits

Unauthenticated GitHub API calls are capped at 60/hr. To raise that to 5000/hr:

```bash
export GITHUB_TOKEN=your_personal_access_token
```
Set it before starting uvicorn (a token with no special scopes is enough — just public repo reads).

## How mode selection works

- Any `huggingface.co` link → always **model mode**.
- Any `github.com` link → the backend checks detected dependencies
  (torch/tensorflow/transformers/etc.) and committed weight files
  (`.pt`, `.safetensors`, `.onnx`, ...):
  - Clear ML signal found → **model mode**, no prompt shown.
  - No clear signal → the frontend shows the "Output-only / Model" choice
    so you decide.

## API reference

### `POST /api/analyze`
```json
{ "url": "https://github.com/ultralytics/yolov5", "mode": null }
```
`mode` is optional — omit it to let the backend auto-detect (or ask, for
ambiguous GitHub repos). Returns the full extracted info object, which
becomes the feature payload for both the mock-output step and, later, the
runtime predictor.

### `POST /api/mock-output`
```json
{ "info": { ...the object returned by /api/analyze... }, "mode": "model" }
```
Returns `{ "mock_output": "<llm-generated text>" }`.

## What's next (Phase 2)

- A local hardware-probe script (CPU/RAM/GPU/VRAM) that runs once and caches
  results, since browser JS alone can't get real specs.
- A trained LightGBM regressor (time / RAM / VRAM) + classifier (feasibility)
  using the `/api/analyze` output as its feature row, shown right below the
  mock-output panel.
