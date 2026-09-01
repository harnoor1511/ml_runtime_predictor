const API_BASE = "";

const form = document.getElementById("analyzeForm");
const urlInput = document.getElementById("urlInput");
const analyzeBtn = document.getElementById("analyzeBtn");
const errorBox = document.getElementById("errorBox");
const modeChoice = document.getElementById("modeChoice");
const resultsSection = document.getElementById("resultsSection");
const infoGrid = document.getElementById("infoGrid");
const modeTag = document.getElementById("modeTag");
const generateMockBtn = document.getElementById("generateMockBtn");
const mockOutput = document.getElementById("mockOutput");
const mockLoading = document.getElementById("mockLoading");
const ollamaDot = document.getElementById("ollamaDot");
const ollamaStatusText = document.getElementById("ollamaStatusText");
const runSysAnalysisBtn = document.getElementById("runSysAnalysisBtn");
const sysInfoGrid = document.getElementById("sysInfoGrid");
const sysInfoLoading = document.getElementById("sysInfoLoading");
const latencyPanel = document.getElementById("latencyPanel");
const calcLatencyBtn = document.getElementById("calcLatencyBtn");
const latencyLoading = document.getElementById("latencyLoading");
const latencyResult = document.getElementById("latencyResult");
const latencyUnsupported = document.getElementById("latencyUnsupported");

let systemSpecsReady = false;

let currentInfo = null;
let pendingUrl = null;

// ---------- Ollama health check (best-effort, non-blocking) ----------
async function checkOllama() {
  try {
    const res = await fetch("http://localhost:11434/api/tags", { method: "GET" });
    if (res.ok) {
      ollamaDot.classList.add("online");
      ollamaStatusText.textContent = "local llm ready";
      return;
    }
    throw new Error();
  } catch {
    ollamaDot.classList.add("offline");
    ollamaStatusText.textContent = "local llm unreachable";
  }
}
checkOllama();

// ---------- Helpers ----------
function showError(msg) {
  errorBox.textContent = msg;
  errorBox.classList.remove("hidden");
}
function clearError() {
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}
function setLoadingAnalyze(loading) {
  analyzeBtn.disabled = loading;
  analyzeBtn.querySelector("span").textContent = loading ? "inspecting…" : "inspect";
}
function fmt(v) {
  if (v === null || v === undefined || v === "") return "—";
  return v;
}

// ---------- Analyze flow ----------
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();
  modeChoice.classList.add("hidden");
  resultsSection.classList.add("hidden");

  const url = urlInput.value.trim();
  if (!url) return;

  pendingUrl = url;
  await runAnalyze(url, null);
});

async function runAnalyze(url, mode) {
  setLoadingAnalyze(true);
  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, mode }),
    });
    const data = await res.json();

    if (!res.ok) {
      showError(data.detail || "Something went wrong analyzing that link.");
      return;
    }

    // GitHub repos with no strong ML signal and no explicit mode chosen yet -> ask the user
    if (data.source_type === "github" && !mode && isAmbiguous(data)) {
      currentInfo = data;
      modeChoice.classList.remove("hidden");
      return;
    }

    currentInfo = data;
    renderResults(data);
  } catch (err) {
    showError("Could not reach the backend. Is the FastAPI server running on this address?");
  } finally {
    setLoadingAnalyze(false);
  }
}

function isAmbiguous(info) {
  // Mirrors guess_mode's actual ML-signal check on the backend (has_strong_ml_signal),
  // rather than re-deriving it from detected_frameworks, which also includes
  // non-ML-specific entries (opencv, "a notebook exists") that shouldn't count
  // as confident evidence this is a model repo.
  return !info.has_strong_ml_signal;
}

modeChoice.querySelectorAll(".mode-opt").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const mode = btn.dataset.mode;
    modeChoice.classList.add("hidden");
    await runAnalyze(pendingUrl, mode);
  });
});

// ---------- Render info grid ----------
function renderResults(info) {
  const mode = info.mode || info.suggested_mode || "output_only";
  modeTag.textContent = mode === "model" ? "model" : "output-only";
  modeTag.className = "mode-tag " + mode;

  infoGrid.innerHTML = "";

  if (info.source_type === "huggingface") {
    addCell("Model ID", info.model_id, true);
    addCell("Pipeline / task", info.pipeline_tag);
    addCell("Library", info.library_name);
    addCell("Inferred modality", info.inferred_modality);
    addCell("Architecture", Array.isArray(info.architecture) ? info.architecture.join(", ") : info.architecture);
    addCell("Hidden size", info.hidden_size);
    addCell("Layers", info.num_layers);
    addCell("Attention heads", info.num_attention_heads);
    addCell("Vocab size", info.vocab_size);
    addCell("Max position embeddings", info.max_position_embeddings);
    addCell("Torch dtype", info.torch_dtype);
    addCell("Total weight size", info.total_weight_size_human);
    addCell("Estimated params", info.estimated_param_count ? info.estimated_param_count.toLocaleString() : null);
    addCell("Downloads", info.downloads ? info.downloads.toLocaleString() : null);
    addCell("Likes", info.likes);
    addTagsCell("Weight files", info.weight_files);
    addTagsCell("Tags", info.tags);
  } else {
    addCell("Repository", info.repo, true);
    addCell("Description", info.description, false, true);
    addCell("Primary language", info.language);
    addCell("Inferred modality", info.inferred_modality);
    addCell("Stars", info.stars);
    addCell("Forks", info.forks);
    addCell("Repo size (KB)", info.size_kb);
    addCell("File count", info.file_count);
    addCell("License", info.license);
    addTagsCell("Detected frameworks", info.detected_frameworks);
    addTagsCell("Dependency files", info.dependency_files);
    addTagsCell("Entrypoint candidates", info.entrypoint_candidates);
    addTagsCell("Weight files in repo", info.weight_files_in_repo);
    addTagsCell("Topics", info.topics);
    if (info.frontend_source && info.frontend_source.has_frontend_components) {
      addTagsCell("Frontend components found", info.frontend_source.files.map((f) => f.path));
    }
  }

  resultsSection.classList.remove("hidden");
  mockOutput.className = "mock-output hidden";
  mockOutput.innerHTML = "";

  // New repo/model analyzed -> any previous latency prediction is stale
  latencyResult.classList.add("hidden");
  latencyResult.innerHTML = "";
  latencyUnsupported.classList.add("hidden");
  if (systemSpecsReady) {
    latencyPanel.classList.remove("hidden");
  }

  const fs = info.frontend_source || {};
  const speedToggle = document.getElementById("speedToggle");
  if (mode === "output_only" && fs.has_frontend_components) {
    generateMockBtn.querySelector("span").textContent = "generate visual from source";
    speedToggle.classList.remove("hidden");
  } else {
    generateMockBtn.querySelector("span").textContent = "generate mock";
    speedToggle.classList.add("hidden");
  }

  resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
}

function addCell(label, value, mono = false, wide = false, targetGrid = infoGrid, desc = null) {
  const cell = document.createElement("div");
  cell.className = "info-cell" + (wide ? " wide" : "");
  cell.innerHTML = `
    <div class="info-cell-label">${label}</div>
    <div class="info-cell-value ${mono ? "mono" : ""}">${escapeHtml(fmt(value))}</div>
    ${desc ? `<div class="info-cell-desc">${escapeHtml(desc)}</div>` : ""}
  `;
  targetGrid.appendChild(cell);
}

function addTagsCell(label, items, targetGrid = infoGrid) {
  if (!items || items.length === 0) return;
  const cell = document.createElement("div");
  cell.className = "info-cell wide";
  const tags = items.slice(0, 20).map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("");
  cell.innerHTML = `
    <div class="info-cell-label">${label}</div>
    <div class="tag-list">${tags}</div>
  `;
  targetGrid.appendChild(cell);
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = String(str);
  return div.innerHTML;
}

// ---------- Mock output generation ----------
generateMockBtn.addEventListener("click", async () => {
  if (!currentInfo) return;
  const mode = currentInfo.mode || currentInfo.suggested_mode || "output_only";
  const fs = currentInfo.frontend_source || {};
  const isReproHtml = mode === "output_only" && fs.has_frontend_components;
  // Frontend reproduction is always the fast, LLM-free parsed template now.
  // Only "model" and plain "output_only" (no frontend source) still hit the local LLM.
  const speed = isReproHtml ? "fast" : "quality";

  if (isReproHtml) {
    document.getElementById("mockLoadingText").textContent = "parsing source and templating a quick mockup…";
  } else {
    document.getElementById("mockLoadingText").textContent = "asking local llama3.2:3b to imagine the output…";
  }

  mockLoading.classList.remove("hidden");
  mockOutput.classList.add("hidden");
  mockOutput.innerHTML = "";
  generateMockBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/api/mock-output`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ info: currentInfo, mode, speed }),
    });
    const data = await res.json();

    if (!res.ok) {
      renderMockText(data.detail || "Could not generate a mock output.");
      return;
    }

    if (data.render_type === "html") {
      renderMockHtml(data.mock_output);
    } else {
      renderMockText(data.mock_output);
    }
  } catch (err) {
    renderMockText("Could not reach the backend to generate a mock output.");
  } finally {
    mockLoading.classList.add("hidden");
    generateMockBtn.disabled = false;
  }
});

function renderMockText(raw) {
  mockOutput.className = "mock-output mock-output-text";
  mockOutput.innerHTML = `<div class="markdown-body">${marked.parse(raw || "")}</div>`;
  mockOutput.classList.remove("hidden");
}

function renderMockHtml(html) {
  mockOutput.className = "mock-output mock-output-frame";
  mockOutput.innerHTML = "";
  const iframe = document.createElement("iframe");
  iframe.className = "mock-iframe";
  iframe.sandbox = "allow-same-origin";
  iframe.srcdoc = html;
  mockOutput.appendChild(iframe);
  mockOutput.classList.remove("hidden");
}

// ---------- Run System Analysis (raw specs, no LLM involved) ----------
runSysAnalysisBtn.addEventListener("click", async () => {
  runSysAnalysisBtn.disabled = true;
  sysInfoGrid.classList.add("hidden");
  sysInfoLoading.classList.remove("hidden");
  try {
    const res = await fetch(`${API_BASE}/api/system-specs`);
    if (!res.ok) throw new Error("Could not read system specs.");
    const specs = await res.json();
    renderSysInfo(specs);
    // Specs read successfully -> offer the "calculate latency" step. Only
    // shown once we actually have both a repo/model analyzed (currentInfo)
    // and this machine's specs, since the prediction needs both.
    systemSpecsReady = true;
    if (currentInfo) {
      latencyPanel.classList.remove("hidden");
    }
  } catch (err) {
    sysInfoGrid.innerHTML = `<div class="info-cell wide"><div class="info-cell-value">${escapeHtml(err.message)}</div></div>`;
    sysInfoGrid.classList.remove("hidden");
  } finally {
    sysInfoLoading.classList.add("hidden");
    runSysAnalysisBtn.disabled = false;
  }
});

function renderSysInfo(specs) {
  sysInfoGrid.innerHTML = "";
  const cpu = specs.cpu || {};
  const mem = specs.memory || {};
  const disk = specs.disk || {};
  const os = specs.os || {};
  const gpu = specs.gpu;

  addCell("CPU", cpu.brand, false, true, sysInfoGrid);
  addCell("Physical cores", cpu.physical_cores, false, false, sysInfoGrid);
  addCell("Logical cores", cpu.logical_cores, false, false, sysInfoGrid);
  addCell("Max CPU freq (MHz)", cpu.max_freq_mhz, false, false, sysInfoGrid);
  addCell("AVX2 support", cpu.supports_avx2, false, false, sysInfoGrid);
  addCell("AVX-512 support", cpu.supports_avx512, false, false, sysInfoGrid);
  addCell("Total RAM (GB)", mem.total_ram_gb, false, false, sysInfoGrid);
  addCell("Available RAM (GB)", mem.available_ram_gb, false, false, sysInfoGrid);
  addCell("Free disk (GB)", disk.free_disk_gb, false, false, sysInfoGrid);
  addCell("OS", os.os ? `${os.os} ${os.os_version || ""}`.trim() : null, false, false, sysInfoGrid);
  addCell("Architecture", os.architecture, false, false, sysInfoGrid);
  if (gpu) {
    addCell("GPU", gpu.name, false, false, sysInfoGrid);
    addCell("VRAM", gpu.vram, false, false, sysInfoGrid);
  } else {
    addCell("GPU", "none detected", false, false, sysInfoGrid);
  }

  sysInfoGrid.classList.remove("hidden");
}

// ---------- Calculate expected latency ----------
calcLatencyBtn.addEventListener("click", async () => {
  if (!currentInfo) return;
  calcLatencyBtn.disabled = true;
  latencyResult.classList.add("hidden");
  latencyResult.innerHTML = "";
  latencyUnsupported.classList.add("hidden");
  latencyLoading.classList.remove("hidden");

  try {
    const res = await fetch(`${API_BASE}/api/predict-latency`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ info: currentInfo }),
    });
    const data = await res.json();

    if (!res.ok) {
      latencyUnsupported.textContent = data.detail || "Could not calculate latency for this machine.";
      latencyUnsupported.classList.remove("hidden");
      return;
    }

    if (data.model_class === "unknown") {
      latencyUnsupported.textContent =
        `${data.message} (${data.reason})`;
      latencyUnsupported.classList.remove("hidden");
      return;
    }

    renderLatencyResult(data);
  } catch (err) {
    latencyUnsupported.textContent = "Could not reach the backend to calculate latency.";
    latencyUnsupported.classList.remove("hidden");
  } finally {
    latencyLoading.classList.add("hidden");
    calcLatencyBtn.disabled = false;
  }
});

function renderLatencyResult(data) {
  latencyResult.innerHTML = "";

  addCell("Detected model type", data.model_class, false, false, latencyResult);
  addCell("Detection confidence", data.detection?.confidence, false, false, latencyResult);
  addCell("Hardware used", data.hardware_used?.device_name, false, false, latencyResult);
  addCell("Hardware match note", data.hardware_used?.match_note, false, true, latencyResult);

  if (data.model_class === "llm") {
    addCell("Decode speed", `${data.tokens_per_sec} tokens/sec`, false, false, latencyResult, data.tokens_per_sec_desc);
    addCell("Decode latency", `${data.decode_ms_per_token} ms/token`, false, false, latencyResult, data.decode_desc);
    addCell("Prefill latency", `${data.prefill_ms_per_token} ms/token`, false, false, latencyResult, data.prefill_desc);
    if (data.load_time_estimate_sec) {
      addCell(
        "Estimated model load time",
        `~${data.load_time_estimate_sec.seconds}s (assumes ${data.load_time_estimate_sec.assumed_disk_read_mbps} MB/s disk)`,
        false, true, latencyResult, data.load_time_estimate_sec.desc
      );
    }
    if (typeof data.ram_pressure_ratio === "number") {
      addCell(
        "Free RAM vs model size",
        `${data.ram_pressure_ratio}x`,
        false, false, latencyResult,
        "How much free memory you have relative to the model's size. Below ~1.5x, expect slowdowns from paging."
      );
    }

    for (const key of ["short_qa", "summarization", "long_form"]) {
      const p = data.presets?.[key];
      if (!p) continue;
      addCell(
        `${p.label} (${p.input_tokens} in / ${p.output_tokens} out)`,
        `~${p.total_latency_sec}s total (TTFT ${p.ttft_ms}ms)`,
        false, true, latencyResult, p.desc
      );
    }
  } else if (data.model_class === "vision") {
    addCell("Per-image latency", `${data.ms_per_image} ms`, false, false, latencyResult, data.ms_per_image_desc);
    addCell("Throughput", `${data.fps} FPS`, false, false, latencyResult, data.fps_desc);
    if (data.load_time_estimate_sec) {
      addCell(
        "Estimated model load time",
        `~${data.load_time_estimate_sec.seconds}s (assumes ${data.load_time_estimate_sec.assumed_disk_read_mbps} MB/s disk)`,
        false, true, latencyResult, data.load_time_estimate_sec.desc
      );
    }
    if (typeof data.ram_pressure_ratio === "number") {
      addCell(
        "Free RAM vs model size",
        `${data.ram_pressure_ratio}x`,
        false, false, latencyResult,
        "How much free memory you have relative to the model's size. Below ~1.5x, expect slowdowns from paging."
      );
    }
  }

  if (data.attribute_notes && data.attribute_notes.length) {
    addTagsCell("Estimated (not from model card)", data.attribute_notes, latencyResult);
  }
  if (data.warnings && data.warnings.length) {
    addTagsCell("Warnings", data.warnings, latencyResult);
  }

  latencyResult.classList.remove("hidden");
}