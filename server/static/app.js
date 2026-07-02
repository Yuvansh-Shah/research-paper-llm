"use strict";

const els = {
  statusPill: document.getElementById("status-pill"),
  banner: document.getElementById("banner"),
  prompt: document.getElementById("prompt"),
  maxTokens: document.getElementById("max-tokens"),
  temperature: document.getElementById("temperature"),
  topK: document.getElementById("top-k"),
  topP: document.getElementById("top-p"),
  generate: document.getElementById("generate"),
  stop: document.getElementById("stop"),
  output: document.getElementById("output"),
  aboutBody: document.getElementById("about-body"),
};

let controller = null;
let modelReady = false;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function showBanner(message, notice) {
  els.banner.textContent = message;
  els.banner.classList.toggle("notice", Boolean(notice));
  els.banner.classList.remove("hidden");
}

function hideBanner() {
  els.banner.classList.add("hidden");
}

function setStatusPill(text, cls) {
  els.statusPill.textContent = text;
  els.statusPill.className = "pill" + (cls ? " " + cls : "");
}

function bindSlider(input, valueEl, format) {
  const update = () => {
    valueEl.textContent = format(input.value);
  };
  input.addEventListener("input", update);
  update();
}

bindSlider(els.maxTokens, document.getElementById("max-tokens-value"), (v) => v);
bindSlider(els.temperature, document.getElementById("temperature-value"), (v) =>
  Number(v) === 0 ? "0 (greedy)" : Number(v).toFixed(2)
);
bindSlider(els.topK, document.getElementById("top-k-value"), (v) =>
  Number(v) === 0 ? "0 (off)" : v
);
bindSlider(els.topP, document.getElementById("top-p-value"), (v) =>
  Number(v) === 1 ? "1.00 (off)" : Number(v).toFixed(2)
);

function formatParamCount(n) {
  if (typeof n !== "number") return "?";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  return n.toLocaleString();
}

function infoRow(label, value) {
  return `<tr><th>${escapeHtml(label)}</th><td>${escapeHtml(value)}</td></tr>`;
}

function renderEvaluation(evaluation) {
  const fallback =
    "<p>Not yet evaluated &mdash; run <code>evaluation/evaluate.py --save_outputs</code> " +
    "to produce <code>evaluation/results/evaluation_report.json</code>.</p>";
  if (!evaluation || !Array.isArray(evaluation.perplexity_results) || evaluation.perplexity_results.length === 0) {
    return fallback;
  }
  const rows = evaluation.perplexity_results
    .map(
      (item) =>
        "<tr>" +
        `<td>${escapeHtml(item.split)}</td>` +
        `<td>${Number(item.loss).toFixed(4)}</td>` +
        `<td>${Number(item.perplexity).toFixed(2)}</td>` +
        `<td>${Number(item.num_tokens).toLocaleString()}</td>` +
        "</tr>"
    )
    .join("");
  const meta = [];
  if (evaluation.timestamp) meta.push(`evaluated ${escapeHtml(evaluation.timestamp)}`);
  if (evaluation.prompt_suite) meta.push(`prompt suite: ${escapeHtml(evaluation.prompt_suite)}`);
  return (
    '<table class="info-table eval-table">' +
    "<tr><th>Split</th><th>Loss</th><th>Perplexity</th><th>Tokens</th></tr>" +
    rows +
    "</table>" +
    (meta.length ? `<p>${meta.join(" &middot; ")}</p>` : "")
  );
}

function renderAbout(info) {
  const blurb =
    "<p>A decoder-only Transformer (GPT-2 style) implemented from scratch in PyTorch and " +
    "trained on the titles and abstracts of ~2.7M arXiv papers (566M tokens), " +
    "tokenized with a 32k BPE tokenizer trained on the same corpus. It autocompletes " +
    "academic-sounding prose; it is a small research demo, not a general-purpose assistant.</p>";

  if (info.status !== "ready" || !info.model) {
    els.aboutBody.innerHTML =
      "<h3>Architecture</h3><p>No checkpoint loaded yet.</p>" +
      "<h3>Training data</h3>" + blurb +
      "<h3>Evaluation</h3>" + renderEvaluation(info.evaluation);
    return;
  }

  const m = info.model;
  const archRows = [
    infoRow("Layers", m.n_layers),
    infoRow("Attention heads", m.n_heads),
    infoRow("Embedding dim (d_model)", m.n_embd),
    infoRow("Context length", m.context_length),
    infoRow("Vocab size", m.vocab_size.toLocaleString()),
    infoRow("Parameters", formatParamCount(m.num_parameters)),
    infoRow("Device", info.device),
    infoRow("Tokenizer", `${info.tokenizer.name} (${info.tokenizer.backend})`),
    infoRow("Checkpoint step", info.checkpoint.step === null ? "?" : info.checkpoint.step),
    infoRow("Checkpoint", info.checkpoint.path),
  ].join("");

  els.aboutBody.innerHTML =
    `<h3>Architecture</h3><table class="info-table">${archRows}</table>` +
    "<h3>Training data</h3>" + blurb +
    "<h3>Evaluation</h3>" + renderEvaluation(info.evaluation);
}

async function loadModelInfo() {
  try {
    const res = await fetch("/api/model-info");
    if (!res.ok) throw new Error(`model-info failed (${res.status})`);
    const info = await res.json();
    modelReady = info.status === "ready";
    if (modelReady) {
      setStatusPill("model loaded", "ready");
      els.generate.disabled = false;
    } else {
      setStatusPill("no checkpoint", "missing");
      showBanner(info.message || "No checkpoint yet - run training first.", true);
    }
    renderAbout(info);
  } catch (err) {
    setStatusPill("offline");
    els.aboutBody.textContent = "Could not load model info.";
    showBanner(String(err));
  }
}

function setRunning(running) {
  els.generate.disabled = running || !modelReady;
  els.stop.disabled = !running;
}

async function generate() {
  const prompt = els.prompt.value;
  if (!prompt.trim()) {
    showBanner("Enter a prompt first.", true);
    return;
  }
  hideBanner();
  els.output.textContent = "";
  controller = new AbortController();
  setRunning(true);
  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        prompt,
        max_tokens: Number(els.maxTokens.value),
        temperature: Number(els.temperature.value),
        top_k: Number(els.topK.value),
        top_p: Number(els.topP.value),
      }),
    });
    if (!res.ok) {
      let message = `Request failed (${res.status})`;
      try {
        const err = await res.json();
        if (err && err.error) message = err.error;
      } catch {
        /* non-JSON error body */
      }
      showBanner(message, res.status === 409 || res.status === 503);
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      els.output.textContent += decoder.decode(value, { stream: true });
      els.output.scrollTop = els.output.scrollHeight;
    }
    els.output.textContent += decoder.decode();
    if (!els.output.textContent) {
      els.output.textContent = "(model emitted end-of-text immediately)";
    }
  } catch (err) {
    if (err.name !== "AbortError") showBanner(String(err));
  } finally {
    controller = null;
    setRunning(false);
  }
}

els.generate.addEventListener("click", generate);
els.stop.addEventListener("click", () => {
  if (controller) controller.abort();
});
els.prompt.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && !els.generate.disabled) {
    generate();
  }
});

loadModelInfo();
