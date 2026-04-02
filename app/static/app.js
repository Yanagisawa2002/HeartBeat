const state = {
  models: [],
  samples: [],
  currentWaveform: null,
  currentInputLabel: null,
  currentInputMeta: null,
  inferenceMode: "single",
  demoConfig: window.HEARTBEAT_DEMO || {
    expectedLeads: 12,
    signalLength: 1000,
    leadNames: [],
  },
};

const elements = {
  modelSelect: document.getElementById("model-select"),
  sampleSelect: document.getElementById("sample-select"),
  loadSampleButton: document.getElementById("load-sample-button"),
  csvFile: document.getElementById("csv-file"),
  form: document.getElementById("predict-form"),
  submitButton: document.getElementById("submit-button"),
  statusText: document.getElementById("status-text"),
  emptyState: document.getElementById("empty-state"),
  resultCard: document.getElementById("result-card"),
  predictedLabel: document.getElementById("predicted-label"),
  probabilityChip: document.getElementById("probability-chip"),
  metricModel: document.getElementById("metric-model"),
  metricAbnormal: document.getElementById("metric-abnormal"),
  metricNormal: document.getElementById("metric-normal"),
  metricShape: document.getElementById("metric-shape"),
  checkpointPath: document.getElementById("checkpoint-path"),
  waveformCanvas: document.getElementById("waveform-canvas"),
  modeInputs: document.querySelectorAll('input[name="inference_mode"]'),
  modeHelp: document.getElementById("mode-help"),
  modelField: document.getElementById("model-field"),
  sampleMetaCard: document.getElementById("sample-meta-card"),
  sampleMetaName: document.getElementById("sample-meta-name"),
  sampleMetaSource: document.getElementById("sample-meta-source"),
  sampleMetaType: document.getElementById("sample-meta-type"),
  sampleMetaDescription: document.getElementById("sample-meta-description"),
  compareCard: document.getElementById("compare-card"),
  compareTableBody: document.getElementById("compare-table-body"),
};

function setStatus(message, isError = false) {
  elements.statusText.textContent = message;
  elements.statusText.classList.toggle("status-error", isError);
}

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

function populateModels(models, defaultModel) {
  elements.modelSelect.innerHTML = "";

  models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = model.name;
    if (defaultModel && model.name === defaultModel) {
      option.selected = true;
    }
    elements.modelSelect.appendChild(option);
  });
}

function populateSamples(samples) {
  elements.sampleSelect.innerHTML = '<option value="">Choose a bundled demo sample</option>';

  samples.forEach((sample) => {
    const option = document.createElement("option");
    option.value = sample.file_name;
    option.textContent = sample.name;
    option.dataset.description = sample.description;
    option.dataset.source = sample.source;
    elements.sampleSelect.appendChild(option);
  });
}

function chooseDefaultSample(samples) {
  return (
    samples.find((sample) => sample.source === "ptb-xl-v1.0.1") ||
    samples[0] ||
    null
  );
}

function parseCsvText(text) {
  return text
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) =>
      line
        .split(",")
        .map((value) => {
          const parsed = Number(value.trim());
          if (!Number.isFinite(parsed)) {
            throw new Error("CSV contains non-numeric values.");
          }
          return parsed;
        })
    );
}

function normalizeWaveformShape(matrix) {
  if (!Array.isArray(matrix) || matrix.length === 0 || !Array.isArray(matrix[0])) {
    throw new Error("CSV must contain a 2D numeric matrix.");
  }

  const rows = matrix.length;
  const cols = matrix[0].length;
  const expectedLeads = state.demoConfig.expectedLeads;
  const expectedLength = state.demoConfig.signalLength;

  if (!matrix.every((row) => row.length === cols)) {
    throw new Error("CSV rows must all have the same number of columns.");
  }

  if (rows === expectedLeads && cols === expectedLength) {
    return matrix;
  }

  if (rows === expectedLength && cols === expectedLeads) {
    return Array.from({ length: expectedLeads }, (_, leadIndex) =>
      matrix.map((row) => row[leadIndex])
    );
  }

  throw new Error(
    `Expected shape ${expectedLeads}x${expectedLength} or ${expectedLength}x${expectedLeads}, received ${rows}x${cols}.`
  );
}

function readFileText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Unable to read file."));
    reader.readAsText(file);
  });
}

function drawWaveformGrid(context, leftPadding, rightPadding, width, height, leadCount) {
  const plotLeft = leftPadding;
  const plotRight = width - rightPadding;
  const plotTop = 10;
  const plotBottom = height - 10;
  const plotWidth = plotRight - plotLeft;
  const plotHeight = plotBottom - plotTop;
  const segmentHeight = plotHeight / leadCount;

  context.fillStyle = "#fffdf9";
  context.fillRect(plotLeft, plotTop, plotWidth, plotHeight);

  const minorVertical = 24;
  const majorVertical = minorVertical * 5;

  for (let x = plotLeft; x <= plotRight; x += minorVertical) {
    const isMajor = (x - plotLeft) % majorVertical === 0;
    context.strokeStyle = isMajor ? "rgba(120, 140, 162, 0.15)" : "rgba(120, 140, 162, 0.05)";
    context.lineWidth = isMajor ? 0.9 : 0.6;
    context.beginPath();
    context.moveTo(x, plotTop);
    context.lineTo(x, plotBottom);
    context.stroke();
  }

  for (let leadIndex = 0; leadIndex < leadCount; leadIndex += 1) {
    const rowTop = plotTop + leadIndex * segmentHeight;
    const minorHorizontal = segmentHeight / 5;
    for (let step = 0; step <= 5; step += 1) {
      const y = rowTop + step * minorHorizontal;
      const isMajor = step === 0 || step === 5;
      context.strokeStyle = isMajor ? "rgba(120, 140, 162, 0.15)" : "rgba(120, 140, 162, 0.05)";
      context.lineWidth = isMajor ? 0.9 : 0.6;
      context.beginPath();
      context.moveTo(plotLeft, y);
      context.lineTo(plotRight, y);
      context.stroke();
    }
  }
}

function drawWaveform(signalMatrix) {
  const canvas = elements.waveformCanvas;
  const context = canvas.getContext("2d");
  const { width, height } = canvas;
  const leadNames = state.demoConfig.leadNames;

  context.clearRect(0, 0, width, height);
  context.fillStyle = "#f8f5ef";
  context.fillRect(0, 0, width, height);

  if (!signalMatrix) {
    return;
  }

  const leadCount = signalMatrix.length;
  const segmentHeight = height / leadCount;
  const leftPadding = 60;
  const rightPadding = 16;
  const innerWidth = width - leftPadding - rightPadding;

  drawWaveformGrid(context, leftPadding, rightPadding, width, height, leadCount);

  context.font = "14px Segoe UI";
  context.lineWidth = 1.4;
  context.strokeStyle = "#1b6a68";
  context.fillStyle = "#5f6d7d";

  signalMatrix.forEach((leadSignal, leadIndex) => {
    const yBase = segmentHeight * leadIndex;
    const centerY = yBase + segmentHeight / 2;
    const amplitude = Math.max(...leadSignal.map((value) => Math.abs(value))) || 1;
    const yScale = (segmentHeight * 0.34) / amplitude;

    context.strokeStyle = "rgba(33, 56, 93, 0.10)";
    context.beginPath();
    context.moveTo(leftPadding, centerY);
    context.lineTo(width - rightPadding, centerY);
    context.stroke();

    context.fillStyle = "#5f6d7d";
    context.fillText(leadNames[leadIndex] || `Lead ${leadIndex + 1}`, 10, centerY + 5);

    context.strokeStyle = "#1b6a68";
    context.beginPath();
    leadSignal.forEach((value, sampleIndex) => {
      const x = leftPadding + (sampleIndex / (leadSignal.length - 1)) * innerWidth;
      const y = centerY - value * yScale;
      if (sampleIndex === 0) {
        context.moveTo(x, y);
      } else {
        context.lineTo(x, y);
      }
    });
    context.stroke();
  });
}

function renderEmptyWaveform() {
  const canvas = elements.waveformCanvas;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#f8f5ef";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#7b8794";
  context.font = "20px Segoe UI";
  context.fillText("Select a sample or upload a valid ECG CSV to preview the waveform.", 60, canvas.height / 2);
}

function resetResultsView() {
  elements.resultCard.classList.add("hidden");
  elements.compareCard.classList.add("hidden");
  elements.emptyState.classList.remove("hidden");
}

function summarizeSource(source) {
  if (source === "ptb-xl-v1.0.1") {
    return "PTB-XL example";
  }
  if (source === "synthetic-demo") {
    return "Synthetic fallback";
  }
  if (source === "user-upload") {
    return "User upload";
  }
  return source || "Unknown";
}

function inferInputType(meta) {
  if (!meta) {
    return "-";
  }
  if (meta.source === "ptb-xl-v1.0.1") {
    return "Public ECG window";
  }
  if (meta.source === "synthetic-demo") {
    return "Synthetic waveform";
  }
  if (meta.source === "user-upload") {
    return "Local CSV";
  }
  return "Demo input";
}

function renderSampleMeta(meta) {
  if (!meta) {
    elements.sampleMetaCard.classList.add("hidden");
    return;
  }

  elements.sampleMetaCard.classList.remove("hidden");
  elements.sampleMetaName.textContent = meta.name || state.currentInputLabel || "Current input";
  elements.sampleMetaSource.textContent = summarizeSource(meta.source);
  elements.sampleMetaType.textContent = inferInputType(meta);
  elements.sampleMetaDescription.textContent =
    meta.description || "Current ECG input loaded for preview and inference.";
}

function setInferenceMode(mode) {
  state.inferenceMode = mode === "compare" ? "compare" : "single";
  const isCompare = state.inferenceMode === "compare";
  elements.modelField.classList.toggle("hidden", isCompare);
  elements.modeHelp.textContent = isCompare
    ? "Run all bundled checkpoints on the current ECG window and compare their outputs side by side."
    : "Run inference with one selected checkpoint on the current ECG window.";
  resetResultsView();
}

function updateResultCard(result) {
  elements.emptyState.classList.add("hidden");
  elements.compareCard.classList.add("hidden");
  elements.resultCard.classList.remove("hidden");
  elements.predictedLabel.textContent = result.predicted_label;
  elements.probabilityChip.textContent = `${(result.probability_abnormal * 100).toFixed(1)}% abnormal`;
  elements.metricModel.textContent = result.model_name;
  elements.metricAbnormal.textContent = result.probability_abnormal.toFixed(4);
  elements.metricNormal.textContent = result.probability_normal.toFixed(4);
  elements.metricShape.textContent = result.input_shape.join(" x ");
  elements.checkpointPath.textContent = `Checkpoint: ${result.checkpoint_path}`;
}

function updateCompareCard(results) {
  elements.emptyState.classList.add("hidden");
  elements.resultCard.classList.add("hidden");
  elements.compareCard.classList.remove("hidden");
  elements.compareTableBody.innerHTML = "";

  results.forEach((result) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td><strong>${result.model_name}</strong></td>
      <td>${result.predicted_label}</td>
      <td>${result.probability_abnormal.toFixed(4)}</td>
      <td>${result.probability_normal.toFixed(4)}</td>
    `;
    elements.compareTableBody.appendChild(row);
  });
}

function setCurrentWaveform(matrix, label, meta = null) {
  state.currentWaveform = matrix;
  state.currentInputLabel = label;
  state.currentInputMeta = meta;
  renderSampleMeta(meta);
  resetResultsView();
  drawWaveform(matrix);
}

async function loadSampleByName(fileName) {
  return loadSampleByNameWithOptions(fileName, { announce: true });
}

async function loadSampleByNameWithOptions(fileName, options = {}) {
  const sample = state.samples.find((record) => record.file_name === fileName);
  if (!sample) {
    throw new Error("Selected sample is not available.");
  }

  const response = await fetch(`/sample-inputs/${encodeURIComponent(sample.file_name)}`);
  if (!response.ok) {
    throw new Error("Unable to load the selected sample file.");
  }

  const csvText = await response.text();
  const parsed = parseCsvText(csvText);
  const normalized = normalizeWaveformShape(parsed);
  setCurrentWaveform(normalized, sample.name, sample);
  if (options.announce !== false) {
    setStatus(`Loaded sample: ${sample.name} (${sample.source}).`);
  }
}

async function loadUploadedFile(file) {
  const text = await readFileText(file);
  const parsed = parseCsvText(text);
  const normalized = normalizeWaveformShape(parsed);
  setCurrentWaveform(normalized, file.name, {
    name: file.name,
    source: "user-upload",
    description: "CSV uploaded from the local browser session for one-window demo inference.",
  });
  setStatus(`Loaded upload: ${file.name}.`);
}

async function loadInitialState() {
  try {
    const [models, samples, demoConfig] = await Promise.all([
      fetchJson("/models"),
      fetchJson("/samples"),
      fetchJson("/demo-config"),
    ]);

    state.models = models;
    state.samples = samples;
    state.demoConfig = {
      expectedLeads: demoConfig.num_leads,
      signalLength: demoConfig.signal_length,
      leadNames: demoConfig.leads,
    };

    populateModels(models, demoConfig.default_model);
    populateSamples(samples);
    const defaultSample = chooseDefaultSample(samples);
    if (defaultSample) {
      elements.sampleSelect.value = defaultSample.file_name;
      await loadSampleByNameWithOptions(defaultSample.file_name, { announce: false });
    } else {
      renderEmptyWaveform();
    }

    if (models.length === 0) {
      elements.submitButton.disabled = true;
      setStatus(
        `Loaded ${samples.length} bundled sample(s), including a PTB-XL example preview, but no checkpoints were found. Mount a demo checkpoint to enable inference.`,
        true
      );
    } else {
      setStatus(
        `Ready. ${models.length} model checkpoint(s) and ${samples.length} bundled sample(s) available. A PTB-XL example window is preloaded for preview.`
      );
    }
  } catch (error) {
    setStatus(`Failed to load demo metadata: ${error.message}`, true);
    elements.submitButton.disabled = true;
  }
}

elements.modeInputs.forEach((input) => {
  input.addEventListener("change", (event) => {
    setInferenceMode(event.target.value);
  });
});

elements.loadSampleButton.addEventListener("click", async () => {
  const fileName = elements.sampleSelect.value;
  if (!fileName) {
    setStatus("Choose a bundled sample first.", true);
    return;
  }

  try {
    await loadSampleByName(fileName);
  } catch (error) {
    setStatus(error.message, true);
  }
});

elements.csvFile.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) {
    state.currentWaveform = null;
    state.currentInputLabel = null;
    state.currentInputMeta = null;
    renderSampleMeta(null);
    renderEmptyWaveform();
    return;
  }

  try {
    await loadUploadedFile(file);
  } catch (error) {
    state.currentWaveform = null;
    state.currentInputLabel = null;
    state.currentInputMeta = null;
    renderSampleMeta(null);
    renderEmptyWaveform();
    setStatus(error.message, true);
  }
});

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!state.currentWaveform) {
    setStatus("Load a bundled sample or upload a CSV before running inference.", true);
    return;
  }

  const modelName = elements.modelSelect.value;
  elements.submitButton.disabled = true;
  setStatus(
    `Running inference for ${state.currentInputLabel || "selected input"}...`
  );

  try {
    const endpoint = state.inferenceMode === "compare" ? "/predict-all" : "/predict";
    const requestBody =
      state.inferenceMode === "compare"
        ? { ecg: state.currentWaveform }
        : { model_name: modelName, ecg: state.currentWaveform };

    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Inference request failed.");
    }

    if (state.inferenceMode === "compare") {
      updateCompareCard(payload.predictions);
    } else {
      updateResultCard(payload);
    }
    drawWaveform(state.currentWaveform);
    setStatus("Inference complete.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    elements.submitButton.disabled = state.models.length === 0;
  }
});

setInferenceMode("single");
loadInitialState();
