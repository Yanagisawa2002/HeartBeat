const state = {
  models: [],
  samples: [],
  currentWaveform: null,
  currentInputLabel: null,
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

function updateResultCard(result) {
  elements.emptyState.classList.add("hidden");
  elements.resultCard.classList.remove("hidden");
  elements.predictedLabel.textContent = result.predicted_label;
  elements.probabilityChip.textContent = `${(result.probability_abnormal * 100).toFixed(1)}% abnormal`;
  elements.metricModel.textContent = result.model_name;
  elements.metricAbnormal.textContent = result.probability_abnormal.toFixed(4);
  elements.metricNormal.textContent = result.probability_normal.toFixed(4);
  elements.metricShape.textContent = result.input_shape.join(" x ");
  elements.checkpointPath.textContent = `Checkpoint: ${result.checkpoint_path}`;
}

function setCurrentWaveform(matrix, label) {
  state.currentWaveform = matrix;
  state.currentInputLabel = label;
  drawWaveform(matrix);
}

async function loadSampleByName(fileName) {
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
  setCurrentWaveform(normalized, sample.name);
  setStatus(`Loaded sample: ${sample.name} (${sample.source}).`);
}

async function loadUploadedFile(file) {
  const text = await readFileText(file);
  const parsed = parseCsvText(text);
  const normalized = normalizeWaveformShape(parsed);
  setCurrentWaveform(normalized, file.name);
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
    renderEmptyWaveform();

    if (models.length === 0) {
      elements.submitButton.disabled = true;
      setStatus(
        `Loaded ${samples.length} bundled sample(s), but no checkpoints were found. Mount a demo checkpoint to enable inference.`,
        true
      );
    } else {
      setStatus(
        `Ready. ${models.length} model checkpoint(s) and ${samples.length} bundled sample(s) available.`
      );
    }
  } catch (error) {
    setStatus(`Failed to load demo metadata: ${error.message}`, true);
    elements.submitButton.disabled = true;
  }
}

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
    renderEmptyWaveform();
    return;
  }

  try {
    await loadUploadedFile(file);
  } catch (error) {
    state.currentWaveform = null;
    state.currentInputLabel = null;
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
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model_name: modelName,
        ecg: state.currentWaveform,
      }),
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || "Inference request failed.");
    }

    updateResultCard(payload);
    drawWaveform(state.currentWaveform);
    setStatus("Inference complete.");
  } catch (error) {
    setStatus(error.message, true);
  } finally {
    elements.submitButton.disabled = state.models.length === 0;
  }
});

loadInitialState();
