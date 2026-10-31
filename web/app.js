const statusEl = document.querySelector("#status");
const classEl = document.querySelector("#pred-class");
const confEl = document.querySelector("#pred-conf");
const barsEl = document.querySelector("#bars");
const previewEl = document.querySelector("#preview");
const dropzone = document.querySelector("#dropzone");
const fileInput = document.querySelector("#file-input");

let session = null;
let config = null;
let classNames = ["leak", "no_leak", "environmental_noise"];
let previewUrl = "";

function setStatus(message) {
  statusEl.textContent = message;
}

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((value) => Math.exp(value - maxLogit));
  const sum = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sum);
}

function hann(size) {
  const window = new Float32Array(size);
  if (size === 1) {
    window[0] = 1;
    return window;
  }
  for (let i = 0; i < size; i += 1) {
    window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (size - 1));
  }
  return window;
}

function fftRadix2(real, imag) {
  const n = real.length;
  let j = 0;
  for (let i = 0; i < n; i += 1) {
    if (i < j) {
      const tr = real[i]; real[i] = real[j]; real[j] = tr;
      const ti = imag[i]; imag[i] = imag[j]; imag[j] = ti;
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = (2 * Math.PI) / size;
    for (let i = 0; i < n; i += size) {
      for (let k = 0; k < half; k += 1) {
        const angle = -step * k;
        const wr = Math.cos(angle);
        const wi = Math.sin(angle);
        const evenR = real[i + k];
        const evenI = imag[i + k];
        const oddR = real[i + k + half];
        const oddI = imag[i + k + half];
        const tr = wr * oddR - wi * oddI;
        const ti = wr * oddI + wi * oddR;
        real[i + k] = evenR + tr;
        imag[i + k] = evenI + ti;
        real[i + k + half] = evenR - tr;
        imag[i + k + half] = evenI - ti;
      }
    }
  }
}

function resample(input, fromRate, toRate) {
  if (fromRate === toRate) return input;
  const length = Math.max(1, Math.round(input.length * toRate / fromRate));
  const output = new Float32Array(length);
  const scale = (input.length - 1) / Math.max(length - 1, 1);
  for (let i = 0; i < length; i += 1) {
    const pos = i * scale;
    const left = Math.floor(pos);
    const right = Math.min(left + 1, input.length - 1);
    const t = pos - left;
    output[i] = input[left] * (1 - t) + input[right] * t;
  }
  return output;
}

function fixLength(input, size) {
  const output = new Float32Array(size);
  output.set(input.subarray(0, Math.min(input.length, size)));
  return output;
}

function logMel(samples, cfg) {
  const nFft = cfg.n_fft;
  const hop = cfg.hop_length;
  const nMels = cfg.n_mels;
  const nFrames = cfg.n_frames;
  const window = hann(nFft);
  const filterbank = cfg.mel_filterbank;
  const nBins = nFft / 2 + 1;
  const power = Array.from({ length: nBins }, () => new Float32Array(nFrames));
  for (let frame = 0; frame < nFrames; frame += 1) {
    const start = frame * hop;
    const real = new Float32Array(nFft);
    const imag = new Float32Array(nFft);
    for (let i = 0; i < nFft; i += 1) {
      real[i] = samples[start + i] * window[i];
    }
    fftRadix2(real, imag);
    for (let bin = 0; bin < nBins; bin += 1) {
      power[bin][frame] = real[bin] * real[bin] + imag[bin] * imag[bin];
    }
  }
  const spec = new Float32Array(nMels * nFrames);
  let peak = -1e9;
  for (let mel = 0; mel < nMels; mel += 1) {
    const row = filterbank[mel];
    for (let frame = 0; frame < nFrames; frame += 1) {
      let acc = 0;
      for (let bin = 0; bin < nBins; bin += 1) acc += row[bin] * power[bin][frame];
      const db = 10 * Math.log10(Math.max(1e-10, acc));
      spec[mel * nFrames + frame] = db;
      if (db > peak) peak = db;
    }
  }
  for (let i = 0; i < spec.length; i += 1) {
    spec[i] = Math.max(spec[i] - peak, -80);
  }
  return spec;
}

async function decodeFile(file) {
  const buffer = await file.arrayBuffer();
  const context = new AudioContext();
  const decoded = await context.decodeAudioData(buffer.slice(0));
  const channel = decoded.getChannelData(0);
  const resampled = resample(channel, decoded.sampleRate, config.sample_rate);
  await context.close();
  return fixLength(resampled, Math.round(config.sample_rate * config.duration));
}

function renderBars(probabilities) {
  const ranked = classNames
    .map((name, index) => ({ name, score: probabilities[index] }))
    .sort((a, b) => b.score - a.score);
  barsEl.innerHTML = ranked.map((row, order) => {
    const width = Math.max(2, Math.round(row.score * 100));
    return `<div class="bar-row${order === 0 ? " top" : ""}"><span>${row.name}</span><div class="bar-track"><div class="bar-fill" style="width:${width}%"></div></div><strong>${(row.score * 100).toFixed(1)}%</strong></div>`;
  }).join("");
}

async function infer(samples) {
  if (!session) throw new Error("Model is still loading.");
  const spec = logMel(samples, config);
  const tensor = new ort.Tensor("float32", spec, [1, 1, config.n_mels, config.n_frames]);
  const result = await session.run({ spectrogram: tensor });
  const logits = Array.from(result.logits.data);
  const probabilities = softmax(logits);
  const best = probabilities.indexOf(Math.max(...probabilities));
  classEl.textContent = classNames[best];
  confEl.textContent = `${(probabilities[best] * 100).toFixed(1)}% confidence`;
  renderBars(probabilities);
  setStatus("Prediction ready. This is an educational demo, not a plumbing decision.");
}

async function runBlob(blob, name) {
  setStatus("Reading audio…");
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(blob);
  previewEl.src = previewUrl;
  previewEl.hidden = false;
  const samples = await decodeFile(new File([blob], name || "clip.wav"));
  await infer(samples);
}

async function runUrl(url) {
  const response = await fetch(url);
  const blob = await response.blob();
  await runBlob(blob, url.split("/").pop());
}

async function boot() {
  try {
    ort.env.wasm.numThreads = 1;
    config = await (await fetch("downloads/config.json")).json();
    classNames = config.classes;
    session = await ort.InferenceSession.create("downloads/model.onnx", { executionProviders: ["wasm"] });
    setStatus("Model loaded. Choose a sample or upload a WAV.");
  } catch (error) {
    setStatus(`Could not load the model: ${error.message}`);
  }
}

document.querySelectorAll(".sample").forEach((button) => {
  button.addEventListener("click", () => {
    runUrl(button.dataset.src).catch((error) => setStatus(error.message));
  });
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  const file = event.dataTransfer.files[0];
  if (file) runBlob(file, file.name).catch((error) => setStatus(error.message));
});
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) runBlob(file, file.name).catch((error) => setStatus(error.message));
});

boot();
