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
let sharedAudioContext = null;
let stageToken = 0;
let inferToken = 0;
let inferBusy = false;
let loadFailed = false;
const MAX_UPLOAD_BYTES = 12 * 1024 * 1024;

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
  // numpy.hanning(M) = 0.5 - 0.5 * cos(2 pi n / (M - 1))
  for (let i = 0; i < size; i += 1) {
    window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (size - 1));
  }
  return window;
}

function mixToMono(buffer) {
  const frames = buffer.length;
  const channels = buffer.numberOfChannels;
  if (channels === 1) return buffer.getChannelData(0);
  const mixed = new Float32Array(frames);
  for (let channel = 0; channel < channels; channel += 1) {
    const data = buffer.getChannelData(channel);
    for (let i = 0; i < frames; i += 1) mixed[i] += data[i];
  }
  const scale = 1 / channels;
  for (let i = 0; i < frames; i += 1) mixed[i] *= scale;
  return mixed;
}

function peakAmplitude(samples) {
  let peak = 0;
  for (let i = 0; i < samples.length; i += 1) peak = Math.max(peak, Math.abs(samples[i]));
  return peak;
}

async function getAudioContext() {
  if (sharedAudioContext && sharedAudioContext.state !== "closed") return sharedAudioContext;
  const Ctor = window.AudioContext || window.webkitAudioContext;
  if (!Ctor) throw new Error("This browser cannot decode audio.");
  sharedAudioContext = new Ctor();
  return sharedAudioContext;
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
  if (!config) throw new Error("The model configuration is still loading.");
  const buffer = await file.arrayBuffer();
  const context = await getAudioContext();
  const decoded = await context.decodeAudioData(buffer.slice(0));
  const channel = mixToMono(decoded);
  const resampled = resample(channel, decoded.sampleRate, config.sample_rate);
  const samples = fixLength(resampled, Math.round(config.sample_rate * config.duration));
  if (!samples.length || peakAmplitude(samples) < 1e-7) {
    throw new Error("This recording is silent. Choose a clip that contains sound.");
  }
  return { samples, sourceRate: decoded.sampleRate, sourceSeconds: decoded.duration, channels: decoded.numberOfChannels };
}

const LABELS = {
  leak: "Leak",
  no_leak: "No leak",
  environmental_noise: "Environmental noise",
};
const TONES = { leak: "is-leak", no_leak: "is-clear", environmental_noise: "is-noise" };
const HEADLINES = {
  leak: "Leak detected",
  no_leak: "No leak detected",
  environmental_noise: "Environmental noise",
};

const analyzeBtn = document.querySelector("#analyze");
const resetBtn = document.querySelector("#reset");
const fileMeta = document.querySelector("#file-meta");
const outcome = document.querySelector("#outcome");
const waveCanvas = document.querySelector("#wave");
const melCanvas = document.querySelector("#mel");
let pendingSamples = null;
let pendingName = "";

function pretty(name) {
  return LABELS[name] || name;
}

function drawWave(samples) {
  if (!waveCanvas) return;
  waveCanvas.hidden = false;
  const ctx = waveCanvas.getContext("2d");
  const width = waveCanvas.width;
  const height = waveCanvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = "#00C9B7";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  const step = Math.max(1, Math.floor(samples.length / width));
  for (let x = 0; x < width; x += 1) {
    const start = x * step;
    let min = 1;
    let max = -1;
    for (let i = 0; i < step && start + i < samples.length; i += 1) {
      const value = samples[start + i];
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
    ctx.moveTo(x + 0.5, height / 2 - max * (height * 0.42));
    ctx.lineTo(x + 0.5, height / 2 - min * (height * 0.42));
  }
  ctx.stroke();
}

function drawMel(spec) {
  if (!melCanvas || !config) return;
  melCanvas.hidden = false;
  const ctx = melCanvas.getContext("2d");
  const width = melCanvas.width;
  const height = melCanvas.height;
  const frames = config.n_frames;
  const mels = config.n_mels;
  ctx.clearRect(0, 0, width, height);
  const cellW = width / frames;
  const cellH = height / mels;
  for (let mel = 0; mel < mels; mel += 1) {
    for (let frame = 0; frame < frames; frame += 1) {
      const value = (spec[mel * frames + frame] + 80) / 80;
      const tone = Math.max(0, Math.min(255, Math.round(value * 255)));
      ctx.fillStyle = `rgb(${Math.round(tone * 0.15)}, ${Math.round(40 + tone * 0.7)}, ${Math.round(80 + tone * 0.6)})`;
      ctx.fillRect(frame * cellW, (mels - 1 - mel) * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
  }
}

function renderBars(probabilities) {
  barsEl.innerHTML = classNames.map((name, index) => {
    const score = probabilities[index];
    const width = Math.max(0, score * 100);
    const label = pretty(name);
    return `<div class="bar-row"><span>${label}</span><div class="bar-track" role="meter" aria-label="${label} confidence" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${width.toFixed(1)}"><div class="bar-fill" style="width:${width}%"></div></div><strong>${width.toFixed(1)}%</strong></div>`;
  }).join("");
}

function clearPrediction() {
  outcome.className = "outcome";
  classEl.textContent = "Awaiting analysis";
  confEl.textContent = "Choose a sample or a WAV, then run analysis.";
  barsEl.innerHTML = "";
}

function setAnalyzeEnabled() {
  analyzeBtn.disabled = !session || !pendingSamples || inferBusy || loadFailed;
}

async function infer(samples) {
  if (inferBusy) return;
  if (!session) throw new Error("The ONNX model is not loaded yet.");
  inferBusy = true;
  const token = ++inferToken;
  setAnalyzeEnabled();
  setStatus("Running the network…");
  try {
    const spec = logMel(samples, config);
    if (spec.length !== config.n_mels * config.n_frames) {
      throw new Error(`Spectrogram shape mismatch: expected ${config.n_mels}×${config.n_frames}.`);
    }
    drawMel(spec);
    const tensor = new ort.Tensor("float32", spec, [1, 1, config.n_mels, config.n_frames]);
    const result = await session.run({ spectrogram: tensor });
    if (token !== inferToken) return;
    const output = result.logits || result[Object.keys(result)[0]];
    const logits = Array.from(output.data);
    const probabilities = softmax(logits);
    const best = probabilities.indexOf(Math.max(...probabilities));
    const name = classNames[best];
    outcome.className = `outcome ${TONES[name] || ""}`;
    classEl.textContent = HEADLINES[name] || pretty(name);
    confEl.textContent = `${(probabilities[best] * 100).toFixed(1)}% confidence · ${pretty(name)}`;
    renderBars(probabilities);
    setStatus("Prediction ready. This is an educational demo, not a plumbing decision.");
  } finally {
    if (token === inferToken) inferBusy = false;
    setAnalyzeEnabled();
  }
}

function describeFile(blob, name, decoded) {
  const size = blob.size ? `${Math.max(1, Math.round(blob.size / 1024))} KB` : "sample";
  const sourceSeconds = decoded.sourceSeconds.toFixed(2);
  const channels = decoded.channels === 1 ? "mono" : `${decoded.channels}-channel`;
  return `${name} · ${channels} · ${Math.round(decoded.sourceRate)} Hz · ${sourceSeconds} s source · resampled to ${config.sample_rate} Hz / ${config.duration.toFixed(1)} s`;
}

async function stageBlob(blob, name) {
  const token = ++stageToken;
  inferToken += 1;
  inferBusy = false;
  pendingSamples = null;
  setAnalyzeEnabled();
  setStatus("Reading audio…");
  clearPrediction();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = "";
  if (blob.size === 0) {
    setStatus("This file is empty.");
    throw new Error("This file is empty.");
  }
  if (blob.size > MAX_UPLOAD_BYTES) {
    setStatus("This file is larger than 12 MB. Use a short WAV clip.");
    throw new Error("This file is larger than 12 MB. Use a short WAV clip.");
  }
  previewUrl = URL.createObjectURL(blob);
  previewEl.src = previewUrl;
  previewEl.hidden = false;
  previewEl.load();
  const file = blob instanceof File ? blob : new File([blob], name || "clip.wav");
  let decoded;
  try {
    decoded = await decodeFile(file);
  } catch (error) {
    if (token !== stageToken) return;
    pendingSamples = null;
    setAnalyzeEnabled();
    const message = String(error && error.message ? error.message : error);
    if (message.includes("silent")) {
      setStatus(message);
      throw error;
    }
    setStatus("This file could not be read as audio. Choose a WAV recording.");
    throw new Error("This file could not be read as audio. Choose a WAV recording.");
  }
  if (token !== stageToken) return;
  pendingSamples = decoded.samples;
  pendingName = name || file.name;
  fileMeta.hidden = false;
  fileMeta.textContent = describeFile(blob, pendingName, decoded);
  drawWave(pendingSamples);
  if (melCanvas) {
    melCanvas.getContext("2d").clearRect(0, 0, melCanvas.width, melCanvas.height);
    melCanvas.hidden = true;
  }
  setAnalyzeEnabled();
  setStatus(session ? "Audio ready. Press Analyze audio." : (loadFailed ? "The model failed to load, so analysis is unavailable." : "Audio ready. Waiting for the model."));
}

async function stageUrl(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error("Sample audio could not be loaded.");
  const blob = await response.blob();
  await stageBlob(blob, url.split("/").pop());
}

function resetDemo() {
  stageToken += 1;
  inferToken += 1;
  inferBusy = false;
  pendingSamples = null;
  pendingName = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = "";
  previewEl.removeAttribute("src");
  previewEl.hidden = true;
  fileMeta.hidden = true;
  fileMeta.textContent = "";
  fileInput.value = "";
  setAnalyzeEnabled();
  clearPrediction();
  if (waveCanvas) {
    waveCanvas.getContext("2d").clearRect(0, 0, waveCanvas.width, waveCanvas.height);
    waveCanvas.hidden = true;
  }
  if (melCanvas) {
    melCanvas.getContext("2d").clearRect(0, 0, melCanvas.width, melCanvas.height);
    melCanvas.hidden = true;
  }
  document.querySelectorAll(".sample").forEach((button) => button.classList.remove("is-selected"));
  if (loadFailed) setStatus("Could not load the ONNX model. Check the network and refresh.");
  else if (session) setStatus("Model loaded. Choose a sample or upload a WAV.");
  else setStatus("Loading the ONNX model…");
}

async function boot() {
  try {
    if (typeof ort === "undefined") throw new Error("ONNX Runtime failed to load from the CDN.");
    ort.env.wasm.numThreads = 1;
    const configResponse = await fetch("downloads/config.json");
    if (!configResponse.ok) throw new Error("config.json could not be downloaded.");
    config = await configResponse.json();
    classNames = config.classes;
    session = await ort.InferenceSession.create("downloads/model.onnx", { executionProviders: ["wasm"] });
    loadFailed = false;
    setStatus("Model loaded. Choose a sample or upload a WAV.");
    setAnalyzeEnabled();
  } catch (error) {
    loadFailed = true;
    session = null;
    setAnalyzeEnabled();
    setStatus(`Could not load the model: ${error.message}`);
  }
}

document.querySelectorAll(".sample").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".sample").forEach((item) => item.classList.remove("is-selected"));
    button.classList.add("is-selected");
    stageUrl(button.dataset.src).catch((error) => setStatus(error.message));
  });
});

analyzeBtn.addEventListener("click", () => {
  if (inferBusy) return;
  if (!pendingSamples) {
    setStatus("Choose a recording before analysis.");
    return;
  }
  infer(pendingSamples).catch((error) => {
    setStatus(error.message);
  });
});

resetBtn.addEventListener("click", resetDemo);

document.querySelector("#browse").addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  const file = event.dataTransfer.files[0];
  if (file) stageBlob(file, file.name).catch((error) => setStatus(error.message));
});
dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) stageBlob(file, file.name).catch((error) => setStatus(error.message));
});

const navToggle = document.querySelector("#nav-toggle");
const nav = document.querySelector("#site-nav");
if (navToggle && nav) {
  navToggle.addEventListener("click", () => {
    const open = nav.classList.toggle("is-open");
    navToggle.setAttribute("aria-expanded", open ? "true" : "false");
  });
  nav.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      nav.classList.remove("is-open");
      navToggle.setAttribute("aria-expanded", "false");
    });
  });
}

const sections = [...document.querySelectorAll("main section[id]")];
const navLinks = [...document.querySelectorAll("#site-nav a[href^='#']")];
if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      navLinks.forEach((link) => {
        link.classList.toggle("is-active", link.getAttribute("href") === `#${entry.target.id}`);
      });
    });
  }, { rootMargin: "-40% 0px -50% 0px", threshold: 0.01 });
  sections.forEach((section) => observer.observe(section));
}

boot();
