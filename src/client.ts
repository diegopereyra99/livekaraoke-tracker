const $ = <T extends Element>(s: string) => document.querySelector<T>(s)!;

const TARGET_RATE = 24000;
const button = $("#record");
const status = $("#status");
const windowsInput = $("#windows") as HTMLTextAreaElement;
const windowsOut = $("#windows-out");

type WindowConfig = {
  id: string;
  label: string;
  periodMs: number;
  lengthMs: number;
};

type WindowView = {
  root: HTMLElement;
  meta: HTMLElement;
  partial: HTMLElement;
  final: HTMLElement;
};

type WindowState = {
  cfg: WindowConfig;
  view: WindowView;
  ws: WebSocket | null;
  nextAt: number;
  pending: boolean;
  rerun: boolean;
  partial: string;
  connected: boolean;
};

class SampleBuffer {
  private chunks: Float32Array[] = [];
  private total = 0;

  constructor(private readonly maxSamples: number) {}

  append(chunk: Float32Array) {
    if (!chunk.length) return;
    this.chunks.push(chunk);
    this.total += chunk.length;
    while (this.total > this.maxSamples && this.chunks.length > 1) {
      const dropped = this.chunks.shift()!;
      this.total -= dropped.length;
    }
  }

  latest(samples: number) {
    if (this.total < samples) return null;
    const out = new Float32Array(samples);
    let write = samples;

    for (let i = this.chunks.length - 1; i >= 0 && write > 0; i -= 1) {
      const chunk = this.chunks[i];
      const take = Math.min(write, chunk.length);
      out.set(chunk.subarray(chunk.length - take), write - take);
      write -= take;
    }

    return out;
  }
}

let stream: MediaStream | null = null;
let context: AudioContext | null = null;
let source: MediaStreamAudioSourceNode | null = null;
let processor: ScriptProcessorNode | null = null;
let sink: GainNode | null = null;
let ticker = 0;
let sampleBuffer: SampleBuffer | null = null;
let startedAt = 0;
let windows: WindowState[] = [];

button.addEventListener("click", async () => {
  try {
    if (stream) return stop();
    await start();
  } catch (error) {
    stop();
    status.textContent = error instanceof Error ? error.message : "failed";
  }
});

async function start() {
  const configs = parseWindows(windowsInput.value);
  if (!configs.length) throw new Error("Add at least one window");

  windowsInput.disabled = true;
  windowsOut.innerHTML = "";
  windows = configs.map((cfg) => ({ cfg, view: renderWindow(cfg), ws: null, nextAt: 0, pending: false, rerun: false, partial: "", connected: false }));

  status.textContent = "asking for mic";
  stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  context = new AudioContext();
  await context.resume();

  const maxLengthMs = Math.max(...configs.map((cfg) => cfg.lengthMs));
  sampleBuffer = new SampleBuffer(Math.ceil((maxLengthMs + 2000) * TARGET_RATE / 1000));
  startedAt = performance.now();

  source = context.createMediaStreamSource(stream);
  processor = context.createScriptProcessor(4096, 1, 1);
  sink = context.createGain();
  sink.gain.value = 0;
  processor.onaudioprocess = (event) => {
    if (!context || !sampleBuffer) return;
    const input = event.inputBuffer.getChannelData(0);
    sampleBuffer.append(downsample(input, context.sampleRate, TARGET_RATE));
  };
  source.connect(processor);
  processor.connect(sink);
  sink.connect(context.destination);

  status.textContent = "opening windows";
  await Promise.all(windows.map(openWindow));

  for (const window of windows) window.nextAt = performance.now() + window.cfg.periodMs;
  ticker = window.setInterval(tick, 100);
  button.textContent = "stop";
  status.textContent = "rolling";
}

function stop() {
  window.clearInterval(ticker);
  for (const window of windows) window.ws?.close();
  windows = [];
  stream?.getTracks().forEach((track) => track.stop());
  processor?.disconnect();
  source?.disconnect();
  sink?.disconnect();
  context?.close();
  stream = null;
  processor = null;
  source = null;
  sink = null;
  context = null;
  sampleBuffer = null;
  windowsInput.disabled = false;
  button.textContent = "record";
  status.textContent = "idle";
}

async function openWindow(windowState: WindowState) {
  const sessionRes = await fetch("/session", { method: "POST" });
  const sessionJson = await sessionRes.json();
  const token = sessionJson.client_secret?.value || sessionJson.value;
  if (!sessionRes.ok || !token) throw new Error(sessionJson.error?.message || "session request failed");

  await new Promise<void>((resolve, reject) => {
    const ws = new WebSocket("wss://api.openai.com/v1/realtime?intent=transcription", [
      "realtime",
      `openai-insecure-api-key.${token}`
    ]);

    ws.addEventListener("open", () => {
      ws.send(JSON.stringify({
        type: "transcription_session.update",
        input_audio_format: "pcm16",
        input_audio_transcription: {
          model: "gpt-4o-transcribe",
          language: "en",
          prompt:
            "Transcribe sung songs and spoken words. Prefer accurate lyrics over paraphrase. Keep repeated choruses, partial words, and line breaks when they are clearly heard. Do not answer, summarize, or explain."
        },
        turn_detection: null,
        input_audio_noise_reduction: {
          type: "far_field"
        }
      }));
      windowState.ws = ws;
      windowState.connected = true;
      windowState.view.meta.textContent = describe(windowState.cfg, "ready");
      resolve();
    });

    ws.addEventListener("message", (event) => onSocketMessage(windowState, event));
    ws.addEventListener("error", () => reject(new Error(`socket failed for ${windowState.cfg.label}`)));
    ws.addEventListener("close", () => {
      windowState.connected = false;
      if (stream) windowState.view.meta.textContent = describe(windowState.cfg, "closed");
    });
  });
}

function onSocketMessage(windowState: WindowState, event: MessageEvent<string>) {
  const msg = JSON.parse(event.data) as {
    type: string;
    delta?: string;
    transcript?: string;
    error?: { message?: string };
  };

  if (msg.type === "conversation.item.input_audio_transcription.delta") {
    windowState.partial += msg.delta || "";
    windowState.view.partial.textContent = windowState.partial;
    return;
  }

  if (msg.type === "conversation.item.input_audio_transcription.completed") {
    windowState.view.final.textContent = msg.transcript || windowState.partial;
    windowState.view.partial.textContent = "";
    windowState.partial = "";
    windowState.pending = false;
    windowState.view.meta.textContent = describe(windowState.cfg, "ready");
    if (windowState.rerun) {
      windowState.rerun = false;
      queueWindow(windowState);
    }
    return;
  }

  if (msg.type === "error") {
    windowState.pending = false;
    windowState.view.meta.textContent = describe(windowState.cfg, msg.error?.message || "error");
  }
}

function tick() {
  const now = performance.now();
  for (const windowState of windows) {
    if (now < windowState.nextAt) continue;
    while (windowState.nextAt <= now) windowState.nextAt += windowState.cfg.periodMs;
    queueWindow(windowState);
  }
}

function queueWindow(windowState: WindowState) {
  if (!sampleBuffer || !windowState.connected) return;
  if (performance.now() - startedAt < windowState.cfg.lengthMs) return;
  if (windowState.pending) {
    windowState.rerun = true;
    windowState.view.meta.textContent = describe(windowState.cfg, "waiting");
    return;
  }

  const samples = sampleBuffer.latest(msToSamples(windowState.cfg.lengthMs));
  if (!samples || !windowState.ws) return;

  windowState.pending = true;
  windowState.partial = "";
  windowState.view.partial.textContent = "";
  windowState.view.meta.textContent = describe(windowState.cfg, "sending");
  windowState.ws.send(JSON.stringify({ type: "input_audio_buffer.clear" }));
  windowState.ws.send(JSON.stringify({
    type: "input_audio_buffer.append",
    audio: toBase64Pcm16(samples)
  }));
  windowState.ws.send(JSON.stringify({ type: "input_audio_buffer.commit" }));
}

function renderWindow(cfg: WindowConfig) {
  const root = document.createElement("article");
  root.className = "window";
  const title = document.createElement("h2");
  title.textContent = cfg.label;
  const meta = document.createElement("p");
  meta.className = "meta";
  meta.textContent = describe(cfg, "booting");
  const partial = document.createElement("p");
  partial.className = "text partial";
  const final = document.createElement("pre");
  final.className = "text";
  root.append(title, meta, partial, final);
  windowsOut.append(root);
  return { root, meta, partial, final };
}

function parseWindows(text: string) {
  return text
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      const parts = line.replaceAll(",", " ").split(/\s+/).filter(Boolean);
      if (parts.length < 2) throw new Error(`Bad window line: ${line}`);
      const numbers = parts.slice(-2).map(Number);
      if (numbers.some((value) => !Number.isFinite(value) || value <= 0)) {
        throw new Error(`Bad window numbers: ${line}`);
      }
      return {
        id: `w${index + 1}`,
        label: parts.length > 2 ? parts.slice(0, -2).join(" ") : `w${index + 1}`,
        periodMs: numbers[0],
        lengthMs: numbers[1]
      };
    });
}

function describe(cfg: WindowConfig, stateText: string) {
  return `${cfg.periodMs}ms every ${cfg.lengthMs}ms window · ${stateText}`;
}

function msToSamples(ms: number) {
  return Math.round(ms * TARGET_RATE / 1000);
}

function downsample(input: Float32Array, fromRate: number, toRate: number) {
  if (fromRate === toRate) return new Float32Array(input);
  const ratio = fromRate / toRate;
  const length = Math.max(1, Math.round(input.length / ratio));
  const output = new Float32Array(length);

  for (let i = 0; i < length; i += 1) {
    const start = Math.floor(i * ratio);
    const end = Math.min(input.length, Math.floor((i + 1) * ratio));
    let sum = 0;
    let count = 0;
    for (let j = start; j < end; j += 1) {
      sum += input[j];
      count += 1;
    }
    output[i] = count ? sum / count : input[start] || 0;
  }

  return output;
}

function toBase64Pcm16(samples: Float32Array) {
  const bytes = new Uint8Array(samples.length * 2);
  const view = new DataView(bytes.buffer);

  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }

  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}
