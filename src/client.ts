const $ = <T extends Element>(s: string) => document.querySelector<T>(s)!;

const TARGET_RATE = 24000;
const WINDOW_COLORS = ["#ffd166", "#7ae582", "#6ec5ff", "#ff8fab", "#c3a6ff", "#ffb86b"];

const button = $("#record");
const status = $("#status");
const windowsInput = $("#windows") as HTMLTextAreaElement;
const lyricsInput = $("#lyrics") as HTMLTextAreaElement;
const metricInput = $("#metric") as HTMLSelectElement;
const focusSizeInput = $("#focus-size") as HTMLInputElement;
const futureSizeInput = $("#future-size") as HTMLInputElement;
const windowsOut = $("#windows-out");
const focusOut = $("#focus-out");
const lyricsOut = $("#lyrics-out");

type Metric = "jaro-winkler" | "levenshtein" | "dice";

type WindowConfig = {
  id: string;
  label: string;
  periodMs: number;
  lengthMs: number;
  color: string;
};

type WindowView = {
  root: HTMLElement;
  meta: HTMLElement;
  partial: HTMLElement;
  final: HTMLElement;
  match: HTMLElement;
};

type MatchResult = {
  lineIndex: number;
  score: number;
  line: string;
};

type WindowState = {
  cfg: WindowConfig;
  view: WindowView;
  ws: WebSocket | null;
  nextAt: number;
  pending: boolean;
  rerun: boolean;
  partial: string;
  finalText: string;
  connected: boolean;
  match: MatchResult | null;
};

type FocusSpan = {
  start: number;
  end: number;
  totalDistance: number;
  penalizedDistance: number;
};

type LyricLine = {
  text: string;
  norm: string;
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
let lyricLines: LyricLine[] = [];
let lastFocusSpan: FocusSpan | null = null;

button.addEventListener("click", async () => {
  try {
    if (stream) return stop();
    await start();
  } catch (error) {
    stop();
    status.textContent = error instanceof Error ? error.message : "failed";
  }
});

lyricsInput.addEventListener("input", refreshLyrics);
metricInput.addEventListener("change", updateMatches);
focusSizeInput.addEventListener("input", updateMatches);
futureSizeInput.addEventListener("input", updateMatches);

refreshLyrics();

async function start() {
  const configs = parseWindows(windowsInput.value);
  if (!configs.length) throw new Error("Add at least one window");

  windowsInput.disabled = true;
  windowsOut.innerHTML = "";
  windows = configs.map((cfg) => ({
    cfg,
    view: renderWindow(cfg),
    ws: null,
    nextAt: 0,
    pending: false,
    rerun: false,
    partial: "",
    finalText: "",
    connected: false,
    match: null
  }));
  updateMatches();

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

  for (const windowState of windows) windowState.nextAt = performance.now() + windowState.cfg.periodMs;
  ticker = window.setInterval(tick, 100);
  button.textContent = "stop";
  status.textContent = "rolling";
}

function stop() {
  window.clearInterval(ticker);
  for (const windowState of windows) windowState.ws?.close();
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
  lastFocusSpan = null;
  windowsInput.disabled = false;
  button.textContent = "record";
  status.textContent = "idle";
  refreshFocusView(null);
  refreshLyricsView();
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
    windowState.finalText = msg.transcript || windowState.partial;
    windowState.view.final.textContent = windowState.finalText;
    windowState.view.partial.textContent = "";
    windowState.partial = "";
    windowState.pending = false;
    windowState.view.meta.textContent = describe(windowState.cfg, "ready");
    updateMatch(windowState);
    const span = findBestFocusSpan();
    lastFocusSpan = span;
    refreshFocusView(span);
    refreshLyricsView();
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
  root.style.setProperty("--window-color", cfg.color);
  const title = document.createElement("h2");
  title.textContent = cfg.label;
  const meta = document.createElement("p");
  meta.className = "meta";
  meta.textContent = describe(cfg, "booting");
  const match = document.createElement("p");
  match.className = "match";
  match.textContent = "No lyric match yet.";
  const partial = document.createElement("p");
  partial.className = "text partial";
  const final = document.createElement("pre");
  final.className = "text";
  root.append(title, meta, match, partial, final);
  windowsOut.append(root);
  return { root, meta, partial, final, match };
}

function refreshLyrics() {
  lyricLines = lyricsInput.value
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean)
    .map((text) => ({ text, norm: normalize(text) }));
  updateMatches();
}

function updateMatches() {
  for (const windowState of windows) updateMatch(windowState);
  const span = findBestFocusSpan();
  lastFocusSpan = span;
  refreshFocusView(span);
  refreshLyricsView();
}

function updateMatch(windowState: WindowState) {
  const transcript = normalize(windowState.finalText);
  if (!transcript || !lyricLines.length) {
    windowState.match = null;
    windowState.view.root.classList.remove("active");
    windowState.view.match.textContent = "No lyric match yet.";
    return;
  }

  let best: MatchResult | null = null;
  for (let i = 0; i < lyricLines.length; i += 1) {
    const line = lyricLines[i];
    const score = scoreText(transcript, line.norm, metricInput.value as Metric);
    if (!best || score > best.score) {
      best = { lineIndex: i, score, line: line.text };
    }
  }

  windowState.match = best;
  windowState.view.root.classList.add("active");
  windowState.view.match.textContent = best
    ? `${best.line}\n${metricInput.value} ${(best.score * 100).toFixed(1)}%`
    : "No lyric match yet.";
}

function refreshLyricsView() {
  lyricsOut.innerHTML = "";
  for (let i = 0; i < lyricLines.length; i += 1) {
    const line = lyricLines[i];
    const row = document.createElement("div");
    row.className = "lyric-line";
    row.textContent = line.text;

    const matches = windows.filter((windowState) => windowState.match?.lineIndex === i);
    if (matches.length) {
      row.classList.add("active");
      row.style.setProperty("--window-color", matches[0].cfg.color);
      const badges = document.createElement("div");
      badges.className = "lyric-badges";
      for (const windowState of matches) {
        const badge = document.createElement("span");
        badge.className = "lyric-badge";
        badge.style.setProperty("--window-color", windowState.cfg.color);
        badge.textContent = windowState.cfg.label;
        badges.append(badge);
      }
      row.append(badges);
    }

    lyricsOut.append(row);
  }
}

function findBestFocusSpan() {
  const activeWindows = windows.filter((windowState) => normalize(windowState.finalText));
  const focusSize = readFocusSize();
  if (!lyricLines.length || !activeWindows.length || !focusSize) return null;

  const distances = activeWindows.map((windowState) =>
    lyricLines.map((line) => distanceText(normalize(windowState.finalText), line.norm, metricInput.value as Metric))
  );

  let best: FocusSpan | null = null;
  const maxStart = Math.max(0, lyricLines.length - focusSize);
  for (let start = 0; start <= maxStart; start += 1) {
    const end = Math.min(lyricLines.length, start + focusSize);
    let totalDistance = 0;

    for (let row = 0; row < distances.length; row += 1) {
      let minDistance = Number.POSITIVE_INFINITY;
      for (let col = start; col < end; col += 1) {
        minDistance = Math.min(minDistance, distances[row][col]);
      }
      totalDistance += minDistance;
    }

    const penalizedDistance =
      lastFocusSpan && Math.abs(start - lastFocusSpan.start) > 1
        ? totalDistance * 2
        : totalDistance;

    if (!best || penalizedDistance < best.penalizedDistance) {
      best = { start, end, totalDistance, penalizedDistance };
    }
  }

  return best;
}

function refreshFocusView(span: FocusSpan | null) {
  focusOut.innerHTML = "";
  const title = document.createElement("p");
  title.className = "focus-title";
  title.textContent = "focus window";
  focusOut.append(title);

  if (!span) {
    const empty = document.createElement("p");
    empty.className = "focus-meta";
    empty.textContent = "No aggregate lyric window yet.";
    focusOut.append(empty);
    return;
  }

  const meta = document.createElement("p");
  meta.className = "focus-meta";
  const futureSize = readFutureSize();
  const shownEnd = Math.min(lyricLines.length, span.end + futureSize);
  meta.textContent = `${span.end - span.start} focused + ${shownEnd - span.end} future · distance ${span.totalDistance.toFixed(3)} · penalized ${span.penalizedDistance.toFixed(3)}`;
  focusOut.append(meta);

  for (let i = span.start; i < shownEnd; i += 1) {
    const line = document.createElement("div");
    line.className = "focus-line";
    if (i >= span.end) line.classList.add("future");
    line.textContent = lyricLines[i].text;
    focusOut.append(line);
  }
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
        lengthMs: numbers[1],
        color: WINDOW_COLORS[index % WINDOW_COLORS.length]
      };
    });
}

function describe(cfg: WindowConfig, stateText: string) {
  return `${cfg.periodMs}ms every ${cfg.lengthMs}ms window · ${stateText}`;
}

function normalize(text: string) {
  return text
    .toLowerCase()
    .replace(/['’]/g, "")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function scoreText(a: string, b: string, metric: Metric) {
  if (!a || !b) return 0;
  if (metric === "levenshtein") return 1 - levenshtein(a, b) / Math.max(a.length, b.length, 1);
  if (metric === "dice") return dice(a, b);
  return jaroWinkler(a, b);
}

function distanceText(a: string, b: string, metric: Metric) {
  return 1 - scoreText(a, b, metric);
}

function readFocusSize() {
  const value = Number(focusSizeInput.value);
  if (!Number.isFinite(value) || value < 1) return 0;
  return Math.max(1, Math.floor(value));
}

function readFutureSize() {
  const value = Number(futureSizeInput.value);
  if (!Number.isFinite(value) || value < 0) return 0;
  return Math.max(0, Math.floor(value));
}

function levenshtein(a: string, b: string) {
  const prev = new Array(b.length + 1).fill(0).map((_, i) => i);
  const next = new Array(b.length + 1).fill(0);
  for (let i = 1; i <= a.length; i += 1) {
    next[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      next[j] = Math.min(next[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost);
    }
    for (let j = 0; j <= b.length; j += 1) prev[j] = next[j];
  }
  return prev[b.length];
}

function dice(a: string, b: string) {
  const left = bigrams(a);
  const right = bigrams(b);
  if (!left.size || !right.size) return a === b ? 1 : 0;

  let shared = 0;
  for (const [gram, count] of left) shared += Math.min(count, right.get(gram) || 0);
  return (2 * shared) / (countMap(left) + countMap(right));
}

function bigrams(text: string) {
  const grams = new Map<string, number>();
  const clean = text.replace(/\s+/g, " ");
  for (let i = 0; i < clean.length - 1; i += 1) {
    const gram = clean.slice(i, i + 2);
    grams.set(gram, (grams.get(gram) || 0) + 1);
  }
  return grams;
}

function countMap(map: Map<string, number>) {
  let total = 0;
  for (const value of map.values()) total += value;
  return total;
}

function jaroWinkler(a: string, b: string) {
  if (a === b) return 1;
  const matchDistance = Math.max(Math.floor(Math.max(a.length, b.length) / 2) - 1, 0);
  const aMatches = new Array(a.length).fill(false);
  const bMatches = new Array(b.length).fill(false);
  let matches = 0;

  for (let i = 0; i < a.length; i += 1) {
    const start = Math.max(0, i - matchDistance);
    const end = Math.min(i + matchDistance + 1, b.length);
    for (let j = start; j < end; j += 1) {
      if (bMatches[j] || a[i] !== b[j]) continue;
      aMatches[i] = true;
      bMatches[j] = true;
      matches += 1;
      break;
    }
  }

  if (!matches) return 0;

  let transpositions = 0;
  let k = 0;
  for (let i = 0; i < a.length; i += 1) {
    if (!aMatches[i]) continue;
    while (!bMatches[k]) k += 1;
    if (a[i] !== b[k]) transpositions += 1;
    k += 1;
  }

  const jaro = (
    matches / a.length +
    matches / b.length +
    (matches - transpositions / 2) / matches
  ) / 3;

  let prefix = 0;
  while (prefix < 4 && a[prefix] === b[prefix]) prefix += 1;
  return jaro + prefix * 0.1 * (1 - jaro);
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
