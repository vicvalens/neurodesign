const wrapper = document.getElementById("wrapper");
const stimulus = document.getElementById("stimulus");
const canvas = document.getElementById("heatmapCanvas");
const ctx = canvas.getContext("2d");

const startBtn = document.getElementById("startBtn");
const showBtn = document.getElementById("showBtn");
const clearBtn = document.getElementById("clearBtn");
const downloadBtn = document.getElementById("downloadBtn");

let tracking = false;
let rawData = [];

// Ajustes principales
const BASE_RADIUS = 34;
const EXTRA_RADIUS = 16;
const MIN_DT = 8;
const SPEED_LOW = 0.05;
const SPEED_HIGH = 1.2;
const STEP_SKIP = 1;

// Menos trayecto, más fijación
const TRAIL_WEIGHT = 0.00;
const SLOW_WEIGHT_BOOST = 0.9;
const STOP_WEIGHT_BOOST = 2.2;

function resizeCanvas() {
  const w = stimulus.clientWidth;
  const h = stimulus.clientHeight;

  if (!w || !h) return;

  canvas.width = w;
  canvas.height = h;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
}

function clearHeatmap() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function getRelativePosition(event) {
  const rect = stimulus.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  if (x < 0 || y < 0 || x > rect.width || y > rect.height) return null;
  return { x, y };
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function colorAt(t) {
  const stops = [
    { t: 0.00, c: [0, 0, 0, 0] },
    { t: 0.08, c: [35, 60, 255, 38] },
    { t: 0.22, c: [0, 190, 255, 56] },
    { t: 0.38, c: [0, 255, 170, 72] },
    { t: 0.54, c: [60, 255, 60, 88] },
    { t: 0.72, c: [255, 235, 0, 104] },
    { t: 0.88, c: [255, 130, 0, 118] },
    { t: 1.00, c: [255, 0, 0, 130] }
  ];

  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i];
    const b = stops[i + 1];
    if (t >= a.t && t <= b.t) {
      const lt = (t - a.t) / (b.t - a.t);
      return [
        Math.round(lerp(a.c[0], b.c[0], lt)),
        Math.round(lerp(a.c[1], b.c[1], lt)),
        Math.round(lerp(a.c[2], b.c[2], lt)),
        Math.round(lerp(a.c[3], b.c[3], lt))
      ];
    }
  }

  return stops[stops.length - 1].c;
}

function createAlphaStamp(radius, alphaStrength = 1.0) {
  const stamp = document.createElement("canvas");
  const size = radius * 2;
  stamp.width = size;
  stamp.height = size;

  const sctx = stamp.getContext("2d");
  const gradient = sctx.createRadialGradient(radius, radius, 0, radius, radius, radius);

  gradient.addColorStop(0.0, `rgba(0,0,0,${0.11 * alphaStrength})`);
  gradient.addColorStop(0.2, `rgba(0,0,0,${0.08 * alphaStrength})`);
  gradient.addColorStop(0.45, `rgba(0,0,0,${0.05 * alphaStrength})`);
  gradient.addColorStop(0.75, `rgba(0,0,0,${0.025 * alphaStrength})`);
  gradient.addColorStop(1.0, "rgba(0,0,0,0)");

  sctx.fillStyle = gradient;
  sctx.fillRect(0, 0, size, size);

  return stamp;
}

function buildWeightedAttentionPoints(data) {
  if (data.length < 2) return [];

  const weighted = [];

  for (let i = 1; i < data.length; i += STEP_SKIP) {
    const prev = data[i - 1];
    const curr = data[i];

    const dx = curr.x - prev.x;
    const dy = curr.y - prev.y;
    const dist = Math.hypot(dx, dy);
    const dt = Math.max(MIN_DT, curr.t - prev.t);
    const speed = dist / dt;

    const speedNorm = clamp((speed - SPEED_LOW) / (SPEED_HIGH - SPEED_LOW), 0, 1);
    const slowFactor = 1 - speedNorm;

    let weight = 0;

    // Solo puntos lentos
    if (speed < 0.20) {
      weight = TRAIL_WEIGHT + slowFactor * SLOW_WEIGHT_BOOST;
    }

    // Casi quieto
    if (speed < 0.08) {
      weight += STOP_WEIGHT_BOOST;
    }

    weight *= clamp(dt / 16, 0.7, 2.5);

    // Descarta trayecto residual
    if (weight < 0.08) continue;

    weighted.push({
      x: curr.x,
      y: curr.y,
      weight,
      speed,
      dt
    });
  }

  return weighted;
}

function renderContinuousHeatmap(weightedPoints) {
  clearHeatmap();
  if (!weightedPoints.length) return;

  const off = document.createElement("canvas");
  off.width = canvas.width;
  off.height = canvas.height;
  const offCtx = off.getContext("2d");

  const maxWeight = Math.max(...weightedPoints.map(p => p.weight), 1);

  weightedPoints.forEach((p) => {
    const ratio = p.weight / maxWeight;
    const radius = Math.round(BASE_RADIUS + ratio * EXTRA_RADIUS);
    const alphaStrength = 0.35 + ratio * 1.2;

    const stamp = createAlphaStamp(radius, alphaStrength);
    offCtx.drawImage(stamp, p.x - radius, p.y - radius);

    if (ratio > 0.55) {
      const coreRadius = Math.round(radius * 0.38);
      const coreStamp = createAlphaStamp(coreRadius, 0.9 + ratio * 1.2);
      offCtx.drawImage(coreStamp, p.x - coreRadius, p.y - coreRadius);
    }
  });

  const imageData = offCtx.getImageData(0, 0, off.width, off.height);
  const data = imageData.data;

  let maxAlpha = 1;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] > maxAlpha) maxAlpha = data[i];
  }

  for (let i = 0; i < data.length; i += 4) {
    const alpha = data[i + 3];
    if (alpha === 0) continue;

    let t = alpha / maxAlpha;

    // Mantiene colores medios visibles
    t = Math.pow(t, 0.68);

    const [r, g, b, a] = colorAt(t);
    data[i] = r;
    data[i + 1] = g;
    data[i + 2] = b;
    data[i + 3] = a;
  }

  ctx.putImageData(imageData, 0, 0);
}

stimulus.addEventListener("load", resizeCanvas);
window.addEventListener("resize", resizeCanvas);

wrapper.addEventListener("mousemove", (event) => {
  if (!tracking) return;

  const pos = getRelativePosition(event);
  if (!pos) return;

  rawData.push({
    x: Math.round(pos.x),
    y: Math.round(pos.y),
    t: performance.now()
  });
});

startBtn.addEventListener("click", () => {
  tracking = true;
  rawData = [];
  clearHeatmap();
  alert("Registro iniciado");
});

showBtn.addEventListener("click", () => {
  if (rawData.length < 2) {
    alert("No hay suficientes datos registrados.");
    return;
  }

  const weighted = buildWeightedAttentionPoints(rawData);
  renderContinuousHeatmap(weighted);
});

clearBtn.addEventListener("click", () => {
  tracking = false;
  rawData = [];
  clearHeatmap();
});

downloadBtn.addEventListener("click", () => {
  const weighted = buildWeightedAttentionPoints(rawData);

  const data = {
    image: stimulus.getAttribute("src"),
    timestamp: new Date().toISOString(),
    raw_mouse_data: rawData,
    weighted_attention_points: weighted
  };

  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json"
  });

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "mouse_continuous_heatmap_data.json";
  a.click();
  URL.revokeObjectURL(url);
});

if (stimulus.complete) {
  resizeCanvas();
}