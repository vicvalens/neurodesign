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
let weightedPoints = [];

let lastPoint = null;
let clusterStartTime = null;
let clusterPoints = [];

/*
  AJUSTES
*/
const FIXATION_RADIUS = 34;     // radio espacial para agrupar permanencia
const MIN_FIXATION_MS = 90;     // mínimo para aceptar microfijaciones
const MERGE_RADIUS = 42;        // fusiona fijaciones cercanas
const BASE_HEAT_RADIUS = 85;    // tamaño base del heatmap visual
const MAX_EXTRA_RADIUS = 45;    // expansión extra según intensidad

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

function distance(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function averagePoint(points) {
  const sum = points.reduce(
    (acc, p) => {
      acc.x += p.x;
      acc.y += p.y;
      return acc;
    },
    { x: 0, y: 0 }
  );

  return {
    x: sum.x / points.length,
    y: sum.y / points.length
  };
}

function finalizeCluster(endTime) {
  if (!clusterPoints.length || clusterStartTime === null) return;

  const duration = endTime - clusterStartTime;
  const center = averagePoint(clusterPoints);

  if (duration >= MIN_FIXATION_MS) {
    weightedPoints.push({
      x: Math.round(center.x),
      y: Math.round(center.y),
      duration,
      samples: clusterPoints.length
    });
  }

  clusterPoints = [];
  clusterStartTime = null;
}

function mergeNearbyFixations(points, mergeRadius = MERGE_RADIUS) {
  const merged = [];

  points.forEach((fix) => {
    const existing = merged.find((m) => {
      const d = Math.hypot(m.x - fix.x, m.y - fix.y);
      return d <= mergeRadius;
    });

    if (existing) {
      const totalWeight = existing.weight + fix.duration;
      existing.x = Math.round(
        (existing.x * existing.weight + fix.x * fix.duration) / totalWeight
      );
      existing.y = Math.round(
        (existing.y * existing.weight + fix.y * fix.duration) / totalWeight
      );
      existing.weight += fix.duration;
      existing.samples += fix.samples || 1;
    } else {
      merged.push({
        x: fix.x,
        y: fix.y,
        weight: fix.duration,
        samples: fix.samples || 1
      });
    }
  });

  return merged;
}

function createAlphaStamp(radius, alphaStrength = 1.0) {
  const stamp = document.createElement("canvas");
  const size = radius * 2;
  stamp.width = size;
  stamp.height = size;

  const sctx = stamp.getContext("2d");
  const gradient = sctx.createRadialGradient(radius, radius, 0, radius, radius, radius);

  gradient.addColorStop(0.0, `rgba(0,0,0,${0.28 * alphaStrength})`);
  gradient.addColorStop(0.2, `rgba(0,0,0,${0.22 * alphaStrength})`);
  gradient.addColorStop(0.45, `rgba(0,0,0,${0.14 * alphaStrength})`);
  gradient.addColorStop(0.7, `rgba(0,0,0,${0.07 * alphaStrength})`);
  gradient.addColorStop(1.0, "rgba(0,0,0,0)");

  sctx.fillStyle = gradient;
  sctx.fillRect(0, 0, size, size);

  return stamp;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function colorAt(t) {
  const stops = [
    { t: 0.00, c: [0, 0, 0, 0] },
    { t: 0.08, c: [0, 60, 255, 70] },
    { t: 0.22, c: [0, 160, 255, 110] },
    { t: 0.40, c: [0, 255, 180, 145] },
    { t: 0.58, c: [120, 255, 0, 170] },
    { t: 0.74, c: [255, 230, 0, 200] },
    { t: 0.88, c: [255, 120, 0, 220] },
    { t: 1.00, c: [255, 0, 0, 235] }
  ];

  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i];
    const b = stops[i + 1];
    if (t >= a.t && t <= b.t) {
      const localT = (t - a.t) / (b.t - a.t);
      return [
        Math.round(lerp(a.c[0], b.c[0], localT)),
        Math.round(lerp(a.c[1], b.c[1], localT)),
        Math.round(lerp(a.c[2], b.c[2], localT)),
        Math.round(lerp(a.c[3], b.c[3], localT))
      ];
    }
  }

  return stops[stops.length - 1].c;
}

function renderProfessionalHeatmap(fixations) {
  clearHeatmap();
  if (!fixations.length) return;

  const off = document.createElement("canvas");
  off.width = canvas.width;
  off.height = canvas.height;
  const offCtx = off.getContext("2d");

  const maxWeight = Math.max(...fixations.map((f) => f.weight), 1);

  fixations.forEach((f) => {
    const ratio = f.weight / maxWeight;

    const radius = Math.round(
      BASE_HEAT_RADIUS + Math.min(MAX_EXTRA_RADIUS, ratio * MAX_EXTRA_RADIUS)
    );

    const alphaStrength = 0.7 + ratio * 1.6;
    const stamp = createAlphaStamp(radius, alphaStrength);

    offCtx.drawImage(stamp, f.x - radius, f.y - radius);

    // Refuerzo del núcleo para que haya centros más cálidos
    const coreRadius = Math.round(radius * 0.42);
    const coreStamp = createAlphaStamp(coreRadius, 1.2 + ratio * 2.0);
    offCtx.drawImage(coreStamp, f.x - coreRadius, f.y - coreRadius);
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

    // Curva para ampliar zonas medias y evitar solo manchas duras
    t = Math.pow(t, 0.78);

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

  const now = performance.now();

  rawData.push({
    x: Math.round(pos.x),
    y: Math.round(pos.y),
    t: now
  });

  if (!lastPoint) {
    lastPoint = pos;
    clusterStartTime = now;
    clusterPoints = [pos];
    return;
  }

  const d = distance(lastPoint, pos);

  if (d <= FIXATION_RADIUS) {
    clusterPoints.push(pos);
  } else {
    finalizeCluster(now);
    clusterStartTime = now;
    clusterPoints = [pos];
  }

  lastPoint = pos;
});

wrapper.addEventListener("mouseleave", () => {
  if (!tracking) return;
  finalizeCluster(performance.now());
  lastPoint = null;
});

startBtn.addEventListener("click", () => {
  tracking = true;
  rawData = [];
  weightedPoints = [];
  lastPoint = null;
  clusterStartTime = null;
  clusterPoints = [];
  clearHeatmap();
  alert("Registro iniciado");
});

showBtn.addEventListener("click", () => {
  if (!rawData.length) {
    alert("No hay datos registrados.");
    return;
  }

  finalizeCluster(performance.now());

  const merged = mergeNearbyFixations(weightedPoints, MERGE_RADIUS);

  if (!merged.length) {
    alert("No se detectaron zonas suficientes. Haz pausas cortas sobre distintas áreas.");
    return;
  }

  renderProfessionalHeatmap(merged);
});

clearBtn.addEventListener("click", () => {
  tracking = false;
  rawData = [];
  weightedPoints = [];
  lastPoint = null;
  clusterStartTime = null;
  clusterPoints = [];
  clearHeatmap();
});

downloadBtn.addEventListener("click", () => {
  finalizeCluster(performance.now());

  const merged = mergeNearbyFixations(weightedPoints, MERGE_RADIUS);

  const data = {
    image: stimulus.getAttribute("src"),
    timestamp: new Date().toISOString(),
    raw_mouse_data: rawData,
    fixation_like_points: weightedPoints,
    merged_attention_areas: merged
  };

  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json"
  });

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "mouse_professional_heatmap_data.json";
  a.click();
  URL.revokeObjectURL(url);
});

if (stimulus.complete) {
  resizeCanvas();
}