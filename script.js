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
let fixationCandidates = [];
let fixations = [];

let lastPoint = null;
let clusterStartTime = null;
let clusterPoints = [];

/*
  AJUSTES CLAVE
  ----------------
  FIXATION_RADIUS: qué tan cerca debe quedarse el cursor para considerarlo
  una misma fijación.
  MIN_FIXATION_MS: tiempo mínimo para aceptar una fijación.
  HEAT_RADIUS: tamaño visual de cada fijación en el mapa.
*/

const FIXATION_RADIUS = 35;
const MIN_FIXATION_MS = 220;
const HEAT_RADIUS = 55;

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

  fixationCandidates.push({
    points: [...clusterPoints],
    duration
  });

  if (duration >= MIN_FIXATION_MS) {
    const center = averagePoint(clusterPoints);

    fixations.push({
      x: Math.round(center.x),
      y: Math.round(center.y),
      duration
    });
  }

  clusterPoints = [];
  clusterStartTime = null;
}

function mergeNearbyFixations(fixations, mergeRadius = 45) {
  const merged = [];

  fixations.forEach((fix) => {
    const existing = merged.find((m) => {
      const d = Math.hypot(m.x - fix.x, m.y - fix.y);
      return d <= mergeRadius;
    });

    if (existing) {
      const totalWeight = existing.duration + fix.duration;
      existing.x = Math.round(
        (existing.x * existing.duration + fix.x * fix.duration) / totalWeight
      );
      existing.y = Math.round(
        (existing.y * existing.duration + fix.y * fix.duration) / totalWeight
      );
      existing.duration += fix.duration;
      existing.count += 1;
    } else {
      merged.push({
        x: fix.x,
        y: fix.y,
        duration: fix.duration,
        count: 1
      });
    }
  });

  return merged;
}

function getColorsForRatio(ratio) {
  if (ratio < 0.15) {
    return {
      center: "rgba(0, 90, 255, 0.30)",
      mid: "rgba(0, 90, 255, 0.16)"
    };
  }
  if (ratio < 0.35) {
    return {
      center: "rgba(0, 220, 255, 0.42)",
      mid: "rgba(0, 220, 255, 0.22)"
    };
  }
  if (ratio < 0.55) {
    return {
      center: "rgba(0, 255, 120, 0.52)",
      mid: "rgba(0, 255, 120, 0.28)"
    };
  }
  if (ratio < 0.75) {
    return {
      center: "rgba(255, 235, 0, 0.68)",
      mid: "rgba(255, 235, 0, 0.34)"
    };
  }
  return {
    center: "rgba(255, 60, 0, 0.86)",
    mid: "rgba(255, 60, 0, 0.44)"
  };
}

function drawBlob(x, y, radius, centerColor, midColor) {
  const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
  gradient.addColorStop(0, centerColor);
  gradient.addColorStop(0.42, midColor);
  gradient.addColorStop(1, "rgba(255,255,255,0)");

  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fill();
}

function drawHeatmapFromFixations(fixationData) {
  clearHeatmap();
  if (!fixationData.length) return;

  const maxDuration = Math.max(...fixationData.map((f) => f.duration), 1);

  fixationData.forEach((f) => {
    const ratio = f.duration / maxDuration;
    const colors = getColorsForRatio(ratio);

    const radius = HEAT_RADIUS + Math.min(18, ratio * 12);

    drawBlob(f.x, f.y, radius, colors.center, colors.mid);
    drawBlob(
      f.x,
      f.y,
      radius * 0.55,
      colors.center.replace(/0\.\d+\)/, "0.92)"),
      colors.mid.replace(/0\.\d+\)/, "0.52)")
    );
  });
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
  fixationCandidates = [];
  fixations = [];
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

  const mergedFixations = mergeNearbyFixations(fixations, 32);

  if (!mergedFixations.length) {
    alert("No se detectaron fijaciones suficientes. Muévete más lento o haz pausas más claras.");
    return;
  }

  drawHeatmapFromFixations(mergedFixations);
});

clearBtn.addEventListener("click", () => {
  tracking = false;
  rawData = [];
  fixationCandidates = [];
  fixations = [];
  lastPoint = null;
  clusterStartTime = null;
  clusterPoints = [];
  clearHeatmap();
});

downloadBtn.addEventListener("click", () => {
  finalizeCluster(performance.now());

  const data = {
    image: stimulus.getAttribute("src"),
    timestamp: new Date().toISOString(),
    raw_mouse_data: rawData,
    fixation_candidates: fixationCandidates,
    accepted_fixations: fixations,
    merged_fixations: mergeNearbyFixations(fixations, 50)
  };

  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json"
  });

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "mouse_fixation_heatmap_data.json";
  a.click();
  URL.revokeObjectURL(url);
});

if (stimulus.complete) {
  resizeCanvas();
}