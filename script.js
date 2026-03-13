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

// Ajustes visuales y de agregación
const GRID_SIZE = 14;   // menor = más concentración visual
const RADIUS = 24;      // menor = manchas más marcadas

function resizeCanvas() {
  const w = stimulus.clientWidth;
  const h = stimulus.clientHeight;

  if (!w || !h) return;

  canvas.width = w;
  canvas.height = h;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
}

function getRelativePosition(event) {
  const rect = stimulus.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
    return null;
  }

  return { x, y };
}

function aggregatePoints(data, gridSize = GRID_SIZE) {
  const map = {};

  data.forEach((p) => {
    const gx = Math.round(p.x / gridSize) * gridSize;
    const gy = Math.round(p.y / gridSize) * gridSize;
    const key = `${gx}_${gy}`;

    if (!map[key]) {
      map[key] = { x: gx, y: gy, value: 0 };
    }

    map[key].value += 1;
  });

  return Object.values(map);
}

function getColorsForRatio(ratio) {
  if (ratio < 0.25) {
    return {
      center: "rgba(0, 0, 255, 0.70)",
      mid: "rgba(0, 0, 255, 0.35)"
    };
  }

  if (ratio < 0.5) {
    return {
      center: "rgba(0, 255, 255, 0.75)",
      mid: "rgba(0, 255, 255, 0.38)"
    };
  }

  if (ratio < 0.7) {
    return {
      center: "rgba(0, 255, 0, 0.80)",
      mid: "rgba(0, 255, 0, 0.42)"
    };
  }

  if (ratio < 0.85) {
    return {
      center: "rgba(255, 255, 0, 0.88)",
      mid: "rgba(255, 255, 0, 0.48)"
    };
  }

  return {
    center: "rgba(255, 0, 0, 0.95)",
    mid: "rgba(255, 0, 0, 0.55)"
  };
}

function drawHeatmap(points) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!points.length) return;

  const maxValue = Math.max(...points.map((p) => p.value), 1);

  points.forEach((p) => {
    const ratio = p.value / maxValue;
    const colors = getColorsForRatio(ratio);

    const gradient = ctx.createRadialGradient(
      p.x, p.y, 0,
      p.x, p.y, RADIUS
    );

    gradient.addColorStop(0, colors.center);
    gradient.addColorStop(0.45, colors.mid);
    gradient.addColorStop(1, "rgba(255, 255, 255, 0)");

    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(p.x, p.y, RADIUS, 0, Math.PI * 2);
    ctx.fill();
  });
}

function clearHeatmap() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
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
  if (!rawData.length) {
    alert("No hay datos registrados.");
    return;
  }

  const aggregated = aggregatePoints(rawData, GRID_SIZE);
  drawHeatmap(aggregated);
});

clearBtn.addEventListener("click", () => {
  tracking = false;
  rawData = [];
  clearHeatmap();
});

downloadBtn.addEventListener("click", () => {
  const data = {
    image: stimulus.getAttribute("src"),
    timestamp: new Date().toISOString(),
    raw_mouse_data: rawData
  };

  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json"
  });

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "mouse_heatmap_data.json";
  a.click();
  URL.revokeObjectURL(url);
});

if (stimulus.complete) {
  resizeCanvas();
}