const wrapper = document.getElementById("wrapper");
const stimulus = document.getElementById("stimulus");
const heatmapContainer = document.getElementById("heatmap");

const startBtn = document.getElementById("startBtn");
const showBtn = document.getElementById("showBtn");
const clearBtn = document.getElementById("clearBtn");
const downloadBtn = document.getElementById("downloadBtn");

let tracking = false;
let points = [];
let rawData = [];

let lastX = null;
let lastY = null;
let lastTime = null;
let stillStart = null;
let lastFixationSaved = 0;

const STILL_RADIUS = 18;      // px
const DWELL_TIME = 600;       // ms
const MIN_GAP_BETWEEN_FIX = 250; // ms

const heatmap = h337.create({
  container: heatmapContainer,
  radius: 35,
  maxOpacity: 0.7,
  minOpacity: 0.0,
  blur: 0.9
});

function resizeHeatmap() {
  heatmapContainer.style.width = stimulus.clientWidth + "px";
  heatmapContainer.style.height = stimulus.clientHeight + "px";
}

window.addEventListener("load", resizeHeatmap);
window.addEventListener("resize", resizeHeatmap);

function getRelativePosition(event) {
  const rect = wrapper.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;

  if (x < 0 || y < 0 || x > rect.width || y > rect.height) return null;

  return { x, y, width: rect.width, height: rect.height };
}

function distance(x1, y1, x2, y2) {
  return Math.hypot(x2 - x1, y2 - y1);
}

wrapper.addEventListener("mousemove", (event) => {
  if (!tracking) return;

  const pos = getRelativePosition(event);
  if (!pos) return;

  const now = performance.now();

  rawData.push({
    x: pos.x,
    y: pos.y,
    t: now
  });

  if (lastX === null) {
    lastX = pos.x;
    lastY = pos.y;
    lastTime = now;
    stillStart = now;
    return;
  }

  const d = distance(lastX, lastY, pos.x, pos.y);

  if (d <= STILL_RADIUS) {
    if (!stillStart) stillStart = now;

    const dwell = now - stillStart;
    const sinceLastSaved = now - lastFixationSaved;

    if (dwell >= DWELL_TIME && sinceLastSaved >= MIN_GAP_BETWEEN_FIX) {
      points.push({
        x: Math.round(pos.x),
        y: Math.round(pos.y),
        value: 1
      });
      lastFixationSaved = now;
      stillStart = now;
    }
  } else {
    stillStart = now;
  }

  lastX = pos.x;
  lastY = pos.y;
  lastTime = now;
});

startBtn.addEventListener("click", () => {
  tracking = true;
  points = [];
  rawData = [];
  lastX = null;
  lastY = null;
  lastTime = null;
  stillStart = null;
  lastFixationSaved = 0;
  heatmap.setData({ max: 5, data: [] });
  alert("Registro iniciado");
});

showBtn.addEventListener("click", () => {
  if (!points.length) {
    alert("No hay puntos registrados todavía.");
    return;
  }

  heatmap.setData({
    max: 10,
    data: points
  });
});

clearBtn.addEventListener("click", () => {
  tracking = false;
  points = [];
  rawData = [];
  heatmap.setData({ max: 5, data: [] });
});

downloadBtn.addEventListener("click", () => {
  const data = {
    image: stimulus.getAttribute("src"),
    timestamp: new Date().toISOString(),
    fixations_proxy: points,
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