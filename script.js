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

// --- CONFIGURACIÓN DE ANÁLISIS ---
const GRID_SIZE = 8;       // Tamaño de celda para agrupar puntos
const BASE_RADIUS = 35;    // Radio de la "fijación" (área de la fóvea)
const BLUR_STRENGTH = 25;  // Difuminado para homogeneizar zonas

function resizeCanvas() {
    const w = stimulus.clientWidth;
    const h = stimulus.clientHeight;
    if (!w || !h) return;
    canvas.width = w;
    canvas.height = h;
}

function getRelativePosition(event) {
    const rect = stimulus.getBoundingClientRect();
    return {
        x: event.clientX - rect.left,
        y: event.clientY - rect.top
    };
}

// 1. Crear la tira de colores profesional (Gradiente de Temperatura)
function getHeatmapGradient() {
    const gradCanvas = document.createElement("canvas");
    gradCanvas.width = 1;
    gradCanvas.height = 256;
    const gCtx = gradCanvas.getContext("2d");

    const gradient = gCtx.createLinearGradient(0, 0, 0, 256);
    gradient.addColorStop(0.1, "rgba(0, 0, 255, 0)"); // Transparente
    gradient.addColorStop(0.2, "blue");
    gradient.addColorStop(0.4, "cyan");
    gradient.addColorStop(0.6, "lime");
    gradient.addColorStop(0.8, "yellow");
    gradient.addColorStop(1.0, "red");

    gCtx.fillStyle = gradient;
    gCtx.fillRect(0, 0, 1, 256);
    return gCtx.getImageData(0, 0, 1, 256).data;
}

function drawHeatmap() {
    if (!rawData.length) return;

    // A. Agrupar puntos por rejilla para densidad
    const map = {};
    rawData.forEach(p => {
        const gx = Math.round(p.x / GRID_SIZE) * GRID_SIZE;
        const gy = Math.round(p.y / GRID_SIZE) * GRID_SIZE;
        const key = `${gx}_${gy}`;
        if (!map[key]) map[key] = { x: gx, y: gy, value: 0 };
        map[key].value += 1;
    });
    const points = Object.values(map);
    const maxValue = Math.max(...points.map(p => p.value));

    // B. Crear canvas temporal de sombras
    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tCtx = tempCanvas.getContext("2d");

    tCtx.shadowBlur = BLUR_STRENGTH;
    tCtx.shadowColor = "black";

    points.forEach(p => {
        const intensity = p.value / maxValue;
        tCtx.globalAlpha = intensity;
        tCtx.beginPath();
        tCtx.arc(p.x, p.y, BASE_RADIUS, 0, Math.PI * 2);
        tCtx.fill();
    });

    // C. Colorear píxel por píxel basado en la densidad
    const imgData = tCtx.getImageData(0, 0, canvas.width, canvas.height);
    const pix = imgData.data;
    const palette = getHeatmapGradient();

    for (let i = 0; i < pix.length; i += 4) {
        const alpha = pix[i + 3]; // La sombra acumulada define el color
        if (alpha > 0) {
            const offset = alpha * 4;
            pix[i] = palette[offset];     // R
            pix[i + 1] = palette[offset + 1]; // G
            pix[i + 2] = palette[offset + 2]; // B
            pix[i + 3] = alpha * 0.9;         // Suavizar transparencia final
        }
    }
    ctx.putImageData(imgData, 0, 0);
}

// Eventos
wrapper.addEventListener("mousemove", (e) => {
    if (!tracking) return;
    const pos = getRelativePosition(e);
    if (pos.x >= 0 && pos.x <= stimulus.clientWidth && pos.y >= 0 && pos.y <= stimulus.clientHeight) {
        rawData.push({ x: pos.x, y: pos.y });
    }
});

startBtn.addEventListener("click", () => {
    tracking = true;
    rawData = [];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    console.log("Grabación iniciada...");
});

showBtn.addEventListener("click", () => {
    tracking = false;
    drawHeatmap();
});

clearBtn.addEventListener("click", () => {
    tracking = false;
    rawData = [];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

downloadBtn.addEventListener("click", () => {
    const data = JSON.stringify({ points: rawData });
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "heatmap_data.json";
    a.click();
});

window.addEventListener("load", resizeCanvas);
window.addEventListener("resize", resizeCanvas);