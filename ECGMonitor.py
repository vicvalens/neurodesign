import flet as ft
import flet_charts as fch
import asyncio
import math
import random
import time
import csv
import os
from datetime import datetime
from collections import deque

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None


# =========================
# CONFIG
# =========================
BUFFER_SIZE = 240
METRIC_BUFFER_SIZE = 180
DEFAULT_BAUD = 115200
UPDATE_INTERVAL = 0.04
SERIAL_TIMEOUT = 0.02

CENTER = 512
RAW_MIN = 0
RAW_MAX = 1023


def main(page: ft.Page):
    page.title = "ECG Monitor - Cardiac Test UI"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 16
    page.window_width = 1550
    page.window_height = 980
    page.scroll = ft.ScrollMode.AUTO

    app_state = {
        "running": False,
        "connected": False,
        "simulate": True,
        "ser": None,
        "smooth_alpha": 0.18,
        "gain": 1.8,
        "last_ecg": float(CENTER),
        "chart_min_y": 420,
        "chart_max_y": 610,
        "signal_quality": 0.85,
        "sampling_hz": 0.0,
        "bpm": 0.0,
        "bpm_avg": 0.0,
        "rr_ms": 0.0,
        "rmssd": 0.0,
        "peak_threshold": 560.0,
        "csv_writer": None,
        "csv_file": None,
        "csv_path": None,
        "recording": False,
        "last_event": "Ninguno",
        "lead_off": False,
        "serial_format": "single_value",
        "last_peak_time": None,
    }

    ecg_buffer = deque([CENTER] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
    bpm_buffer = deque([0.0] * METRIC_BUFFER_SIZE, maxlen=METRIC_BUFFER_SIZE)
    rr_buffer = deque([0.0] * METRIC_BUFFER_SIZE, maxlen=METRIC_BUFFER_SIZE)
    quality_buffer = deque([85.0] * METRIC_BUFFER_SIZE, maxlen=METRIC_BUFFER_SIZE)

    raw_history = deque([CENTER] * 40, maxlen=40)
    rr_history = deque([], maxlen=30)
    sample_timestamps = deque([], maxlen=120)

    def safe_update():
        try:
            page.update()
            return True
        except Exception:
            app_state["running"] = False
            return False

    def clamp(v, vmin, vmax):
        return max(vmin, min(vmax, v))

    def smooth(prev, new, alpha):
        return prev + alpha * (new - prev)

    def normalize(value, min_val, max_val):
        if max_val - min_val == 0:
            return 0.0
        return (value - min_val) / (max_val - min_val)

    def map_to_0_100(value_0_1):
        value_0_1 = clamp(value_0_1, 0.0, 1.0)
        return int(value_0_1 * 100)

    def list_serial_ports():
        if serial is None:
            return []
        try:
            return [p.device for p in serial.tools.list_ports.comports()]
        except Exception:
            return []

    def refill_ports(e=None):
        ports = list_serial_ports()
        port_dropdown.options = [ft.DropdownOption(key=p, text=p) for p in ports]
        if ports and port_dropdown.value not in ports:
            port_dropdown.value = ports[0]
        if not ports:
            port_dropdown.value = None
        safe_update()

    def set_status(msg, color=ft.Colors.BLUE_200):
        status_text.value = msg
        status_text.color = color
        safe_update()

    def reset_buffers():
        ecg_buffer.clear()
        bpm_buffer.clear()
        rr_buffer.clear()
        quality_buffer.clear()
        raw_history.clear()
        rr_history.clear()
        sample_timestamps.clear()

        for _ in range(BUFFER_SIZE):
            ecg_buffer.append(CENTER)
        for _ in range(METRIC_BUFFER_SIZE):
            bpm_buffer.append(0.0)
            rr_buffer.append(0.0)
            quality_buffer.append(85.0)
        for _ in range(40):
            raw_history.append(CENTER)

    def parse_serial_line(line: str):
        line = line.strip()
        if not line:
            return None

        if app_state["serial_format"] == "single_value":
            try:
                return float(line)
            except ValueError:
                return None

        if app_state["serial_format"] == "timestamp_value":
            parts = line.split(",")
            if len(parts) != 2:
                return None
            try:
                return float(parts[1])
            except ValueError:
                return None

        return None

    def apply_gain(raw_value, gain):
        return CENTER + (raw_value - CENTER) * gain

    def simulated_ecg_sample(t):
        base_hr = 74 + 3 * math.sin(2 * math.pi * 0.03 * t)
        beat_period = 60.0 / base_hr
        phase = (t % beat_period) / beat_period

        p = 10 * math.exp(-((phase - 0.16) ** 2) / 0.0012)
        q = -16 * math.exp(-((phase - 0.39) ** 2) / 0.00020)
        r = 78 * math.exp(-((phase - 0.40) ** 2) / 0.00005)
        s = -24 * math.exp(-((phase - 0.43) ** 2) / 0.00016)
        tw = 24 * math.exp(-((phase - 0.66) ** 2) / 0.0045)
        drift = 5 * math.sin(2 * math.pi * 0.22 * t)
        noise = random.uniform(-3.0, 3.0)

        return CENTER + drift + p + q + r + s + tw + noise

    def update_sampling_hz():
        if len(sample_timestamps) < 2:
            app_state["sampling_hz"] = 0.0
            return
        dt = sample_timestamps[-1] - sample_timestamps[0]
        if dt <= 0:
            return
        app_state["sampling_hz"] = (len(sample_timestamps) - 1) / dt

    def compute_signal_quality(ecg_value):
        raw_history.append(ecg_value)
        values = list(raw_history)
        amplitude = max(values) - min(values)
        mean_value = sum(values) / len(values)
        variance = sum((v - mean_value) ** 2 for v in values) / len(values)
        std_value = math.sqrt(variance)

        amp_score = clamp(normalize(amplitude, 20, 180), 0.0, 1.0)
        std_score = clamp(normalize(std_value, 4, 40), 0.0, 1.0)
        quality = (amp_score * 0.65) + (std_score * 0.35)

        if app_state["lead_off"]:
            quality *= 0.2

        app_state["signal_quality"] = clamp(quality, 0.0, 1.0)

    def detect_peak_and_metrics(now_ts):
        if len(ecg_buffer) < 3:
            return

        prev2 = ecg_buffer[-3]
        prev1 = ecg_buffer[-2]
        curr = ecg_buffer[-1]
        threshold = app_state["peak_threshold"]
        refractory_sec = 0.28

        local_peak = prev1 > prev2 and prev1 > curr and prev1 > threshold
        if not local_peak:
            return

        if app_state["last_peak_time"] is not None:
            elapsed = now_ts - app_state["last_peak_time"]
            if elapsed < refractory_sec:
                return
            rr_ms = elapsed * 1000.0
            if 300 <= rr_ms <= 2000:
                app_state["rr_ms"] = rr_ms
                rr_history.append(rr_ms)
                app_state["bpm"] = 60000.0 / rr_ms
                if len(rr_history) > 0:
                    avg_rr = sum(rr_history) / len(rr_history)
                    if avg_rr > 0:
                        app_state["bpm_avg"] = 60000.0 / avg_rr
                if len(rr_history) >= 2:
                    diffs = []
                    rr_list = list(rr_history)
                    for i in range(1, len(rr_list)):
                        diffs.append(rr_list[i] - rr_list[i - 1])
                    if diffs:
                        app_state["rmssd"] = math.sqrt(sum(d * d for d in diffs) / len(diffs))
        app_state["last_peak_time"] = now_ts

    def get_state_label():
        bpm = app_state["bpm"]
        q = map_to_0_100(app_state["signal_quality"])

        if app_state["lead_off"]:
            return "Electrodos desconectados"
        if q < 25:
            return "Señal pobre"
        if bpm == 0:
            return "Sin detección aún"
        if bpm < 55:
            return "Ritmo bajo"
        if bpm <= 100:
            return "Ritmo esperado"
        return "Ritmo alto"

    def auto_adjust_signal_chart():
        data = list(ecg_buffer)
        if not data:
            return

        local_min = min(data)
        local_max = max(data)
        pad = 18

        ymin = max(0, local_min - pad)
        ymax = min(1023, local_max + pad)

        if ymax - ymin < 90:
            mid = (ymax + ymin) / 2
            ymin = max(0, mid - 45)
            ymax = min(1023, mid + 45)

        app_state["chart_min_y"] = ymin
        app_state["chart_max_y"] = ymax

    def signal_buffer_to_points(buffer_data):
        return [fch.LineChartDataPoint(x=float(i), y=float(v)) for i, v in enumerate(buffer_data)]

    def metric_buffer_to_points(buffer_data):
        return [fch.LineChartDataPoint(x=float(i), y=float(v)) for i, v in enumerate(buffer_data)]

    def metric_color(value):
        if value >= 70:
            return ft.Colors.GREEN_300
        if value >= 40:
            return ft.Colors.AMBER_300
        return ft.Colors.PINK_300

    def update_metric_cards():
        bpm_rounded = int(app_state["bpm"]) if app_state["bpm"] > 0 else 0
        bpm_avg_rounded = int(app_state["bpm_avg"]) if app_state["bpm_avg"] > 0 else 0
        rr_rounded = int(app_state["rr_ms"]) if app_state["rr_ms"] > 0 else 0
        rmssd_rounded = int(app_state["rmssd"]) if app_state["rmssd"] > 0 else 0
        quality_0100 = map_to_0_100(app_state["signal_quality"])
        hz_rounded = f"{app_state['sampling_hz']:.1f}"

        bpm_value.value = f"{bpm_rounded}"
        bpm_avg_value.value = f"{bpm_avg_rounded}"
        rr_value.value = f"{rr_rounded}"
        rmssd_value.value = f"{rmssd_rounded}"
        quality_value.value = f"{quality_0100}"
        hz_value.value = hz_rounded

        bpm_value.color = metric_color(bpm_rounded if bpm_rounded <= 100 else 65)
        bpm_avg_value.color = metric_color(bpm_avg_rounded if bpm_avg_rounded <= 100 else 65)
        rr_value.color = metric_color(80 if 500 <= rr_rounded <= 1200 else 35)
        rmssd_value.color = metric_color(60 if 10 <= rmssd_rounded <= 120 else 35)
        quality_value.color = metric_color(quality_0100)
        hz_value.color = ft.Colors.CYAN_300

        bpm_bar.value = clamp(bpm_rounded / 140.0, 0.0, 1.0)
        bpm_avg_bar.value = clamp(bpm_avg_rounded / 140.0, 0.0, 1.0)
        rr_bar.value = clamp(rr_rounded / 1500.0, 0.0, 1.0)
        rmssd_bar.value = clamp(rmssd_rounded / 200.0, 0.0, 1.0)
        quality_bar.value = quality_0100 / 100.0
        hz_bar.value = clamp(app_state["sampling_hz"] / 400.0, 0.0, 1.0)

        interpret_text.value = f"Estado estimado: {get_state_label()}"
        last_event_text.value = f"Último evento: {app_state['last_event']}"
        range_text.value = f"Rango señal: {app_state['chart_min_y']:.0f} - {app_state['chart_max_y']:.0f}"
        value_ecg.value = f"ECG actual: {app_state['last_ecg']:.1f}"
        mode_text.value = "Modo: Simulación" if app_state["simulate"] else "Modo: Arduino Serial"
        connection_text.value = "Conectado" if app_state["connected"] else "Desconectado"
        lead_text.value = "Lead-off: sí" if app_state["lead_off"] else "Lead-off: no"

    def update_charts():
        auto_adjust_signal_chart()

        signal_chart.min_y = app_state["chart_min_y"]
        signal_chart.max_y = app_state["chart_max_y"]

        signal_chart.data_series = [
            fch.LineChartData(
                points=signal_buffer_to_points(ecg_buffer),
                stroke_width=2.3,
                curved=True,
                color=ft.Colors.CYAN_400,
            ),
        ]

        metric_chart.data_series = [
            fch.LineChartData(
                points=metric_buffer_to_points(bpm_buffer),
                stroke_width=2.4,
                curved=True,
                color=ft.Colors.GREEN_300,
            ),
            fch.LineChartData(
                points=metric_buffer_to_points(rr_buffer),
                stroke_width=2.2,
                curved=True,
                color=ft.Colors.AMBER_300,
            ),
            fch.LineChartData(
                points=metric_buffer_to_points(quality_buffer),
                stroke_width=2.2,
                curved=True,
                color=ft.Colors.PURPLE_300,
            ),
        ]

        update_metric_cards()
        return safe_update()

    def write_csv_row():
        if not app_state["recording"] or app_state["csv_writer"] is None:
            return

        now = datetime.now().isoformat(timespec="milliseconds")
        app_state["csv_writer"].writerow([
            now,
            round(app_state["last_ecg"], 3),
            round(app_state["bpm"], 3),
            round(app_state["bpm_avg"], 3),
            round(app_state["rr_ms"], 3),
            round(app_state["rmssd"], 3),
            round(app_state["signal_quality"], 4),
            int(app_state["lead_off"]),
            get_state_label(),
            app_state["last_event"],
        ])

        try:
            app_state["csv_file"].flush()
        except Exception:
            pass

    def on_gain_change(e):
        app_state["gain"] = float(gain_slider.value)
        gain_label.value = f"Ganancia visual: {app_state['gain']:.2f}"
        safe_update()

    def on_smooth_change(e):
        app_state["smooth_alpha"] = float(smooth_slider.value)
        smooth_label.value = f"Suavizado: {app_state['smooth_alpha']:.2f}"
        safe_update()

    def on_threshold_change(e):
        app_state["peak_threshold"] = float(threshold_slider.value)
        threshold_label.value = f"Threshold R-peak: {app_state['peak_threshold']:.0f}"
        safe_update()

    def on_mode_change(e):
        app_state["simulate"] = mode_switch.value
        if app_state["simulate"]:
            disconnect_serial()
            set_status("Modo simulación activo", ft.Colors.GREEN_300)
        else:
            set_status("Modo Arduino activo. Conecta un puerto serial.", ft.Colors.AMBER_300)

    def on_format_change(e):
        app_state["serial_format"] = format_dropdown.value or "single_value"
        app_state["last_event"] = f"Formato serial -> {app_state['serial_format']}"
        set_status(f"Formato serial: {app_state['serial_format']}", ft.Colors.CYAN_200)

    def connect_serial(e=None):
        if serial is None:
            set_status("pyserial no está instalado", ft.Colors.RED_300)
            return

        if app_state["simulate"]:
            set_status("Desactiva simulación para conectar Arduino", ft.Colors.AMBER_300)
            return

        port = port_dropdown.value
        baud = int(baud_dropdown.value or DEFAULT_BAUD)

        if not port:
            set_status("Selecciona un puerto serial", ft.Colors.RED_300)
            return

        try:
            ser = serial.Serial(port, baud, timeout=SERIAL_TIMEOUT)
            time.sleep(1.5)
            app_state["ser"] = ser
            app_state["connected"] = True
            app_state["last_event"] = f"Conectado a {port}"
            set_status(f"Conectado a {port} @ {baud}", ft.Colors.GREEN_300)
            update_charts()
        except Exception as ex:
            app_state["ser"] = None
            app_state["connected"] = False
            set_status(f"No se pudo conectar: {ex}", ft.Colors.RED_300)

    def disconnect_serial(e=None):
        try:
            if app_state["ser"] is not None:
                app_state["ser"].close()
        except Exception:
            pass

        app_state["ser"] = None
        app_state["connected"] = False
        update_charts()

    def start_stream(e=None):
        if app_state["running"]:
            return

        app_state["running"] = True
        app_state["last_event"] = "Monitoreo iniciado"
        set_status("Adquisición iniciada", ft.Colors.GREEN_300)

        if not app_state["simulate"]:
            page.run_thread(serial_reader_loop)

        page.run_task(ui_update_loop)

    def stop_stream(e=None):
        app_state["running"] = False
        app_state["last_event"] = "Monitoreo detenido"
        set_status("Adquisición detenida", ft.Colors.BLUE_200)

    def clear_graph(e=None):
        reset_buffers()
        app_state["last_ecg"] = float(CENTER)
        app_state["signal_quality"] = 0.85
        app_state["sampling_hz"] = 0.0
        app_state["bpm"] = 0.0
        app_state["bpm_avg"] = 0.0
        app_state["rr_ms"] = 0.0
        app_state["rmssd"] = 0.0
        app_state["last_peak_time"] = None
        app_state["lead_off"] = False
        app_state["last_event"] = "Gráficas reiniciadas"
        update_charts()
        set_status("Gráficas reiniciadas", ft.Colors.BLUE_200)

    def start_recording(e=None):
        if app_state["recording"]:
            set_status("Ya se está grabando", ft.Colors.AMBER_300)
            return

        folder = os.path.join(os.getcwd(), "participants")
        os.makedirs(folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(folder, f"ecg_session_{timestamp}.csv")

        try:
            f = open(filepath, "w", newline="", encoding="utf-8")
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "ecg",
                "bpm",
                "bpm_avg",
                "rr_ms",
                "rmssd",
                "signal_quality",
                "lead_off",
                "state_label",
                "last_event",
            ])
            app_state["csv_file"] = f
            app_state["csv_writer"] = writer
            app_state["csv_path"] = filepath
            app_state["recording"] = True
            app_state["last_event"] = "Grabación iniciada"
            set_status("Grabando CSV en participants/", ft.Colors.GREEN_300)
            recording_text.value = f"Grabación: ACTIVA\n{filepath}"
            safe_update()
        except Exception as ex:
            set_status(f"No se pudo iniciar grabación: {ex}", ft.Colors.RED_300)

    def stop_recording(e=None):
        if not app_state["recording"]:
            set_status("No hay grabación activa", ft.Colors.AMBER_300)
            return

        try:
            if app_state["csv_file"] is not None:
                app_state["csv_file"].close()
        except Exception:
            pass

        path = app_state["csv_path"]
        app_state["csv_file"] = None
        app_state["csv_writer"] = None
        app_state["csv_path"] = None
        app_state["recording"] = False
        app_state["last_event"] = "Grabación finalizada"
        recording_text.value = "Grabación: inactiva"
        set_status(f"CSV guardado: {path}", ft.Colors.BLUE_200)

    def serial_reader_loop():
        while app_state["running"] and not app_state["simulate"]:
            ser = app_state["ser"]
            if ser is None or not app_state["connected"]:
                time.sleep(0.05)
                continue

            try:
                raw = ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue

                raw_value = parse_serial_line(raw)
                if raw_value is None:
                    continue

                g = app_state["gain"]
                alpha = app_state["smooth_alpha"]

                raw_value = clamp(raw_value, RAW_MIN, RAW_MAX)
                raw_value = apply_gain(raw_value, g)
                raw_value = clamp(raw_value, RAW_MIN, RAW_MAX)

                app_state["last_ecg"] = smooth(app_state["last_ecg"], raw_value, alpha)
                app_state["lead_off"] = raw_value <= 3 or raw_value >= 1020
            except Exception:
                pass

            time.sleep(0.001)

    async def ui_update_loop():
        while app_state["running"]:
            now_ts = time.time()
            sample_timestamps.append(now_ts)
            update_sampling_hz()

            if app_state["simulate"]:
                raw_value = simulated_ecg_sample(now_ts)

                g = app_state["gain"]
                alpha = app_state["smooth_alpha"]

                raw_value = apply_gain(raw_value, g)
                raw_value = clamp(raw_value, RAW_MIN, RAW_MAX)
                app_state["last_ecg"] = smooth(app_state["last_ecg"], raw_value, alpha)
                app_state["lead_off"] = False

            app_state["last_ecg"] = clamp(app_state["last_ecg"], RAW_MIN, RAW_MAX)
            ecg_buffer.append(app_state["last_ecg"])

            compute_signal_quality(app_state["last_ecg"])
            detect_peak_and_metrics(now_ts)

            bpm_buffer.append(app_state["bpm"])
            rr_buffer.append(app_state["rr_ms"])
            quality_buffer.append(map_to_0_100(app_state["signal_quality"]))

            write_csv_row()

            ok = update_charts()
            if not ok:
                break

            await asyncio.sleep(UPDATE_INTERVAL)

    port_dropdown = ft.Dropdown(
        label="Puerto serial",
        width=220,
        options=[],
    )

    baud_dropdown = ft.Dropdown(
        label="Baud rate",
        width=140,
        value=str(DEFAULT_BAUD),
        options=[
            ft.DropdownOption(key="9600", text="9600"),
            ft.DropdownOption(key="57600", text="57600"),
            ft.DropdownOption(key="115200", text="115200"),
            ft.DropdownOption(key="230400", text="230400"),
        ],
    )

    format_dropdown = ft.Dropdown(
        label="Formato serial",
        width=180,
        value="single_value",
        options=[
            ft.DropdownOption(key="single_value", text="single_value"),
            ft.DropdownOption(key="timestamp_value", text="timestamp,value"),
        ],
    )
    format_dropdown.on_change = on_format_change

    mode_switch = ft.Switch(
        label="Simulación",
        value=True,
    )
    mode_switch.on_change = on_mode_change

    gain_slider = ft.Slider(
        min=0.5,
        max=5.0,
        value=1.8,
        divisions=45,
        width=280,
        on_change=on_gain_change,
    )
    gain_label = ft.Text("Ganancia visual: 1.80")

    smooth_slider = ft.Slider(
        min=0.01,
        max=0.90,
        value=0.18,
        divisions=44,
        width=280,
        on_change=on_smooth_change,
    )
    smooth_label = ft.Text("Suavizado: 0.18")

    threshold_slider = ft.Slider(
        min=500,
        max=760,
        value=560,
        divisions=52,
        width=280,
        on_change=on_threshold_change,
    )
    threshold_label = ft.Text("Threshold R-peak: 560")

    status_text = ft.Text("Listo", color=ft.Colors.BLUE_200, size=14)
    protocol_hint_text = ft.Text("Usa esta UI para probar ECG serial y revisar BPM, RR y calidad.", size=12, color=ft.Colors.WHITE70)
    compact_hint_text = ft.Text("Recomendado: Arduino enviando un valor por línea con Serial.println(valor);", size=12, color=ft.Colors.WHITE70)
    mode_text = ft.Text("Modo: Simulación", weight=ft.FontWeight.BOLD)
    connection_text = ft.Text("Desconectado", weight=ft.FontWeight.BOLD)
    lead_text = ft.Text("Lead-off: no", weight=ft.FontWeight.BOLD)
    range_text = ft.Text("Rango señal: --", size=13, color=ft.Colors.WHITE70)
    interpret_text = ft.Text("Estado estimado: Sin detección aún", size=14, color=ft.Colors.GREEN_200)
    last_event_text = ft.Text("Último evento: Ninguno", size=13, color=ft.Colors.WHITE70)
    recording_text = ft.Text("Grabación: inactiva", size=13, color=ft.Colors.AMBER_200)

    value_ecg = ft.Text("ECG actual: 512.0", size=16, color=ft.Colors.CYAN_300)

    def metric_card(title, value_control, subtitle, bar_control=None, icon=ft.Icons.ANALYTICS, icon_color=ft.Colors.CYAN_300):
        controls = [
            ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                controls=[
                    ft.Text(title, size=13, color=ft.Colors.WHITE70),
                    ft.Icon(icon, size=18, color=icon_color),
                ],
            ),
            value_control,
        ]
        if bar_control is not None:
            controls.append(bar_control)
        controls.append(ft.Text(subtitle, size=11, color=ft.Colors.WHITE54))
        return ft.Container(
            bgcolor=ft.Colors.with_opacity(0.03, ft.Colors.WHITE),
            border_radius=16,
            padding=12,
            width=150,
            content=ft.Column(
                spacing=6,
                controls=controls,
            ),
        )

    def make_metric_bar(value=0.0, color=ft.Colors.CYAN_300):
        return ft.ProgressBar(value=value, width=140, color=color, bgcolor=ft.Colors.with_opacity(0.12, ft.Colors.WHITE))

    bpm_value = ft.Text("0", size=28, weight=ft.FontWeight.BOLD)
    bpm_avg_value = ft.Text("0", size=28, weight=ft.FontWeight.BOLD)
    rr_value = ft.Text("0", size=28, weight=ft.FontWeight.BOLD)
    rmssd_value = ft.Text("0", size=28, weight=ft.FontWeight.BOLD)
    quality_value = ft.Text("85", size=28, weight=ft.FontWeight.BOLD)
    hz_value = ft.Text("0.0", size=28, weight=ft.FontWeight.BOLD)

    bpm_bar = make_metric_bar(0.0, ft.Colors.GREEN_300)
    bpm_avg_bar = make_metric_bar(0.0, ft.Colors.CYAN_300)
    rr_bar = make_metric_bar(0.0, ft.Colors.AMBER_300)
    rmssd_bar = make_metric_bar(0.0, ft.Colors.PINK_300)
    quality_bar = make_metric_bar(0.85, ft.Colors.PURPLE_300)
    hz_bar = make_metric_bar(0.0, ft.Colors.BLUE_300)

    signal_chart = fch.LineChart(
        min_x=0,
        max_x=BUFFER_SIZE - 1,
        min_y=420,
        max_y=610,
        expand=True,
        horizontal_grid_lines=fch.ChartGridLines(
            interval=20,
            color=ft.Colors.with_opacity(0.10, ft.Colors.WHITE),
            width=1,
        ),
        vertical_grid_lines=fch.ChartGridLines(
            interval=20,
            color=ft.Colors.with_opacity(0.06, ft.Colors.WHITE),
            width=1,
        ),
        left_axis=fch.ChartAxis(label_size=42, title=ft.Text("Amplitud")),
        bottom_axis=fch.ChartAxis(label_size=28, title=ft.Text("Muestras")),
        bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.WHITE),
        data_series=[],
    )

    metric_chart = fch.LineChart(
        min_x=0,
        max_x=METRIC_BUFFER_SIZE - 1,
        min_y=0,
        max_y=150,
        expand=True,
        horizontal_grid_lines=fch.ChartGridLines(
            interval=20,
            color=ft.Colors.with_opacity(0.10, ft.Colors.WHITE),
            width=1,
        ),
        vertical_grid_lines=fch.ChartGridLines(
            interval=20,
            color=ft.Colors.with_opacity(0.06, ft.Colors.WHITE),
            width=1,
        ),
        left_axis=fch.ChartAxis(label_size=42, title=ft.Text("Métrica")),
        bottom_axis=fch.ChartAxis(label_size=28, title=ft.Text("Tiempo")),
        bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.WHITE),
        data_series=[],
    )

    refill_ports()

    controls_panel = ft.Card(
        content=ft.Container(
            width=410,
            padding=12,
            content=ft.Column(
                spacing=8,
                controls=[
                    ft.Text("Configuración y monitoreo ECG", size=18, weight=ft.FontWeight.BOLD),
                    ft.Row(
                        spacing=8,
                        wrap=True,
                        controls=[
                            port_dropdown,
                            baud_dropdown,
                        ],
                    ),
                    ft.Row(
                        spacing=8,
                        wrap=True,
                        controls=[
                            format_dropdown,
                            mode_switch,
                        ],
                    ),
                    ft.Row(
                        spacing=8,
                        wrap=True,
                        controls=[
                            ft.ElevatedButton("Actualizar puertos", on_click=refill_ports, icon=ft.Icons.REFRESH),
                            ft.ElevatedButton("Conectar", on_click=connect_serial, icon=ft.Icons.LINK),
                            ft.OutlinedButton("Desconectar", on_click=disconnect_serial, icon=ft.Icons.LINK_OFF),
                        ],
                    ),
                    ft.Row(
                        spacing=8,
                        wrap=True,
                        controls=[
                            ft.ElevatedButton("Iniciar", on_click=start_stream, icon=ft.Icons.PLAY_ARROW),
                            ft.OutlinedButton("Detener", on_click=stop_stream, icon=ft.Icons.STOP),
                            ft.OutlinedButton("Limpiar", on_click=clear_graph, icon=ft.Icons.DELETE_SWEEP),
                        ],
                    ),
                    ft.Row(
                        spacing=8,
                        wrap=True,
                        controls=[
                            ft.ElevatedButton("Iniciar CSV", on_click=start_recording, icon=ft.Icons.FIBER_MANUAL_RECORD),
                            ft.OutlinedButton("Detener CSV", on_click=stop_recording, icon=ft.Icons.SAVE_ALT),
                        ],
                    ),
                    ft.Divider(height=12),
                    gain_label,
                    gain_slider,
                    smooth_label,
                    smooth_slider,
                    threshold_label,
                    threshold_slider,
                    ft.Divider(height=12),
                    status_text,
                    mode_text,
                    connection_text,
                    lead_text,
                    range_text,
                    value_ecg,
                    interpret_text,
                    last_event_text,
                    recording_text,
                    ft.Divider(height=12),
                    protocol_hint_text,
                    compact_hint_text,
                ],
            ),
        )
    )

    metrics_row = ft.Row(
        wrap=True,
        spacing=10,
        run_spacing=10,
        controls=[
            metric_card("BPM actual", bpm_value, "Latidos por minuto detectados", bpm_bar, ft.Icons.FAVORITE, ft.Colors.RED_300),
            metric_card("BPM promedio", bpm_avg_value, "Promedio de los RR detectados", bpm_avg_bar, ft.Icons.MONITOR_HEART, ft.Colors.CYAN_300),
            metric_card("RR (ms)", rr_value, "Intervalo entre picos R", rr_bar, ft.Icons.TIMELINE, ft.Colors.AMBER_300),
            metric_card("RMSSD", rmssd_value, "Variabilidad simple demo", rmssd_bar, ft.Icons.SHOW_CHART, ft.Colors.PINK_300),
            metric_card("Calidad", quality_value, "Calidad estimada de señal", quality_bar, ft.Icons.ANALYTICS, ft.Colors.PURPLE_300),
            metric_card("Hz estimados", hz_value, "Frecuencia de muestreo observada", hz_bar, ft.Icons.SPEED, ft.Colors.BLUE_300),
        ],
    )

    chart_card_1 = ft.Card(
        content=ft.Container(
            padding=12,
            height=360,
            content=ft.Column(
                spacing=8,
                controls=[
                    ft.Text("ECG en tiempo real", size=18, weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True, content=signal_chart),
                ],
            ),
        )
    )

    chart_card_2 = ft.Card(
        content=ft.Container(
            padding=12,
            height=300,
            content=ft.Column(
                spacing=8,
                controls=[
                    ft.Text("BPM / RR / Calidad", size=18, weight=ft.FontWeight.BOLD),
                    ft.Container(expand=True, content=metric_chart),
                ],
            ),
        )
    )

    right_panel = ft.Column(
        expand=True,
        spacing=12,
        controls=[
            metrics_row,
            chart_card_1,
            chart_card_2,
        ],
    )

    page.add(
        ft.Column(
            spacing=12,
            controls=[
                ft.Text("ECG Monitor / Cardiac Test Interface", size=28, weight=ft.FontWeight.BOLD),
                ft.Text("Base derivada de all graphs, adaptada para lectura cardiaca por puerto COM.", size=13, color=ft.Colors.WHITE70),
                ft.Row(
                    expand=True,
                    vertical_alignment=ft.CrossAxisAlignment.START,
                    controls=[
                        controls_panel,
                        right_panel,
                    ],
                ),
            ],
        )
    )

    update_charts()


ft.run(main)
