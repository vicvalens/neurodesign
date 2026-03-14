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
METRIC_BUFFER_SIZE = 240
DEFAULT_BAUD = 115200
UPDATE_INTERVAL = 0.04
SERIAL_TIMEOUT = 0.02

CENTER = 512
RAW_MIN = 0
RAW_MAX = 1023


def main(page: ft.Page):
    page.title = "Neurodiseño - Bioseñales y Métricas Afectivas"
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
        "smooth_alpha": 0.20,
        "gain": 1.8,
        "last_ch1": float(CENTER),
        "last_ch2": float(CENTER),
        "chart_min_y": 420,
        "chart_max_y": 610,
        "phase": "Baseline",
        "simulation_profile": "Calmado",
        "signal_quality": 0.85,
        "valence": 0.0,
        "arousal": 0.0,
        "engagement": 0.0,
        "relaxation": 0.0,
        "csv_writer": None,
        "csv_file": None,
        "csv_path": None,
        "recording": False,
        "last_event": "Ninguno",
    }

    ch1_buffer = deque([CENTER] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
    ch2_buffer = deque([CENTER] * BUFFER_SIZE, maxlen=BUFFER_SIZE)

    valence_buffer = deque([0.0] * METRIC_BUFFER_SIZE, maxlen=METRIC_BUFFER_SIZE)
    arousal_buffer = deque([0.0] * METRIC_BUFFER_SIZE, maxlen=METRIC_BUFFER_SIZE)
    engagement_buffer = deque([0.0] * METRIC_BUFFER_SIZE, maxlen=METRIC_BUFFER_SIZE)
    relaxation_buffer = deque([0.0] * METRIC_BUFFER_SIZE, maxlen=METRIC_BUFFER_SIZE)

    energy_history = deque([0.0] * 40, maxlen=40)
    diff_history = deque([0.0] * 40, maxlen=40)

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

    def map_to_bipolar_0_100(value_minus1_to_1):
        value_minus1_to_1 = clamp(value_minus1_to_1, -1.0, 1.0)
        return int((value_minus1_to_1 + 1.0) * 50)

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
        ch1_buffer.clear()
        ch2_buffer.clear()
        valence_buffer.clear()
        arousal_buffer.clear()
        engagement_buffer.clear()
        relaxation_buffer.clear()

        for _ in range(BUFFER_SIZE):
            ch1_buffer.append(CENTER)
            ch2_buffer.append(CENTER)

        for _ in range(METRIC_BUFFER_SIZE):
            valence_buffer.append(0.0)
            arousal_buffer.append(0.0)
            engagement_buffer.append(0.0)
            relaxation_buffer.append(0.0)

        energy_history.clear()
        diff_history.clear()
        for _ in range(40):
            energy_history.append(0.0)
            diff_history.append(0.0)

    def parse_serial_line(line: str):
        parts = line.strip().split(",")
        if len(parts) != 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None

    def apply_gain(raw_value, gain):
        return CENTER + (raw_value - CENTER) * gain

    def get_profile_params(profile_name):
        profiles = {
            "Calmado": {
                "base_amp": 10,
                "wave1": 18,
                "wave2": 14,
                "noise": 5,
                "fast": 5.0,
                "slow": 0.16,
            },
            "Activado": {
                "base_amp": 16,
                "wave1": 26,
                "wave2": 22,
                "noise": 8,
                "fast": 8.0,
                "slow": 0.22,
            },
            "Estrés leve": {
                "base_amp": 18,
                "wave1": 34,
                "wave2": 28,
                "noise": 12,
                "fast": 10.5,
                "slow": 0.28,
            },
            "Enfoque alto": {
                "base_amp": 12,
                "wave1": 22,
                "wave2": 20,
                "noise": 6,
                "fast": 7.0,
                "slow": 0.20,
            },
        }
        return profiles.get(profile_name, profiles["Calmado"])

    def simulated_sample(t):
        params = get_profile_params(app_state["simulation_profile"])

        base1 = CENTER + params["base_amp"] * math.sin(2 * math.pi * params["slow"] * t)
        base2 = CENTER + (params["base_amp"] - 2) * math.sin(2 * math.pi * (params["slow"] * 0.9) * t + 1.1)

        w1 = params["wave1"] * math.sin(2 * math.pi * params["fast"] * t)
        w2 = params["wave2"] * math.sin(2 * math.pi * (params["fast"] * 0.85) * t + 0.8)

        mod1 = 8 * math.sin(2 * math.pi * 11.0 * t)
        mod2 = 6 * math.sin(2 * math.pi * 9.0 * t + 0.5)

        noise1 = random.uniform(-params["noise"], params["noise"])
        noise2 = random.uniform(-params["noise"], params["noise"])

        ch1 = base1 + w1 + mod1 + noise1
        ch2 = base2 + w2 + mod2 + noise2

        phase_bias = {
            "Baseline": (0, 0),
            "Relajación": (-8, -10),
            "Estímulo": (8, 6),
            "Tarea cognitiva": (12, 10),
            "VR": (10, 8),
            "Recuperación": (-4, -6),
        }
        b1, b2 = phase_bias.get(app_state["phase"], (0, 0))
        ch1 += b1
        ch2 += b2

        return ch1, ch2

    def compute_signal_quality(ch1, ch2):
        abs_diff = abs(ch1 - ch2)
        amp1 = abs(ch1 - CENTER)
        amp2 = abs(ch2 - CENTER)

        rough_noise = normalize(abs_diff, 0, 120)
        activity = normalize((amp1 + amp2) / 2, 0, 160)

        quality = 0.95 - (rough_noise * 0.45)
        quality += activity * 0.10
        return clamp(quality, 0.0, 1.0)

    def compute_metrics(ch1, ch2):
        d1 = ch1 - CENTER
        d2 = ch2 - CENTER

        energy = (abs(d1) + abs(d2)) / 2.0
        asymmetry = d1 - d2
        stability = abs(d1 - d2)

        energy_history.append(energy)
        diff_history.append(stability)

        avg_energy = sum(energy_history) / len(energy_history)
        avg_diff = sum(diff_history) / len(diff_history)

        arousal = clamp(normalize(avg_energy, 5, 95), 0.0, 1.0)
        valence_raw = clamp(asymmetry / 80.0, -1.0, 1.0)
        engagement = clamp((arousal * 0.6) + (normalize(avg_energy, 10, 70) * 0.4), 0.0, 1.0)
        relaxation = clamp(1.0 - arousal + 0.12, 0.0, 1.0)
        quality = compute_signal_quality(ch1, ch2)

        app_state["valence"] = valence_raw
        app_state["arousal"] = arousal
        app_state["engagement"] = engagement
        app_state["relaxation"] = relaxation
        app_state["signal_quality"] = quality

    def get_state_label():
        v = app_state["valence"]
        a = app_state["arousal"]
        e = app_state["engagement"]

        if a < 0.35 and v >= 0:
            return "Relajado / positivo"
        if a < 0.35 and v < 0:
            return "Relajado / baja valencia"
        if a >= 0.35 and a < 0.7 and e >= 0.5:
            return "Atento / involucrado"
        if a >= 0.7 and v >= 0:
            return "Activado / estimulación alta"
        if a >= 0.7 and v < 0:
            return "Alta activación / posible tensión"
        return "Estado intermedio"

    def auto_adjust_signal_chart():
        data = list(ch1_buffer) + list(ch2_buffer)
        if not data:
            return

        local_min = min(data)
        local_max = max(data)
        pad = 18

        ymin = max(0, local_min - pad)
        ymax = min(1023, local_max + pad)

        if ymax - ymin < 100:
            mid = (ymax + ymin) / 2
            ymin = max(0, mid - 50)
            ymax = min(1023, mid + 50)

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
        valence_0100 = map_to_bipolar_0_100(app_state["valence"])
        arousal_0100 = map_to_0_100(app_state["arousal"])
        engagement_0100 = map_to_0_100(app_state["engagement"])
        relaxation_0100 = map_to_0_100(app_state["relaxation"])
        quality_0100 = map_to_0_100(app_state["signal_quality"])

        valence_value.value = f"{valence_0100}"
        arousal_value.value = f"{arousal_0100}"
        engagement_value.value = f"{engagement_0100}"
        relaxation_value.value = f"{relaxation_0100}"
        quality_value.value = f"{quality_0100}"

        valence_value.color = metric_color(valence_0100)
        arousal_value.color = metric_color(arousal_0100)
        engagement_value.color = metric_color(engagement_0100)
        relaxation_value.color = metric_color(relaxation_0100)
        quality_value.color = metric_color(quality_0100)

        valence_bar.value = valence_0100 / 100
        arousal_bar.value = arousal_0100 / 100
        engagement_bar.value = engagement_0100 / 100
        relaxation_bar.value = relaxation_0100 / 100
        quality_bar.value = quality_0100 / 100

        phase_text.value = f"Fase actual: {app_state['phase']}"
        interpret_text.value = f"Estado estimado: {get_state_label()}"
        last_event_text.value = f"Último evento: {app_state['last_event']}"

    def update_charts():
        auto_adjust_signal_chart()

        signal_chart.min_y = app_state["chart_min_y"]
        signal_chart.max_y = app_state["chart_max_y"]

        signal_chart.data_series = [
            fch.LineChartData(
                points=signal_buffer_to_points(ch1_buffer),
                stroke_width=2.3,
                curved=True,
                color=ft.Colors.CYAN_400,
            ),
            fch.LineChartData(
                points=signal_buffer_to_points(ch2_buffer),
                stroke_width=2.3,
                curved=True,
                color=ft.Colors.PINK_300,
            ),
        ]

        metric_chart.data_series = [
            fch.LineChartData(
                points=metric_buffer_to_points(valence_buffer),
                stroke_width=2.6,
                curved=True,
                color=ft.Colors.GREEN_300,
            ),
            fch.LineChartData(
                points=metric_buffer_to_points(arousal_buffer),
                stroke_width=2.6,
                curved=True,
                color=ft.Colors.AMBER_300,
            ),
            fch.LineChartData(
                points=metric_buffer_to_points(engagement_buffer),
                stroke_width=2.6,
                curved=True,
                color=ft.Colors.CYAN_300,
            ),
            fch.LineChartData(
                points=metric_buffer_to_points(relaxation_buffer),
                stroke_width=2.6,
                curved=True,
                color=ft.Colors.PURPLE_300,
            ),
        ]

        range_text.value = f"Rango señal: {app_state['chart_min_y']:.0f} - {app_state['chart_max_y']:.0f}"
        value_ch1.value = f"Canal 1: {app_state['last_ch1']:.1f}"
        value_ch2.value = f"Canal 2: {app_state['last_ch2']:.1f}"
        mode_text.value = "Modo: Simulación" if app_state["simulate"] else "Modo: Arduino Serial"
        connection_text.value = "Conectado" if app_state["connected"] else "Desconectado"

        update_metric_cards()
        return safe_update()

    def write_csv_row():
        if not app_state["recording"] or app_state["csv_writer"] is None:
            return

        now = datetime.now().isoformat(timespec="milliseconds")
        app_state["csv_writer"].writerow([
            now,
            app_state["phase"],
            app_state["simulation_profile"] if app_state["simulate"] else "Arduino",
            round(app_state["last_ch1"], 3),
            round(app_state["last_ch2"], 3),
            round(app_state["valence"], 4),
            round(app_state["arousal"], 4),
            round(app_state["engagement"], 4),
            round(app_state["relaxation"], 4),
            round(app_state["signal_quality"], 4),
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

    def on_mode_change(e):
        app_state["simulate"] = mode_switch.value
        if app_state["simulate"]:
            disconnect_serial()
            set_status("Modo simulación activo", ft.Colors.GREEN_300)
        else:
            set_status("Modo Arduino activo. Conecta un puerto serial.", ft.Colors.AMBER_300)

    def on_profile_change(e):
        app_state["simulation_profile"] = profile_dropdown.value or "Calmado"
        app_state["last_event"] = f"Perfil: {app_state['simulation_profile']}"
        set_status(f"Perfil de simulación: {app_state['simulation_profile']}", ft.Colors.CYAN_200)

    def refresh_phase_buttons():
        for phase_name, btn in phase_buttons.items():
            active = app_state["phase"] == phase_name
            btn.style = ft.ButtonStyle(
                bgcolor=ft.Colors.BLUE_400 if active else ft.Colors.with_opacity(0.0, ft.Colors.WHITE),
                color=ft.Colors.WHITE if active else ft.Colors.BLUE_100,
                side=ft.BorderSide(1, ft.Colors.BLUE_200 if active else ft.Colors.WHITE38),
                shape=ft.RoundedRectangleBorder(radius=12),
            )

    def set_phase(phase_name):
        app_state["phase"] = phase_name
        app_state["last_event"] = f"Fase -> {phase_name}"
        set_status(f"Fase activa: {phase_name}", ft.Colors.GREEN_200)
        update_metric_cards()
        refresh_phase_buttons()
        safe_update()

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
        set_status("Adquisición iniciada", ft.Colors.GREEN_300)

        if not app_state["simulate"]:
            page.run_thread(serial_reader_loop)

        page.run_task(ui_update_loop)

    def stop_stream(e=None):
        app_state["running"] = False
        set_status("Adquisición detenida", ft.Colors.BLUE_200)

    def clear_graph(e=None):
        reset_buffers()
        app_state["last_ch1"] = float(CENTER)
        app_state["last_ch2"] = float(CENTER)
        app_state["valence"] = 0.0
        app_state["arousal"] = 0.0
        app_state["engagement"] = 0.0
        app_state["relaxation"] = 0.0
        app_state["signal_quality"] = 0.85
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
        filepath = os.path.join(folder, f"biosignals_session_{timestamp}.csv")

        try:
            f = open(filepath, "w", newline="", encoding="utf-8")
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "phase",
                "mode_or_profile",
                "ch1",
                "ch2",
                "valence",
                "arousal",
                "engagement",
                "relaxation",
                "signal_quality",
                "state_label",
                "last_event",
            ])
            app_state["csv_file"] = f
            app_state["csv_writer"] = writer
            app_state["csv_path"] = filepath
            app_state["recording"] = True
            app_state["last_event"] = "Grabación iniciada"
            set_status(f"Grabando CSV en participants/", ft.Colors.GREEN_300)
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

                parsed = parse_serial_line(raw)
                if parsed is None:
                    continue

                raw_ch1, raw_ch2 = parsed

                g = app_state["gain"]
                alpha = app_state["smooth_alpha"]

                raw_ch1 = apply_gain(raw_ch1, g)
                raw_ch2 = apply_gain(raw_ch2, g)

                app_state["last_ch1"] = smooth(app_state["last_ch1"], raw_ch1, alpha)
                app_state["last_ch2"] = smooth(app_state["last_ch2"], raw_ch2, alpha)

            except Exception:
                pass

            time.sleep(0.001)

    async def ui_update_loop():
        while app_state["running"]:
            if app_state["simulate"]:
                t = time.time()
                raw1, raw2 = simulated_sample(t)

                g = app_state["gain"]
                alpha = app_state["smooth_alpha"]

                raw1 = apply_gain(raw1, g)
                raw2 = apply_gain(raw2, g)

                app_state["last_ch1"] = smooth(app_state["last_ch1"], raw1, alpha)
                app_state["last_ch2"] = smooth(app_state["last_ch2"], raw2, alpha)

            app_state["last_ch1"] = clamp(app_state["last_ch1"], RAW_MIN, RAW_MAX)
            app_state["last_ch2"] = clamp(app_state["last_ch2"], RAW_MIN, RAW_MAX)

            ch1_buffer.append(app_state["last_ch1"])
            ch2_buffer.append(app_state["last_ch2"])

            compute_metrics(app_state["last_ch1"], app_state["last_ch2"])

            valence_buffer.append(map_to_bipolar_0_100(app_state["valence"]))
            arousal_buffer.append(map_to_0_100(app_state["arousal"]))
            engagement_buffer.append(map_to_0_100(app_state["engagement"]))
            relaxation_buffer.append(map_to_0_100(app_state["relaxation"]))

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
        ],
    )

    profile_dropdown = ft.Dropdown(
        label="Perfil simulación",
        width=180,
        value="Calmado",
        options=[
            ft.DropdownOption(key="Calmado", text="Calmado"),
            ft.DropdownOption(key="Activado", text="Activado"),
            ft.DropdownOption(key="Estrés leve", text="Estrés leve"),
            ft.DropdownOption(key="Enfoque alto", text="Enfoque alto"),
        ],
        on_select=on_profile_change,
    )

    mode_switch = ft.Switch(
        label="Simulación",
        value=True,
        on_change=on_mode_change,
    )

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
        value=0.20,
        divisions=44,
        width=280,
        on_change=on_smooth_change,
    )
    smooth_label = ft.Text("Suavizado: 0.20")

    status_text = ft.Text("Listo", color=ft.Colors.BLUE_200, size=14)
    protocol_hint_text = ft.Text("Usa las fases para marcar el momento del experimento.", size=12, color=ft.Colors.WHITE70)
    compact_hint_text = ft.Text("Selecciona una fase y controla la adquisición desde el panel izquierdo.", size=12, color=ft.Colors.WHITE70)
    mode_text = ft.Text("Modo: Simulación", weight=ft.FontWeight.BOLD)
    connection_text = ft.Text("Desconectado", weight=ft.FontWeight.BOLD)
    range_text = ft.Text("Rango señal: --", size=13, color=ft.Colors.WHITE70)
    phase_text = ft.Text("Fase actual: Baseline", size=14, color=ft.Colors.CYAN_200)
    interpret_text = ft.Text("Estado estimado: Estado intermedio", size=14, color=ft.Colors.GREEN_200)
    last_event_text = ft.Text("Último evento: Ninguno", size=13, color=ft.Colors.WHITE70)
    recording_text = ft.Text("Grabación: inactiva", size=13, color=ft.Colors.AMBER_200)

    value_ch1 = ft.Text("Canal 1: 512.0", size=16, color=ft.Colors.CYAN_300)
    value_ch2 = ft.Text("Canal 2: 512.0", size=16, color=ft.Colors.PINK_300)

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

    valence_value = ft.Text("50", size=28, weight=ft.FontWeight.BOLD)
    arousal_value = ft.Text("0", size=28, weight=ft.FontWeight.BOLD)
    engagement_value = ft.Text("0", size=28, weight=ft.FontWeight.BOLD)
    relaxation_value = ft.Text("0", size=28, weight=ft.FontWeight.BOLD)
    quality_value = ft.Text("85", size=28, weight=ft.FontWeight.BOLD)

    valence_bar = make_metric_bar(0.5, ft.Colors.GREEN_300)
    arousal_bar = make_metric_bar(0.0, ft.Colors.AMBER_300)
    engagement_bar = make_metric_bar(0.0, ft.Colors.CYAN_300)
    relaxation_bar = make_metric_bar(0.85, ft.Colors.PURPLE_300)
    quality_bar = make_metric_bar(0.85, ft.Colors.BLUE_300)

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
        max_y=100,
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
        left_axis=fch.ChartAxis(label_size=42, title=ft.Text("Índice")),
        bottom_axis=fch.ChartAxis(label_size=28, title=ft.Text("Tiempo")),
        bgcolor=ft.Colors.with_opacity(0.02, ft.Colors.WHITE),
        data_series=[],
    )

    refill_ports()

    phase_buttons = {
        "Baseline": ft.OutlinedButton("Baseline", on_click=lambda e: set_phase("Baseline"), height=34),
        "Relajación": ft.OutlinedButton("Relajación", on_click=lambda e: set_phase("Relajación"), height=34),
        "Estímulo": ft.OutlinedButton("Estímulo", on_click=lambda e: set_phase("Estímulo"), height=34),
        "Tarea cognitiva": ft.OutlinedButton("Tarea cognitiva", on_click=lambda e: set_phase("Tarea cognitiva"), height=34),
        "VR": ft.OutlinedButton("VR", on_click=lambda e: set_phase("VR"), height=34),
        "Recuperación": ft.OutlinedButton("Recuperación", on_click=lambda e: set_phase("Recuperación"), height=34),
    }

    controls_panel = ft.Card(
        content=ft.Container(
            width=410,
            padding=12,
            content=ft.Column(
                spacing=8,
                controls=[
                    ft.Text("Configuración y experimento", size=18, weight=ft.FontWeight.BOLD),
                    ft.Row(
                        controls=[
                            port_dropdown,
                            baud_dropdown,
                            ft.IconButton(
                                icon=ft.Icons.REFRESH,
                                tooltip="Actualizar puertos",
                                on_click=refill_ports,
                            ),
                        ],
                        wrap=True,
                        spacing=8,
                    ),
                    ft.Row(
                        controls=[mode_switch, profile_dropdown],
                        wrap=True,
                        spacing=10,
                    ),
                    ft.Row(
                        controls=[
                            ft.FilledButton("Conectar", icon=ft.Icons.USB, on_click=connect_serial, height=36),
                            ft.OutlinedButton("Desconectar", icon=ft.Icons.LINK_OFF, on_click=disconnect_serial, height=36),
                        ],
                        wrap=True,
                        spacing=8,
                    ),
                    ft.Divider(height=8),
                    ft.Text("Fases de la experimentación", size=14, weight=ft.FontWeight.BOLD),
                    ft.Row(
                        wrap=True,
                        spacing=8,
                        run_spacing=8,
                        controls=list(phase_buttons.values()),
                    ),
                    ft.Divider(height=8),
                    ft.Text("Ajustes de señal", size=14, weight=ft.FontWeight.BOLD),
                    gain_label,
                    ft.Container(padding=ft.Padding.only(left=4, right=4), content=gain_slider),
                    smooth_label,
                    ft.Container(padding=ft.Padding.only(left=4, right=4), content=smooth_slider),
                    ft.Divider(height=8),
                    ft.Row(
                        controls=[
                            ft.FilledButton("Iniciar", icon=ft.Icons.PLAY_ARROW, on_click=start_stream, height=36),
                            ft.OutlinedButton("Detener", icon=ft.Icons.STOP, on_click=stop_stream, height=36),
                            ft.TextButton("Limpiar", icon=ft.Icons.DELETE_SWEEP, on_click=clear_graph),
                        ],
                        wrap=True,
                        spacing=8,
                    ),
                    ft.Row(
                        controls=[
                            ft.FilledButton("Iniciar CSV", icon=ft.Icons.SAVE, on_click=start_recording, height=36),
                            ft.OutlinedButton("Detener CSV", icon=ft.Icons.STOP_CIRCLE, on_click=stop_recording, height=36),
                        ],
                        wrap=True,
                        spacing=8,
                    ),
                ],
            ),
        )
    )

    status_panel = ft.Card(
        content=ft.Container(
            padding=16,
            content=ft.Column(
                spacing=6,
                controls=[
                    ft.Text("Estado del sistema", size=18, weight=ft.FontWeight.BOLD),
                    ft.ResponsiveRow(
                        run_spacing=8,
                        spacing=8,
                        controls=[
                            ft.Container(col={"xs": 6, "md": 6}, content=mode_text),
                            ft.Container(col={"xs": 6, "md": 6}, content=connection_text),
                            ft.Container(col={"xs": 6, "md": 6}, content=phase_text),
                            ft.Container(col={"xs": 6, "md": 6}, content=recording_text),
                            ft.Container(col={"xs": 6, "md": 6}, content=value_ch1),
                            ft.Container(col={"xs": 6, "md": 6}, content=value_ch2),
                        ],
                    ),
                    ft.Divider(height=8),
                    range_text,
                    interpret_text,
                    last_event_text,
                    compact_hint_text,
                    status_text,
                ],
            ),
        )
    )

    metrics_panel = ft.Card(
        content=ft.Container(
            padding=16,
            content=ft.Column(
                spacing=12,
                controls=[
                    ft.Text("Métricas afectivas y cognitivas", size=18, weight=ft.FontWeight.BOLD),
                    ft.Row(
                        wrap=True,
                        spacing=10,
                        run_spacing=10,
                        controls=[
                            metric_card("Valencia", valence_value, "0-100", valence_bar, icon=ft.Icons.MOOD, icon_color=ft.Colors.GREEN_300),
                            metric_card("Arousal", arousal_value, "0-100", arousal_bar, icon=ft.Icons.LOCAL_FIRE_DEPARTMENT, icon_color=ft.Colors.AMBER_300),
                            metric_card("Engagement", engagement_value, "0-100", engagement_bar, icon=ft.Icons.PSYCHOLOGY, icon_color=ft.Colors.CYAN_300),
                            metric_card("Relajación", relaxation_value, "0-100", relaxation_bar, icon=ft.Icons.SPA, icon_color=ft.Colors.PURPLE_300),
                            metric_card("Calidad señal", quality_value, "0-100", quality_bar, icon=ft.Icons.GRAPHIC_EQ, icon_color=ft.Colors.BLUE_300),
                        ],
                    ),
                ],
            ),
        )
    )

    graphs_panel = ft.Card(
        content=ft.Container(
            padding=16,
            expand=True,
            content=ft.Column(
                expand=True,
                spacing=12,
                controls=[
                    ft.Text(
                        "Visualización de bioseñales y estados afectivos",
                        size=20,
                        weight=ft.FontWeight.BOLD,
                    ),
                    ft.Text(
                        "Interfaz experimental. Los índices son demostrativos para valencia, arousal, engagement y relajación.",
                        size=13,
                        color=ft.Colors.WHITE70,
                    ),
                    ft.Text("Señal bioeléctrica (2 canales)", size=15, weight=ft.FontWeight.BOLD),
                    ft.Container(height=250, content=signal_chart),
                    ft.Row(
                        wrap=True,
                        spacing=18,
                        run_spacing=6,
                        controls=[
                            ft.Row([ft.Container(width=12, height=12, bgcolor=ft.Colors.GREEN_300, border_radius=99), ft.Text("Valencia", size=12, color=ft.Colors.WHITE70)], spacing=6),
                            ft.Row([ft.Container(width=12, height=12, bgcolor=ft.Colors.AMBER_300, border_radius=99), ft.Text("Arousal", size=12, color=ft.Colors.WHITE70)], spacing=6),
                            ft.Row([ft.Container(width=12, height=12, bgcolor=ft.Colors.CYAN_300, border_radius=99), ft.Text("Engagement", size=12, color=ft.Colors.WHITE70)], spacing=6),
                            ft.Row([ft.Container(width=12, height=12, bgcolor=ft.Colors.PURPLE_300, border_radius=99), ft.Text("Relajación", size=12, color=ft.Colors.WHITE70)], spacing=6),
                        ],
                    )
                    ,
                    ft.Text("Valencia, Arousal, Engagement y Relajación", size=15, weight=ft.FontWeight.BOLD),
                    ft.Container(height=220, content=metric_chart),
                ],
            ),
        )
    )

    top_dashboard = ft.ResponsiveRow(
        controls=[
            ft.Column(col={"xs": 12, "lg": 8}, controls=[metrics_panel]),
            ft.Column(col={"xs": 12, "lg": 4}, controls=[status_panel]),
        ]
    )

    page.add(
        ft.Column(
            expand=True,
            spacing=10,
            controls=[
                top_dashboard,
                ft.ResponsiveRow(
                    expand=True,
                    controls=[
                        ft.Column(col={"xs": 12, "lg": 4}, controls=[controls_panel]),
                        ft.Column(col={"xs": 12, "lg": 8}, controls=[graphs_panel], expand=True),
                    ],
                ),
            ],
        )
    )

    refresh_phase_buttons()
    update_charts()


if __name__ == "__main__":
    ft.run(main)
