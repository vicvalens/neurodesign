import argparse
import threading
from collections import deque
import time
import os
from datetime import datetime

import numpy as np
from scipy.signal import welch, butter, sosfiltfilt, iirnotch, filtfilt
from scipy.interpolate import griddata, Rbf, make_interp_spline
from scipy.ndimage import gaussian_filter

# Flet for GUI
import flet as ft

# Matplotlib
import matplotlib
matplotlib.use("agg") # Use non-interactive backend for Flet
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import matplotlib.collections as mcoll
from matplotlib import colormaps
import io
import asyncio # Import asyncio
import base64 # Import base64 for image encoding

# LSL
try:
    from pylsl import resolve_byprop, resolve_streams, StreamInlet
except Exception:
    print("ERROR: pylsl no está instalado. Instala con: pip install pylsl")
    raise

# ===================== Parámetros por defecto =====================
DEFAULT_STREAM_NAME = "AURA"
DEFAULT_FS = 250
N_CH = 8
CH_NAMES = ["F3", "Fz", "F4", "C3", "Cz", "C4", "Pz", "Oz"]
CH_LOCS_2D = { # Standard 10/20 system 2D locations (simplified for visualization)
    "F3": (-0.35,  0.45),
    "Fz": ( 0.00,  0.52),
    "F4": ( 0.35,  0.45),

    "C3": (-0.45,  0.00),
    "Cz": ( 0.00,  0.00),
    "C4": ( 0.45,  0.00),

    "Pz": ( 0.00, -0.45),
    "Oz": ( 0.00, -0.70),
}

# Buffer / ventana
TIME_WINDOW_S = 20.0   # segundos visibles por defecto (más horizontal)
BUFFER_S      = 60.0   # historial almacenado para PSD

# Welch (PSD)
PSD_WIN_S   = 2.0
PSD_OVERLAP = 0.5

# Bandas
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 7),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
BAND_COLORS = {
    "delta": "#4C78A8",  # azul
    "theta": "#B279A2",  # morado
    "alpha": "#54A24B",  # verde
    "beta":  "#F58518",  # naranja
    "gamma": "#E45756",  # rojo
}
CH_COLORS = ["#7FDBFF", "#B10DC9", "#2ECC40", "#FF851B",
             "#FF4136", "#39CCCC", "#F012BE", "#01FF70"]


class EEGViewerAppFlet:
    def __init__(self, page: ft.Page, args):
        self.page = page
        self.page.title = "Monitor EEG"
        self.page.bgcolor = "white"
        self.page.theme_mode = ft.ThemeMode.LIGHT
        self.page.window_width = 1200
        self.page.window_height = 800
        # Args
        self.stream_name = args.name
        self.fs_expected = args.fs
        # self.update_hz   = args.update_hz # Now controlled by slider

        # Estado
        self.inlet     = None
        self.running   = False
        self.stop_flag = False
        self.freeze    = ft.Ref[ft.Checkbox]()
        self.show_bands = ft.Ref[ft.Checkbox]()
        self.mode      = ft.Ref[ft.Tabs]() # Use ft.Tabs for mode management

        # Controles de visualización
        self.update_hz_slider = ft.Ref[ft.Slider]() # Slider for update_hz
        self.use_micro = ft.Ref[ft.Checkbox]()
        self.gain_slider  = ft.Ref[ft.Slider]()
        self.auto_y    = ft.Ref[ft.Checkbox]()
        self.yrange_slider = ft.Ref[ft.Slider]()
        self.time_win_slider = ft.Ref[ft.Slider]()
        self.topomap_band_dropdown = ft.Ref[ft.Dropdown]()
        self.topomap_cmap_dropdown = ft.Ref[ft.Dropdown]()
        self.topomap_electrodes_dropdown = ft.Ref[ft.Dropdown]()
        # Buffer
        maxlen_samples = int(BUFFER_S * self.fs_expected)
        self.buff = deque(maxlen=maxlen_samples)
        self.lock = threading.Lock()
        self.band_power_uv2_per_ch = None # To store band powers for topomap

        # Nombres de canales (default). Se actualizarán si el stream aporta etiquetas.
        # Importante: el loop de actualización puede correr antes de conectar, así que
        # esto evita errores al dibujar topomaps sin stream.
        self.ch_names = list(CH_NAMES[:N_CH])

        # Matplotlib figures
        self.fig_time = Figure(figsize=(16, 8.5), dpi=100)
        self.ax_time, self.lines_time = [], []

        self.fig_psd = Figure(figsize=(16, 7.0), dpi=100)
        self.ax_psd, self.lines_psd = [], []
        self.band_patches = []

        self.fig_topomap = Figure(figsize=(8, 8), dpi=100)
        self.ax_topomap = None # Will be set in _init_topomap
        
        # Topomap specific attributes (from user's patch)
        self.topo_head_rx = 1.0         # Eje X de la cabeza (circular)
        self.topo_head_ry = 1.0         # Eje Y de la cabeza (circular)
        self.topo_grid_n = 220          # More resolution = smoother edge
        self.topo_clip_circle = None
        self.topo_cbar = None # Store the colorbar object
        self.topomap_scat = None
        self.topomap_interp = None
        self.topo_grid_x = None
        self.topo_grid_y = None
        self.topo_gx = None
        self.topo_gy = None
        self.topo_mask = None
        self.topo_cmap_name = 'turbo'
        self.topo_cmap = colormaps.get_cmap(self.topo_cmap_name)
        # Nota: 'turbo' es arcoíris pero más uniforme perceptualmente que 'jet'
        self.topo_cmap.set_bad(color='white', alpha=0)  # Fuera de la cabeza = transparente
        self.topo_contours = []
        self.topomap_last_ts = 0.0
        self.topomap_grid_prev = None
        self.topomap_last_band = None

        self.fig_band_power = Figure(figsize=(12, 6), dpi=100, layout='constrained') # New figure for band power bar chart
        self.ax_band_power = None # Will be set in _init_band_power_fig
        self.band_power_texts = []
        self.band_history = deque(maxlen=120000)  # historial amplio (~1h a 30 Hz) para no perder eventos
        self.record_window_n = 60  # tamaño de ventana para bandas de confianza en Record
        self.fig_record = Figure(figsize=(12, 6), dpi=100, layout='constrained')
        self.ax_record = None
        self.events = []
        self.event_input = ft.Ref[ft.TextField]()
        self.image_format = ft.Ref[ft.Dropdown]()
        self.theme_dropdown = ft.Ref[ft.Dropdown]()
        self.theme_label = "Claro"
        # Tema de gráficas (Matplotlib)
        self.plot_theme_label = "Claro"
        self.plot_bg_light = "white"
        self.plot_bg_dark = "#2b2b2b"   # mismo gris que el fondo oscuro de la app
        self.plot_fg_light = "#111111"
        self.plot_fg_dark = "#f2f2f2"
        self.plot_grid_dark = "#555555"
        self.plot_grid_light = "#dddddd"
        self.plot_bgcolor = self.plot_bg_light
        self.plot_fgcolor = self.plot_fg_light

        self.event_counter = 1
        # Filtros
        self.notch_enable = ft.Ref[ft.Checkbox]()
        self.notch_freq = ft.Ref[ft.Dropdown]()
        self.bandpass_enable = ft.Ref[ft.Checkbox]()
        self.bandpass_min = ft.Ref[ft.Slider]()
        self.bandpass_max = ft.Ref[ft.Slider]()
        self.notch_q = ft.Ref[ft.Slider]()
        # Calidad de señal
        self.quality_row = ft.Row(spacing=4, scroll=ft.ScrollMode.AUTO)
# Grabación
        self.recording = False
        self.record_data = []
        self.session_id = 0  # contador de sesiones dentro del mismo archivo antes de exportar
        self.fig_boxplot = Figure(figsize=(12, 6), dpi=100, layout='constrained')
        self.ax_boxplot = None


        self._build_ui()
        self._init_figs()
        self.page.update() # Initial page update
        asyncio.create_task(self._update_loop()) # Start Flet's update loop

    async def _update_loop(self):
        while True:
            await self._update()
            # Use the slider's current value for update_hz
            await asyncio.sleep(1 / (self.update_hz_slider.current.value if self.update_hz_slider.current.value > 0 else 1))

    def _matplotlib_to_flet_image(self, fig: Figure):
        if fig is None:
            return ft.Text("Figura no disponible")
        buf = io.BytesIO()
        try:
            # Asegura tema de la figura (fondo/textos) antes de renderizar
            try:
                self._apply_plot_theme_to_fig(fig)
            except Exception:
                pass
            # Fuerza un render real en Agg antes de exportar a bytes (evita imágenes en blanco)
            FigureCanvasAgg(fig).draw()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=150, facecolor=fig.get_facecolor())
        except Exception as exc:
            return ft.Text(f"Error renderizando figura: {exc}")
        return ft.Image(src_base64=base64.b64encode(buf.getvalue()).decode("utf-8"), fit=ft.ImageFit.CONTAIN, expand=True)

    # ---------------- UI ----------------

    def _refresh_topomap_canvas(self):
        """Actualiza el canvas del Topomap (sin romper si Flet cambia de versión)."""
        try:
            self.topomap_canvas.content = self._matplotlib_to_flet_image(self.fig_topomap)
            # En versiones nuevas, update() está en el control; en otras, usamos page.update()
            try:
                self.topomap_canvas.update()
            except Exception:
                self.page.update()
        except Exception:
            # Evita que el loop muera por un fallo de render.
            pass

    def _build_ui(self):
        # Top bar
        self.status_text = ft.Text("Desconectado")

        self.stream_info_text = ft.Text(f"Stream: {self.stream_name}  |  fs: {self.fs_expected} Hz  |  ch: {N_CH}")

        self.top_bar_container = ft.Container(
            content=ft.Row(
                [
                    self.stream_info_text,
                    self.status_text,
                ],
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            ),
            padding=ft.padding.only(left=8, right=8, top=6, bottom=6),
            bgcolor=self.plot_bgcolor,
        )

        # Controls
        self.connect_button = ft.ElevatedButton("Conectar", on_click=self.on_connect)
        self.stop_button = ft.ElevatedButton("Desconectar", on_click=self.on_stop, disabled=True)
        self.freeze_checkbox = ft.Checkbox(label="Congelar", ref=self.freeze, on_change=self._on_freeze_change)
        self.show_bands_checkbox = ft.Checkbox(label="Mostrar bandas", ref=self.show_bands, on_change=self._redraw_psd_bands)

        band_options = [ft.dropdown.Option(band) for band in BANDS.keys()]
        self.topomap_band_selector = ft.Dropdown(
            ref=self.topomap_band_dropdown,
            options=band_options,
            value="alpha",
            width=150,
            on_change=lambda e: self._draw_topomap(force=True)
        )

        # Selector de paleta de color (colormap) para el topomapa
        cmap_opts = [
            ft.dropdown.Option("turbo"),
            ft.dropdown.Option("viridis"),
            ft.dropdown.Option("plasma"),
            ft.dropdown.Option("inferno"),
            ft.dropdown.Option("magma"),
            ft.dropdown.Option("cividis"),
            ft.dropdown.Option("Spectral_r"),
            ft.dropdown.Option("RdYlBu_r"),
            ft.dropdown.Option("jet"),
        ]
        self.topomap_cmap_selector = ft.Dropdown(
            ref=self.topomap_cmap_dropdown,
            label="Colores",
            options=cmap_opts,
            value=self.topo_cmap_name,
            width=150,
            dense=True,
            on_change=self._on_topomap_cmap_change,
        )

        # Mostrar/ocultar electrodos (o dibujarlos en blanco)
        elec_opts = [
            ft.dropdown.Option("Color"),
            ft.dropdown.Option("Blanco"),
            ft.dropdown.Option("Ocultar"),
        ]
        self.topomap_electrodes_selector = ft.Dropdown(
            ref=self.topomap_electrodes_dropdown,
            label="Electrodos",
            options=elec_opts,
            value="Color",
            width=150,
            dense=True,
            on_change=self._on_topomap_electrodes_change,
        )


        # Modo de relleno del topomapa (con pocos electrodos)

        controls_column = ft.Column(
            [
                ft.Row(
                    [
                        self.connect_button,
                        self.stop_button,
                        self.freeze_checkbox,
                        self.show_bands_checkbox,
                        ft.ElevatedButton("Recalc PSD", on_click=self._force_recalc_psd),
                        ft.TextField(ref=self.event_input, width=110, label="Evento", dense=True),
                        ft.ElevatedButton("Marcar evento", on_click=self._on_mark_event),
                        ft.ElevatedButton("Iniciar Rec", on_click=self._start_record),
                        ft.ElevatedButton("Detener Rec", on_click=self._stop_record),
                        ft.ElevatedButton("Exportar CSV", on_click=self._export_csv),
                        ft.ElevatedButton("Guardar imagen", on_click=self._save_current_plot),
                        ft.Dropdown(ref=self.image_format, label="Formato", options=[ft.dropdown.Option("PNG"), ft.dropdown.Option("JPG")], value="PNG", width=110, dense=True),
                        ft.Dropdown(ref=self.theme_dropdown, label="Tema", options=[ft.dropdown.Option("Claro"), ft.dropdown.Option("Oscuro")], value=self.theme_label, width=120, dense=True, on_change=self._on_theme_change),
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=8
                ),
                ft.Row(
                    [
                        ft.Text("Actualización (Hz)"),
                        ft.Slider(
                            min=1, max=30, divisions=29, value=10, width=120,
                            ref=self.update_hz_slider, on_change=lambda e: self.page.update()
                        ),
                        ft.Text("Ventana (s)"),
                        ft.Slider(
                            min=2.0, max=30.0, divisions=28, value=TIME_WINDOW_S, width=140,
                            ref=self.time_win_slider, on_change=lambda e: self.page.update()
                        ),
                        ft.Checkbox(label="µV", ref=self.use_micro, value=True, on_change=lambda e: self.page.update()),
                        ft.Text("Gain"),
                        ft.Slider(
                            min=0.2, max=20.0, divisions=198, value=2.0, width=120,
                            ref=self.gain_slider, on_change=lambda e: self.page.update()
                        ),
                        ft.Checkbox(label="Auto eje Y", ref=self.auto_y, on_change=lambda e: self.page.update()),
                        ft.Text("Y µV"),
                        ft.Slider(
                            min=20.0, max=300.0, divisions=280, value=100.0, width=140,
                            ref=self.yrange_slider, on_change=lambda e: self.page.update()
                        ),
                        ft.Text("Topomap"),
                        ft.Container(self.topomap_band_selector, width=120),
                                                ft.Container(self.topomap_cmap_selector, width=160),
                        ft.Container(self.topomap_electrodes_selector, width=160),
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=8, scroll=ft.ScrollMode.AUTO,
                ),
                ft.Row(
                    [
                        ft.Container(
                            content=ft.Column(
                                [
                                    ft.Row(
                                        [
                                            ft.Text("Filtros:", weight=ft.FontWeight.W_600, size=12),
                                            ft.Checkbox(
                                                label="Notch",
                                                ref=self.notch_enable,
                                                value=False,
                                                on_change=lambda e: self._force_recalc_psd(e),
                                            ),
                                            ft.Dropdown(
                                                ref=self.notch_freq,
                                                options=[ft.dropdown.Option("50"), ft.dropdown.Option("60")],
                                                value="50",
                                                width=100,  # más ancho para que se lea bien 50/60
                                                on_change=lambda e: self._force_recalc_psd(e),
                                            ),
                                            ft.Text("Q", size=12),
                                            ft.Slider(
                                                ref=self.notch_q,
                                                min=10,
                                                max=50,
                                                value=30,
                                                divisions=8,
                                                width=130,
                                                on_change=lambda e: self._force_recalc_psd(e),
                                            ),
                                        ],
                                        alignment=ft.MainAxisAlignment.START,
                                        spacing=8, scroll=ft.ScrollMode.AUTO,
                                    ),
                                    ft.Row(
                                        [
                                            ft.Checkbox(
                                                label="Pasa-banda",
                                                ref=self.bandpass_enable,
                                                value=True,
                                                on_change=lambda e: self._force_recalc_psd(e),
                                            ),
                                            ft.Text("Min Hz", size=12),
                                            ft.Slider(
                                                ref=self.bandpass_min,
                                                min=0.5,
                                                max=20,
                                                value=1.0,
                                                divisions=39,
                                                width=140,
                                                on_change=lambda e: self._force_recalc_psd(e),
                                            ),
                                            ft.Text("Max Hz", size=12),
                                            ft.Slider(
                                                ref=self.bandpass_max,
                                                min=20,
                                                max=80,
                                                value=40.0,
                                                divisions=60,
                                                width=140,
                                                on_change=lambda e: self._force_recalc_psd(e),
                                            ),
                                        ],
                                        alignment=ft.MainAxisAlignment.START,
                                        spacing=8, scroll=ft.ScrollMode.AUTO,
                                    ),
                                ],
                                spacing=2,
                            ),
                            padding=ft.padding.all(6),
                            border=ft.border.all(1, "#dddddd"),
                            border_radius=8,
                        ),
                        ft.Container(
                            content=ft.Row(
                                [ft.Text("Quality:", size=12), self.quality_row],
                                spacing=6, scroll=ft.ScrollMode.AUTO,
                            ),
                            padding=ft.padding.only(left=8, top=6, bottom=6),
                            expand=True,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=12, scroll=ft.ScrollMode.AUTO,
                )
            ],
            spacing=6,
        )
        self.controls_container = ft.Container(
            content=controls_column,
            padding=ft.padding.only(left=8, right=8, top=6, bottom=6),
            bgcolor="#f7f7f7",
            border_radius=10,
        )

        # Matplotlib canvases
        self.time_canvas = ft.Container(content=ft.Text("Time Plot"), expand=True, bgcolor=self.plot_bgcolor, padding=10)
        self.psd_canvas = ft.Container(content=ft.Text("Spectrum Plot"), expand=True, bgcolor=self.plot_bgcolor, padding=10)
        self.topomap_canvas = ft.Container(content=ft.Text("Topomap Plot"), expand=True, bgcolor=self.plot_bgcolor, padding=10)
        self.band_power_canvas = ft.Container(content=ft.Text("Band Power Bars"), expand=True, bgcolor=self.plot_bgcolor, padding=10) # New canvas
        self.boxplot_canvas = ft.Container(content=ft.Text("Boxplots"), expand=True, bgcolor=self.plot_bgcolor, padding=10)
        self.record_canvas = ft.Container(content=ft.Text("Record"), expand=True, bgcolor=self.plot_bgcolor, padding=10)

        # Lista para actualizar el fondo de las áreas de gráficas según el tema
        self._plot_containers = [self.time_canvas, self.psd_canvas, self.band_power_canvas, self.record_canvas, self.boxplot_canvas, self.topomap_canvas]


        self.tabs = ft.Tabs(
            ref=self.mode,
            selected_index=0,
            animation_duration=300,
            on_change=self._on_tab_change,
            tabs=[
                ft.Tab(text="Tiempo", content=self.time_canvas),
                ft.Tab(text="Espectro", content=self.psd_canvas),
                ft.Tab(
                    text="Metricas",
                    content=ft.Row(
                        [
                            ft.Container(
                                content=self.band_power_canvas,
                                expand=1,
                                padding=8,
                            ),
                        ],
                        expand=True,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                    ),
                ),
                ft.Tab(text="Registro", content=self.record_canvas),
                ft.Tab(text="Cajas", content=self.boxplot_canvas),
                ft.Tab(text="Topomapa", content=self.topomap_canvas),
            ],
            expand=1
        )


        self.page.add(
            self.top_bar_container,
            self.controls_container,
            self.tabs
        )

    # ---------------- Figuras ----------------
    
    def _on_topomap_cmap_change(self, e):
        """Cambia la paleta de colores del topomapa en caliente."""
        try:
            self.topo_cmap_name = e.control.value
        except Exception:
            return
        try:
            self.topo_cmap = colormaps.get_cmap(self.topo_cmap_name)
        except Exception:
            self.topo_cmap_name = "turbo"
            self.topo_cmap = colormaps.get_cmap(self.topo_cmap_name)

        # Actualiza colormap de los artistas existentes
        try:
            if hasattr(self, "topomap_interp") and self.topomap_interp is not None:
                self.topomap_interp.set_cmap(self.topo_cmap_name)
            if hasattr(self, "topomap_scat") and self.topomap_scat is not None:
                self.topomap_scat.set_cmap(self.topo_cmap_name)
            if getattr(self, "topo_cbar", None) is not None and getattr(self, "topomap_interp", None) is not None:
                self.topo_cbar.update_normal(self.topomap_interp)
        except Exception:
            pass

        # Redibuja solo si estamos corriendo
        try:
            self._draw_topomap(force=True)
        except Exception:
            pass

    
    def _on_theme_change(self, e):
        """Alterna entre tema Claro y Oscuro."""
        try:
            self.theme_label = e.control.value or "Claro"
        except Exception:
            self.theme_label = "Claro"
        # Tema de gráficas (Matplotlib)
        self.plot_theme_label = "Claro"
        self.plot_bg_light = "white"
        self.plot_bg_dark = "#2b2b2b"   # mismo gris que el fondo oscuro de la app
        self.plot_fg_light = "#111111"
        self.plot_fg_dark = "#f2f2f2"
        self.plot_grid_dark = "#555555"
        self.plot_grid_light = "#dddddd"
        self.plot_bgcolor = self.plot_bg_light
        self.plot_fgcolor = self.plot_fg_light

        self._apply_theme(self.theme_label)

    def _apply_theme(self, mode_label: str):
        """Aplica tema a la app (Flet)."""
        if str(mode_label).lower().startswith("oscu"):
            self.page.theme_mode = ft.ThemeMode.DARK
            self.page.bgcolor = "#2b2b2b"
            try:
                self.top_bar_container.bgcolor = "#2b2b2b"
                self.controls_container.bgcolor = "#3a3a3a"
            except Exception:
                pass
        else:
            self.page.theme_mode = ft.ThemeMode.LIGHT
            self.page.bgcolor = "white"
            try:
                self.top_bar_container.bgcolor = "white"
                self.controls_container.bgcolor = "#f7f7f7"
            except Exception:
                pass
        try:
            self._apply_plot_theme(mode_label)
        except Exception:
            pass
        try:
            self.page.update()
        except Exception:
            pass


    # ---- Tema de gráficas (Matplotlib) ----
    def _apply_plot_theme(self, mode_label: str):
        """Configura colores de Matplotlib para que coincidan con el tema de la UI."""
        if str(mode_label).lower().startswith("oscu"):
            self.plot_theme_label = "Oscuro"
            self.plot_bgcolor = self.plot_bg_dark
            self.plot_fgcolor = self.plot_fg_dark
            grid = self.plot_grid_dark
            edge = "#aaaaaa"
        else:
            self.plot_theme_label = "Claro"
            self.plot_bgcolor = self.plot_bg_light
            self.plot_fgcolor = self.plot_fg_light
            grid = self.plot_grid_light
            edge = "#333333"

        # Ajustes globales (afecta nuevas figuras y también ayuda a consistencia al redibujar)
        try:
            import matplotlib as mpl
            mpl.rcParams.update({
                "figure.facecolor": self.plot_bgcolor,
                "axes.facecolor": self.plot_bgcolor,
                "savefig.facecolor": self.plot_bgcolor,
                "savefig.edgecolor": self.plot_bgcolor,
                "text.color": self.plot_fgcolor,
                "axes.labelcolor": self.plot_fgcolor,
                "axes.edgecolor": edge,
                "xtick.color": self.plot_fgcolor,
                "ytick.color": self.plot_fgcolor,
                "grid.color": grid,
                "legend.facecolor": self.plot_bgcolor,
                "legend.edgecolor": edge,
            })
        except Exception:
            pass

        # Aplica a figuras existentes y al fondo de los contenedores
        try:
            self._update_plot_containers_bg()
            self._apply_plot_theme_to_fig(self.fig_time)
            self._apply_plot_theme_to_fig(self.fig_psd)
            self._apply_plot_theme_to_fig(self.fig_record)
            self._apply_plot_theme_to_fig(self.fig_band_power)
            self._apply_plot_theme_to_fig(self.fig_boxplot)
            self._apply_plot_theme_to_fig(self.fig_topomap)
        except Exception:
            pass

    def _update_plot_containers_bg(self):
        """Ajusta el fondo del área donde se renderizan las imágenes de las gráficas."""
        try:
            bg = self.plot_bgcolor
            for c in getattr(self, "_plot_containers", []):
                try:
                    c.bgcolor = bg
                except Exception:
                    pass
        except Exception:
            pass

    
    def _apply_plot_theme_to_fig(self, fig: Figure):
        """Ajusta fondo, ejes, textos, spines y leyendas para una figura existente.
        Nota: evitamos modificar colecciones globalmente porque puede romper colormaps/colorbars.
        """
        if fig is None:
            return
        is_dark = (self.plot_theme_label == "Oscuro")
        bg = self.plot_bgcolor
        fg = self.plot_fgcolor

        # Fondo de figura
        try:
            fig.patch.set_facecolor(bg)
        except Exception:
            pass

        for ax in list(getattr(fig, "axes", [])):
            try:
                ax.set_facecolor(bg)

                # ticks / labels / title
                ax.tick_params(colors=fg, which="both")
                if ax.title:
                    ax.title.set_color(fg)
                ax.xaxis.label.set_color(fg)
                ax.yaxis.label.set_color(fg)

                # spines
                for sp in ax.spines.values():
                    try:
                        sp.set_color(fg)
                    except Exception:
                        pass

                # grid: sólo en ejes "normales"; en colorbar suele estorbar, pero no hace daño.
                try:
                    ax.grid(True, alpha=0.25)
                except Exception:
                    pass

                # legend
                leg = ax.get_legend()
                if leg is not None:
                    try:
                        leg.get_frame().set_facecolor(bg)
                        leg.get_frame().set_edgecolor(fg)
                        for t in leg.get_texts():
                            t.set_color(fg)
                    except Exception:
                        pass

                # Patches (ej. contorno de la cabeza en topomap)
                try:
                    for patch in getattr(ax, "patches", []):
                        try:
                            if hasattr(patch, "set_edgecolor"):
                                # En claro mantenemos negros si ya eran negros
                                if not is_dark:
                                    continue
                                patch.set_edgecolor(fg)
                        except Exception:
                            pass
                except Exception:
                    pass

                # Líneas: en modo oscuro, si alguna quedó en negro, pásala a color de primer plano
                if is_dark:
                    try:
                        for ln in getattr(ax, "lines", []):
                            try:
                                c = str(ln.get_color()).lower()
                                if c in ("k", "black", "#000", "#000000"):
                                    ln.set_color(fg)
                            except Exception:
                                pass
                    except Exception:
                        pass

                # Colecciones: SOLO tocamos PathCollection (scatter) para ajustar el borde.
                # No tocamos QuadMesh/LineCollection (colormaps, contornos), para no “borrar” colorbars.
                try:
                    for coll in getattr(ax, "collections", []):
                        try:
                            if isinstance(coll, mcoll.PathCollection):
                                desired_edge = fg if is_dark else "#000000"
                                coll.set_edgecolor(desired_edge)
                        except Exception:
                            pass
                except Exception:
                    pass

            except Exception:
                continue

        # Ajustes específicos para topomap (revertir contorno al alternar tema)
        try:
            if fig is self.fig_topomap:
                self._set_topomap_outline_colors()
        except Exception:
            pass



    def _set_topomap_outline_colors(self):
        """Ajusta colores del contorno (cabeza/nariz/orejas) y contornos internos según el tema.
        - Tema claro: cabeza/nariz/orejas en negro, contornos internos en blanco.
        - Tema oscuro: todo (cabeza/nariz/orejas/contornos internos) en blanco.
        """
        is_dark = (self.plot_theme_label == "Oscuro")
        outline = "white" if is_dark else "black"
        inner = "white"  # según el requisito actual

        # Contorno de cabeza (Ellipse)
        try:
            if self.topo_head_ellipse is not None:
                self.topo_head_ellipse.set_edgecolor(outline)
        except Exception:
            pass

        # Nariz (Line2D)
        try:
            if self.topo_nose_line is not None:
                self.topo_nose_line.set_color(outline)
        except Exception:
            pass

        # Orejas (Arc patches)
        try:
            for ear in (self.topo_ear_arcs or []):
                try:
                    ear.set_edgecolor(outline)
                except Exception:
                    pass
        except Exception:
            pass

        # Contornos internos (si existen)
        try:
            for cs in (self.topo_contours or []):
                cols = getattr(cs, "collections", [])
                for coll in cols:
                    try:
                        coll.set_color(inner)
                    except Exception:
                        pass
        except Exception:
            pass

        # Colorbar estilo (ticks/label/outline)
        try:
            if self.topo_cbar is not None:
                fg = self.plot_fgcolor
                self.topo_cbar.ax.tick_params(colors=fg)
                self.topo_cbar.ax.yaxis.label.set_color(fg)
                self.topo_cbar.outline.set_edgecolor(fg)
        except Exception:
            pass

    def _on_topomap_electrodes_change(self, e):
        """Alterna visibilidad/modo de los marcadores de electrodos."""
        try:
            self._draw_topomap(force=True)
        except Exception:
            pass
    def _init_figs(self):
        # ===== TIME (stacked 8x1) =====
        self.ax_time.clear() # Clear existing axes
        self.ax_time, self.lines_time = [], []

        for i in range(N_CH):
            share = self.ax_time[0] if i > 0 else None
            ax = self.fig_time.add_subplot(8, 1, i+1, sharex=share)
            ax.grid(True, alpha=0.15)
            ax.set_xlim(0, TIME_WINDOW_S)
            ax.set_ylim(-100e-6, 100e-6)  # ±100 µV inicial
            ax.set_ylabel(CH_NAMES[i] if i < len(CH_NAMES) else f"Ch{i+1}",
                          rotation=0, ha="right", va="center", labelpad=25)
            if i < N_CH-1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel("Time (s)")

            line, = ax.plot([], [], lw=1.1, solid_capstyle="round")
            line.set_color(CH_COLORS[i % len(CH_COLORS)])
            self.ax_time.append(ax); self.lines_time.append(line)

        self.fig_time.subplots_adjust(top=0.97, bottom=0.08, left=0.10, right=0.98, hspace=0.08)
        self.time_canvas.content = self._matplotlib_to_flet_image(self.fig_time)


        # ===== SPECTRUM (2x4) =====
        self.ax_psd.clear() # Clear existing axes
        self.ax_psd, self.lines_psd = [], []
        for i in range(N_CH):
            ax = self.fig_psd.add_subplot(2, 4, i+1)
            ax.set_title(CH_NAMES[i] if i < len(CH_NAMES) else f"Ch{i+1}")
            ax.set_xlim(0, 60)
            ax.set_ylim(1e-12, 1e-3)
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            line, = ax.plot([], [], lw=1.0, color=CH_COLORS[i % len(CH_COLORS)])
            self.ax_psd.append(ax); self.lines_psd.append(line)
            if i//4 == 1: ax.set_xlabel("Hz")
            if i%4 == 0:  ax.set_ylabel("PSD (V²/Hz)")
        self.fig_psd.subplots_adjust(top=0.95, bottom=0.08, left=0.07, right=0.99, wspace=0.18, hspace=0.28)
        self._create_psd_bands()
        self.psd_canvas.content = self._matplotlib_to_flet_image(self.fig_psd)


        # ===== TOPOMAP =====
        self.ax_topomap = self.fig_topomap.add_subplot(111)
        self._init_topomap()
        self.topomap_canvas.content = self._matplotlib_to_flet_image(self.fig_topomap)

        # ===== BAND POWER BARS =====
        self.ax_band_power = self.fig_band_power.add_subplot(111)
        self._init_band_power_fig()
        self.band_power_canvas.content = self._matplotlib_to_flet_image(self.fig_band_power)

        # ===== RECORD (histórico de bandas) =====
        self.ax_record = self.fig_record.add_subplot(111)
        self._init_record_fig()
        self.record_canvas.content = self._matplotlib_to_flet_image(self.fig_record)

        # ===== BOXPLOTS =====
        self.ax_boxplot = self.fig_boxplot.add_subplot(111)
        self._init_boxplot_fig()
        self.boxplot_canvas.content = self._matplotlib_to_flet_image(self.fig_boxplot)

    def _init_topomap(self):
        ax = self.ax_topomap
        ax.clear()
        ax.set_title("EEG Topomap")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect('equal', adjustable='box')

        r_x = self.topo_head_rx
        r_y = self.topo_head_ry

        # Head outline (lo guardamos para clip)
        is_dark = (self.plot_theme_label == "Oscuro")
        outline = "white" if is_dark else "black"

        head_ellipse = mpatches.Ellipse((0, 0), width=2*r_x, height=2*r_y, fill=False, ec=outline, lw=2, zorder=5)
        ax.add_patch(head_ellipse)
        self.topo_clip_circle = head_ellipse
        self.topo_head_ellipse = head_ellipse

        # Nose + ears (proporcionales al radio)
        nose_line, = ax.plot([-0.10*r_x, 0, 0.10*r_x], [r_y, 1.10*r_y, r_y], color=outline, lw=2, zorder=6)
        self.topo_nose_line = nose_line
        ear_r = 0.12 * r_x
        self.topo_ear_arcs = []
        # Arcos de oreja que sólo se dibujan fuera de la cabeza (evita trazar dentro del mapa)
        ear_left = mpatches.Arc((-r_x * 1.02, 0), 2*ear_r, 2*ear_r, theta1=90, theta2=270, ec=outline, lw=2, zorder=6)
        ear_right = mpatches.Arc(( r_x * 1.02, 0), 2*ear_r, 2*ear_r, theta1=-90, theta2=90, ec=outline, lw=2, zorder=6)
        ax.add_patch(ear_left)
        ax.add_patch(ear_right)
        self.topo_ear_arcs.extend([ear_left, ear_right])

        ax.set_xlim(-1.25*r_x, 1.25*r_x)
        ax.set_ylim(-1.30*r_y, 1.30*r_y)

        # Imagen interpolada (crear una vez y actualizar con set_data)
        n = self.topo_grid_n
        dummy = np.zeros((n, n), dtype=float)
        self.topo_grid_x = np.linspace(-r_x, r_x, n)
        self.topo_grid_y = np.linspace(-r_y, r_y, n)
        self.topo_gx, self.topo_gy = np.meshgrid(self.topo_grid_x, self.topo_grid_y)
        self.topo_mask = (self.topo_gx**2 / r_x**2 + self.topo_gy**2 / r_y**2) > 1.0
        self.topomap_interp = ax.imshow(
            dummy,
            extent=(-r_x, r_x, -r_y, r_y),
            origin='lower',
            cmap=self.topo_cmap_name,            # mejor que jet (si quieres arcoíris, turbo es buena opción)
            interpolation='bicubic',
            alpha=0.95,
            zorder=1
        )
        self.topomap_interp.set_clip_path(self.topo_clip_circle)

        # Sensores (scatter) arriba del mapa
        self.topomap_scat = ax.scatter(
            [], [],
            c=[],
            cmap=self.topo_cmap_name,
            s=90,
            edgecolor='k',
            linewidth=0.6,
            zorder=10
        )

        # Marcadores alternativos (blancos) para ocultar el color de los electrodos si se desea
        self.topomap_scat_white = ax.scatter(
            [], [],
            c="white",
            s=90,
            edgecolor="k",
            linewidth=0.6,
            zorder=11
        )
        self.topomap_scat_white.set_visible(False)

        # Colorbar: que dependa de la imagen (no del scatter)
        if self.topo_cbar is None:
            self.topo_cbar = self.fig_topomap.colorbar(
                self.topomap_interp, ax=ax, orientation='vertical', fraction=0.046, pad=0.04
            )
            self.topo_cbar.set_label("Band power (log10 µV²)")
            try:
                self.topo_cbar.ax.tick_params(colors=self.plot_fgcolor)
                self.topo_cbar.ax.yaxis.label.set_color(self.plot_fgcolor)
                self.topo_cbar.outline.set_edgecolor(self.plot_fgcolor)
            except Exception:
                pass

    def _init_band_power_fig(self):
        ax = self.ax_band_power
        ax.clear()
        ax.set_title("Band Power (µV²)")
        ax.set_ylabel("Power (µV²)")
        ax.set_yscale("log")
        ax.set_ylim(0.01, 1000) # Establecer un rango válido para la escala logarítmica (e.g., 0.01 a 1000 µV^2)
        ax.grid(True, alpha=0.3)

    def _init_boxplot_fig(self):
        ax = self.ax_boxplot
        ax.clear()
        ax.set_title("Band Power boxplots (log10 µV²)")
        ax.set_ylabel("log10 Power (µV²)")
        ax.grid(True, alpha=0.3)

    def _init_record_fig(self):
        ax = self.ax_record
        ax.clear()
        ax.set_title("Histórico de bandas (stacked)")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Power (µV²)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    # ---------------- LSL Connection ----------------
    def on_connect(self, e):
        if self.running:
            return
        try:
            info = None

            # 1) Si el usuario pasó un nombre explícito (distinto de 'auto'), intenta por nombre
            if self.stream_name and str(self.stream_name).strip().lower() not in ["", "auto"]:
                streams = resolve_byprop("name", self.stream_name, timeout=2)
                if streams:
                    info = streams[0]

            # 2) Si no hay coincidencia, intenta por tipo EEG
            if info is None:
                eeg_streams = resolve_byprop("type", "EEG", timeout=3)
                if eeg_streams:
                    info = eeg_streams[0]

            # 3) Fallback: primer stream disponible
            if info is None:
                all_streams = resolve_streams(timeout=3)
                if all_streams:
                    info = all_streams[0]

            if info is None:
                raise RuntimeError("No se detectó ningún stream LSL.")

            # Toma el nombre real detectado
            self.stream_name = info.name()
            self.inlet = StreamInlet(info, max_buflen=int(BUFFER_S))

        except Exception as ex:
            self.status_text.value = f"Error al conectar: {ex}"
            self.page.update()
            return

        self.running = True
        self.stop_flag = False
        self.connect_button.disabled = True
        self.stop_button.disabled = False
        self.status_text.value = f"Conectado a {self.stream_name}"
        self.events = []  # reinicia log de eventos al conectar
        self.event_counter = 1

        # Actualiza barra superior con el nombre detectado
        try:
            self.stream_info_text.value = f"Stream: {self.stream_name}  |  fs: {self.fs_expected} Hz  |  ch: {N_CH}"
        except Exception:
            pass

        self.page.update()
        threading.Thread(target=self._acquire_loop, daemon=True).start()


    def on_stop(self, e):
        self.stop_flag = True
        self.running = False
        self.connect_button.disabled = False
        self.stop_button.disabled = True
        self.status_text.value = "Desconectado"
        self.page.update()

    def _acquire_loop(self):
        while not self.stop_flag and self.inlet is not None:
            sample, ts = self.inlet.pull_sample(timeout=0.2)
            if sample is None: continue
            s = sample[:N_CH] if len(sample) >= N_CH else list(sample) + [0.0]*(N_CH-len(sample))
            with self.lock:
                self.buff.append(s)

    # ---------------- Update Loop ----------------
    def _on_tab_change(self, e):
        current_tab_text = self.tabs.tabs[self.tabs.selected_index].text.lower()
        if current_tab_text == "metrics":
            self._draw_band_power()
        self.page.update()

    def _set_mode(self, m):
        # This function is primarily for internal state. Tab changes handle UI selection.
        # Ensure _update is called to refresh content.
        pass

    def _force_recalc_psd(self, e):
        self._draw_psd(force=True)
        self.page.update()

    def _on_mark_event(self, e):
        label = (self.event_input.current.value or "").strip()
        if not label:
            label = f"Evt {self.event_counter}"
        self.event_counter += 1
        self.events.append({"ts": time.time(), "label": label})
        # Feedback en la UI
        self.status_text.value = f"Evento marcado: {label}"
        self.page.update()

    def _start_record(self, e):
        # Inicia una nueva sesión de grabación (se acumula en memoria hasta exportar).
        if not self.running:
            self.status_text.value = "Primero conecta a un stream LSL."
            self.page.update()
            return
        if self.recording:
            self.status_text.value = "Ya estás grabando."
            self.page.update()
            return
        self.session_id += 1
        self.recording = True
        self.status_text.value = f"Grabando sesión {self.session_id}..."
        self.page.update()

    def _stop_record(self, e):
        if not self.recording:
            self.status_text.value = "No hay grabación activa."
            self.page.update()
            return
        self.recording = False
        self.status_text.value = f"Grabación detenida (sesión {self.session_id}). Puedes iniciar otra sesión o exportar."
        self.page.update()

    def _export_csv(self, e):
        if self.recording:
            # Fuerza detener antes de exportar para evitar filas incompletas.
            self.recording = False

        if not self.record_data:
            self.status_text.value = "No hay datos grabados para exportar."
            self.page.update()
            return

        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"monitor_eeg_sesiones_{ts_tag}.csv"
        path = os.path.join(os.path.dirname(__file__), filename)

        band_names = list(BANDS.keys())
        headers = ["sesion", "timestamp"] + [f"{ch}_{band}" for ch in CH_NAMES for band in band_names] + ["evento"]
        # Mapea eventos al registro (columna 'evento') usando el timestamp más cercano.
        times = []
        try:
            times = [float(r[1]) for r in self.record_data]
        except Exception:
            times = []
        event_labels = ["" for _ in range(len(self.record_data))]
        if times and self.events:
            import bisect as _bisect
            for evt in self.events:
                t_evt = evt.get("ts", None)
                if t_evt is None:
                    continue
                try:
                    t_evt = float(t_evt)
                except Exception:
                    continue
                idx = _bisect.bisect_left(times, t_evt)
                candidates = []
                if 0 <= idx < len(times):
                    candidates.append(idx)
                if idx - 1 >= 0:
                    candidates.append(idx - 1)
                if not candidates:
                    continue
                best = min(candidates, key=lambda kk: abs(times[kk] - t_evt))
                label = str(evt.get("label", "")).strip()
                if not label:
                    label = "Evento"
                if event_labels[best]:
                    event_labels[best] += ";" + label
                else:
                    event_labels[best] = label


        try:
            import csv
            from numbers import Real
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=",")
                writer.writerow(headers)

                n_cols = len(headers)
                for i, row in enumerate(self.record_data):
                    # Asegura exactamente una columna 'evento' al final
                    evento = event_labels[i] if i < len(event_labels) else ""
                    out = list(row) + [evento]

                    # Normaliza longitud por seguridad (evita columnas extra)
                    if len(out) < n_cols:
                        out += [""] * (n_cols - len(out))
                    elif len(out) > n_cols:
                        out = out[:n_cols]

                    out_str = []
                    for j, v in enumerate(out):
                        if j == 0:
                            # sesión: entero sin decimales
                            try:
                                out_str.append(str(int(round(float(v)))))
                            except Exception:
                                out_str.append(str(v))
                        elif j == 1:
                            # timestamp: conserva decimales
                            try:
                                out_str.append(f"{float(v):.6f}")
                            except Exception:
                                out_str.append(str(v))
                        else:
                            if isinstance(v, Real) and not isinstance(v, bool):
                                out_str.append(f"{float(v):.6f}")
                            else:
                                out_str.append(str(v))
                    writer.writerow(out_str)

            self.status_text.value = f"CSV exportado: {filename}  (filas: {len(self.record_data)})"
            # Limpia para comenzar un nuevo archivo
            self.record_data = []
            self.session_id = 0
            self.events = []
            self.event_counter = 1
        except Exception as exc:
            self.status_text.value = f"Error al exportar: {exc}"

        self.page.update()
    def _save_current_plot(self, e):
        # Guarda la gráfica visible en este momento como PNG/JPG.
        try:
            tab_text = self.tabs.tabs[self.tabs.selected_index].text.lower()
        except Exception:
            tab_text = ""

        fmt = "PNG"
        try:
            if self.image_format.current and self.image_format.current.value:
                fmt = str(self.image_format.current.value).upper()
        except Exception:
            pass

        ext = "png" if fmt == "PNG" else "jpg"

        fig_map = {
            "tiempo": self.fig_time,
            "espectro": self.fig_psd,
            "metricas": self.fig_band_power,
            "métricas": self.fig_band_power,
            "registro": self.fig_record,
            "boxplots": self.fig_boxplot,
            "cajas": self.fig_boxplot,
            "topomapa": self.fig_topomap,
            "topomap": self.fig_topomap,  # fallback si aún está en inglés
            "time": self.fig_time,
            "spectrum": self.fig_psd,
            "metrics": self.fig_band_power,
            "record": self.fig_record,
        }

        fig = fig_map.get(tab_text)
        if fig is None:
            self.status_text.value = f"No se pudo guardar: pestaña desconocida ({tab_text})."
            self.page.update()
            return

        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_tab = tab_text.replace(" ", "_").replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
        filename = f"captura_{safe_tab}_{ts_tag}.{ext}"
        path = os.path.join(os.path.dirname(__file__), filename)

        try:
            # Asegura que la figura esté actualizada antes de guardar (y que los eventos sigan visibles).
            if tab_text in ("registro", "record"):
                try:
                    self._draw_record()
                except Exception:
                    pass
            elif tab_text in ("espectro", "spectrum"):
                try:
                    self._draw_psd(force=True, render=True)
                except Exception:
                    pass
            elif tab_text in ("tiempo", "time"):
                try:
                    self._draw_time()
                except Exception:
                    pass
            elif tab_text in ("topomapa", "topomap"):
                try:
                    self._draw_topomap(force=True)
                except Exception:
                    pass

            # Aplica tema (fondo/textos) y fuerza render real en Agg ANTES de guardar
            try:
                self._apply_plot_theme_to_fig(fig)
            except Exception:
                pass
            try:
                FigureCanvasAgg(fig).draw()
            except Exception as exc:
                raise RuntimeError(f"No se pudo renderizar la figura: {exc}")

            fig.savefig(
                path,
                format=ext,
                bbox_inches="tight",
                pad_inches=0.1,
                dpi=150,
                facecolor=fig.get_facecolor(),
                transparent=False,
            )
            self.status_text.value = f"Imagen guardada: {filename}"
        except Exception as exc:
            self.status_text.value = f"Error al guardar imagen: {exc}"

        self.page.update()

    def _on_freeze_change(self, e):
        # No necesitamos llamar a la corrutina; el loop principal se encarga.
        self.page.update()

    async def _update(self):
        if not self.running or self.freeze.current.value:
            return

        current_tab_text = self.tabs.tabs[self.tabs.selected_index].text.lower()
        # Si estamos grabando, calculamos bandas aunque no estés en 'Espectro' (para que el CSV no salga vacío).
        if self.recording and current_tab_text not in ("espectro", "spectrum"):
            self._draw_psd(render=False)


        if current_tab_text in ("tiempo", "time"):
            self._draw_time()
        elif current_tab_text in ("espectro", "spectrum"):
            self._draw_psd()
        elif current_tab_text in ("metricas", "metrics"):
            # Llama _draw_psd para asegurar que band_power_uv2_per_ch esté actualizado
            self._draw_psd()
            self._draw_band_power()
        elif current_tab_text in ("registro", "record"):
            self._draw_psd()
            self._draw_record()
        elif current_tab_text in ("cajas", "boxplots"):
            self._draw_psd()
            self._draw_boxplots()
        elif current_tab_text in ("topomapa", "topomap"):
            self._draw_topomap()

        self.page.update()



    # ---------------- Draw: Time ----------------
    def _draw_time(self):
        with self.lock:
            if not self.buff: return
            arr = np.array(self.buff)  # (n, ch)

        arr = self._apply_filters(arr, self.fs_expected)
        quality = self._compute_quality(arr, self.fs_expected)
        self._render_quality(quality)

        fs = self.fs_expected
        win_s = float(self.time_win_slider.current.value)
        nwin = int(win_s * fs)
        arr = arr[-nwin:] if len(arr) > nwin else arr
        t = np.arange(len(arr)) / fs
        if len(t) == 0: return
        t = t - t[0]

        to_micro = 1e6 if self.use_micro.current.value else 1.0
        gain = float(self.gain_slider.current.value)

        for i in range(N_CH):
            y = arr[:, i] if i < arr.shape[1] else np.zeros_like(t)
            y_plot = y * gain * to_micro
            self.lines_time[i].set_data(t, y_plot)

            if self.auto_y.current.value:
                if len(y_plot) > 10:
                    p1, p99 = np.percentile(y_plot, [1, 99])
                    if p1 == p99: p1 -= 1.0; p99 += 1.0
                    pad = 0.3 * (p99 - p1)
                    self.ax_time[i].set_ylim(p1 - pad, p99 + pad)
            else:
                yr = float(self.yrange_slider.current.value)
                if not self.use_micro.current.value:  # si estás en Voltios, convierte µV a V
                    yr = yr * 1e-6
                self.ax_time[i].set_ylim(-yr, yr)

            self.ax_time[i].set_xlim(0, t[-1] if t[-1] > 1e-3 else win_s)

            # formatea ticks Y
            if self.use_micro.current.value:
                self.ax_time[i].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.0f'))
            else:
                self.ax_time[i].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

        self.time_canvas.content = self._matplotlib_to_flet_image(self.fig_time)


    # ---------------- Draw: PSD + Metrics ----------------
    def _draw_psd(self, force=False, render=True):
        with self.lock:
            if not self.buff: return
            arr = np.array(self.buff)
        fs = self.fs_expected
        nseg = int(PSD_WIN_S * fs)
        if len(arr) < nseg: return
        x = arr[-nseg:]  # (nseg, ch)
        x = self._apply_filters(x, fs)

        # Dibujar PSDs
        for i in range(N_CH):
            y = x[:, i]
            if np.allclose(y, 0.0):
                f = np.linspace(0, fs/2, 10); pxx = np.ones_like(f)*1e-12
            else:
                f, pxx = welch(y, fs=fs, nperseg=nseg, noverlap=int(PSD_OVERLAP*nseg),
                               scaling="density", return_onesided=True)
            self.lines_psd[i].set_data(f, pxx)
            mask = (f >= 0.5) & (f <= 60)
            if np.any(mask):
                ymin = max(np.nanmin(pxx[mask]), 1e-12)
                ymax = max(np.nanmax(pxx[mask]), ymin*10)
                self.ax_psd[i].set_xlim(0, 60)
                self.ax_psd[i].set_ylim(ymin, ymax)

        if render:
            self._apply_band_patches()
            self.psd_canvas.content = self._matplotlib_to_flet_image(self.fig_psd)


        # === Métricas por banda (integración PSD) ===
        band_power_uv2_per_ch = []
        for i in range(N_CH):
            line = self.lines_psd[i]
            f = line.get_xdata(); pxx = line.get_ydata()
            bp = {"delta":0.0,"theta":0.0,"alpha":0.0,"beta":0.0,"gamma":0.0}
            if f is not None and len(f) > 1:
                for name, (f1, f2) in BANDS.items():
                    m = (f >= f1) & (f <= f2)
                    if np.any(m):
                        area_v2 = np.trapezoid(pxx[m], f[m])   # V^2
                        bp[name] = max(float(area_v2) * 1e12, 0.0)  # a µV^2
            band_power_uv2_per_ch.append(bp)

        self.band_power_uv2_per_ch = band_power_uv2_per_ch # Store for topomap
        self._record_band_history(band_power_uv2_per_ch)
        self._record_row(band_power_uv2_per_ch)
        if render:
            self._draw_band_power()


    # ---------------- Bandas coloreadas ----------------
    def _create_psd_bands(self):
        self.band_patches = []
        for ax in self.ax_psd:
            patches_ax = []
            for name, (f1, f2) in BANDS.items():
                color = BAND_COLORS.get(name, "gray")
                rect = mpatches.Rectangle((f1, 1e-12), f2 - f1, 1.0,
                                                    alpha=0.3, color=color, label=name) # Increased alpha
                ax.add_patch(rect)
                patches_ax.append(rect)
            self.band_patches.append(patches_ax)

        # Leyenda en el primer subplot (compatibilidad de API)
        handles = [mpatches.Patch(color=BAND_COLORS.get(n,"gray"), alpha=0.5, label=n) for n in BANDS.keys()]
        leg = self.ax_psd[0].legend(handles=handles, loc="upper right", framealpha=0.6, title="Bands")
        lh_list = getattr(leg, "legend_handles", None)
        if lh_list is None:
            lh_list = getattr(leg, "legendHandles", [])
        for lh in lh_list:
            try: lh.set_alpha(0.6)
            except Exception: pass


    def _apply_band_patches(self):
        vis = self.show_bands.current.value
        for ax, patches_ax in zip(self.ax_psd, self.band_patches):
            y0, y1 = ax.get_ylim()
            height = max(y1 - y0, 1e-12)
            for rect in patches_ax:
                rect.set_visible(vis)
                rect.set_y(y0)
                rect.set_height(height)


    def _redraw_psd_bands(self, e):
        self._apply_band_patches()
        self.psd_canvas.content = self._matplotlib_to_flet_image(self.fig_psd)
        self.page.update()


    def _draw_band_power(self):
        ax = self.ax_band_power
        ax.clear()
        for t in self.band_power_texts:
            try: t.remove()
            except Exception: pass
        self.band_power_texts = []
        if self.band_power_uv2_per_ch is None:
            ax.set_title("Band Power (µV²)")
            ax.set_ylabel("Power (µV²)")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            self.band_power_canvas.content = self._matplotlib_to_flet_image(self.fig_band_power)
            return

        # Prepare data for bar chart
        band_names = list(BANDS.keys())
        x = np.arange(len(band_names))
        global_max = 0.0

        # Trazar una línea por electrodo a lo largo de las bandas
        for ch_idx, ch_name in enumerate(CH_NAMES):
            y = np.array([self.band_power_uv2_per_ch[ch_idx].get(band, 0.0) for band in band_names], dtype=float)
            eps = 1e-6
            x_dense = np.linspace(x.min(), x.max(), 200)
            try:
                spline = make_interp_spline(x, np.log10(y + eps))
                y_smooth = spline(x_dense)
                ax.plot(x_dense, np.power(10, y_smooth),
                        color=CH_COLORS[ch_idx % len(CH_COLORS)], alpha=0.9, linewidth=1.4)
            except Exception:
                ax.plot(x, y, color=CH_COLORS[ch_idx % len(CH_COLORS)], alpha=0.9, linewidth=1.4)
            ax.plot(x, y, marker='o', linestyle='None', markersize=4,
                    color=CH_COLORS[ch_idx % len(CH_COLORS)], label=ch_name if ch_idx < len(CH_NAMES) else f"Ch{ch_idx+1}")
            # Marca la banda dominante de este canal
            if len(y) > 0:
                dom_idx = int(np.nanargmax(y))
                ax.scatter(x[dom_idx], y[dom_idx], s=60, marker='*',
                           color=CH_COLORS[ch_idx % len(CH_COLORS)], edgecolors='k', linewidths=0.6, zorder=6)

        ax.set_xticks(x)
        ax.set_xticklabels([b.capitalize() for b in band_names])
        ax.set_title("Band Power por canal (µV²)")
        ax.set_ylabel("Power (µV²)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=4, fontsize=8)
        self.band_power_canvas.content = self._matplotlib_to_flet_image(self.fig_band_power)

    def _apply_filters(self, data, fs):
        """Aplicar notch y bandpass opcionales."""
        if data is None or len(data) == 0:
            return data
        y = np.array(data, dtype=float)
        try:
            # Notch
            if self.notch_enable.current.value:
                try:
                    f0 = float(self.notch_freq.current.value or 50.0)
                except Exception:
                    f0 = 50.0
                w0 = f0 / (fs / 2)
                q = float(self.notch_q.current.value or 30.0)
                b, a = iirnotch(w0, Q=q)
                y = filtfilt(b, a, y, axis=0)
            # Bandpass
            if self.bandpass_enable.current.value:
                try:
                    low = float(self.bandpass_min.current.value or 1.0)
                    high = float(self.bandpass_max.current.value or 40.0)
                except Exception:
                    low, high = 1.0, 40.0
                if low < 0.1: low = 0.1
                if high > fs/2 - 1: high = fs/2 - 1
                if low >= high: low = max(high - 1.0, 0.1)
                sos = butter(4, [low, high], btype='band', fs=fs, output='sos')
                y = sosfiltfilt(sos, y, axis=0)
        except Exception:
            return data
        return y

    def _compute_quality(self, arr, fs):
        """Evalúa calidad básica por canal (NaN, flatline, saturación)."""
        if arr is None or len(arr) == 0:
            return []
        nwin = min(len(arr), int(fs * 2))
        seg = arr[-nwin:]
        status = []
        for i, ch in enumerate(CH_NAMES):
            y = seg[:, i] if i < seg.shape[1] else np.zeros(nwin)
            if not np.all(np.isfinite(y)) or len(y) == 0:
                status.append((ch, "bad", "nan"))
                continue
            std = np.std(y)
            maxabs = np.max(np.abs(y))
            if std < 1e-9:
                status.append((ch, "bad", "flat"))
            elif maxabs > 200e-6:  # >200 µV
                status.append((ch, "warn", "high"))
            else:
                status.append((ch, "ok", ""))
        return status

    def _render_quality(self, status):
        """Actualiza la fila de calidad como 'chips' compactos (una sola línea)."""
        self.quality_row.controls.clear()
        colors = {"ok": "#2ecc71", "warn": "#e67e22", "bad": "#e74c3c"}

        for ch, level, msg in status:
            bg = colors.get(level, "#888888")
            pill = ft.Container(
                content=ft.Text(ch, size=10, color="white", weight=ft.FontWeight.W_600),
                bgcolor=bg,
                padding=ft.padding.symmetric(horizontal=6, vertical=2),
                border_radius=8,
                tooltip=(msg if msg else None),
            )
            self.quality_row.controls.append(pill)

        # No llamar page.update() aquí; el loop principal ya hace el update.

    def _record_band_history(self, band_power_uv2_per_ch):
        # Guarda suma de bandas en el tiempo para un histórico (stackplot)
        ts = time.time()
        totals = {band: 0.0 for band in BANDS}
        for bp in band_power_uv2_per_ch:
            for band, val in bp.items():
                totals[band] += val
        self.band_history.append((ts, totals))

    def _record_row(self, band_power_uv2_per_ch):
        if not self.recording:
            return
        ts = time.time()
        if self.session_id <= 0:
            self.session_id = 1
        band_names = list(BANDS.keys())
        row = [self.session_id, ts]
        for i, ch in enumerate(CH_NAMES):
            bp = band_power_uv2_per_ch[i] if i < len(band_power_uv2_per_ch) else {}
            for band in band_names:
                row.append(float(bp.get(band, 0.0)))
        self.record_data.append(row)

    def _draw_record(self):
        ax = self.ax_record
        ax.clear()
        if not self.band_history:
            ax.set_title("Histórico de bandas (curvas)")
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Power (µV²)")
            ax.set_yscale("log")
            ax.grid(True, alpha=0.3)
            self.record_canvas.content = self._matplotlib_to_flet_image(self.fig_record)
            return

        times = [t for t, _ in self.band_history]
        t0 = times[0]
        times = [t - t0 for t in times]
        band_names = list(BANDS.keys())
        for band in band_names:
            vals = np.array([entry[1].get(band, 0.0) for entry in self.band_history], dtype=float)
            eps = 1e-6
            # Bandas de confianza con ventana móvil
            win = self.record_window_n
            lows, highs = [], []
            for i in range(len(vals)):
                seg = vals[max(0, i - win + 1): i + 1]
                lows.append(np.nanpercentile(seg, 25))
                highs.append(np.nanpercentile(seg, 75))
            if len(vals) >= 4:
                x_dense = np.linspace(times[0], times[-1], 300)
                try:
                    spline = make_interp_spline(times, np.log10(vals + eps))
                    y_smooth = spline(x_dense)
                    low_i = np.interp(x_dense, times, np.log10(np.array(lows) + eps))
                    high_i = np.interp(x_dense, times, np.log10(np.array(highs) + eps))
                    ax.fill_between(x_dense, np.power(10, low_i), np.power(10, high_i),
                                    color=BAND_COLORS.get(band, 'gray'), alpha=0.18, linewidth=0)
                    ax.plot(x_dense, np.power(10, y_smooth),
                            color=BAND_COLORS.get(band, 'gray'), linewidth=1.4, alpha=0.9,
                            label=band.capitalize())
                except Exception:
                    ax.plot(times, vals, color=BAND_COLORS.get(band, 'gray'),
                            linewidth=1.4, alpha=0.9, label=band.capitalize())
            else:
                ax.plot(times, vals, color=BAND_COLORS.get(band, 'gray'),
                        linewidth=1.4, alpha=0.9, label=band.capitalize())
            if len(vals) > 0:
                ax.scatter(times[-1], vals[-1], color=BAND_COLORS.get(band, 'gray'),
                           s=20, alpha=0.9)

        # Dibujar eventos como líneas verticales con etiquetas
        if self.events:
            for ev in self.events:
                ev_t = ev["ts"] - t0
                ax.axvline(ev_t, color="red", linestyle="--", linewidth=0.9, alpha=0.7)
                ax.text(ev_t, ax.get_ylim()[1], ev["label"], rotation=90,
                        va="bottom", ha="right", fontsize=8, color="red", alpha=0.8)

        ax.set_title("Histórico de bandas (curvas)")
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("Power (µV²)")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", ncol=2, fontsize=8)
        self.record_canvas.content = self._matplotlib_to_flet_image(self.fig_record)


    # ---------------- Tabla de métricas ----------------
    # ---------------- Draw: Topomap ----------------
    def _draw_topomap(self, force=False):
            if not self.running:
                return
            if not force:
                now = time.monotonic()
                if now - self.topomap_last_ts < 0.25:
                    return

            # Asegurar band powers
            if self.band_power_uv2_per_ch is None:
                self._draw_psd()
                if self.band_power_uv2_per_ch is None:
                    return

            band_name = self.topomap_band_dropdown.current.value
            if band_name not in BANDS:
                return

            # Recolectar valores por canal (µV²)
            ch_xy = []
            ch_data = []
            for i, ch_name in enumerate(self.ch_names):
                if ch_name in CH_LOCS_2D:
                    x, y = CH_LOCS_2D[ch_name]
                    ch_xy.append((x, y))
                    ch_data.append(self.band_power_uv2_per_ch[i].get(band_name, 0.0))

            if len(ch_xy) < 3:
                return

            points = np.array(ch_xy, dtype=float)
            values = np.array(ch_data, dtype=float)

            # Comprimir rango (log) para transiciones más suaves
            eps = 1e-6
            values_vis = np.log10(np.maximum(values, 0.0) + eps)

            r_x = self.topo_head_rx
            r_y = self.topo_head_ry
            gx, gy = self.topo_gx, self.topo_gy

                        # --- Interpolación estilo "orgánico" (tipo MNE): RBF suave (sin mezcla en el borde) ---
            # Nota: con pocos electrodos, cualquier método que "pegue" al borde (nearest/guard points)
            # puede generar deformaciones en extremos. Aquí preferimos un RBF suave (extrapolante)
            # y solo enmascaramos fuera de la cabeza.
            try:
                rbf = Rbf(
                    points[:, 0], points[:, 1], values_vis,
                    function="multiquadric", smooth=0.18, epsilon=0.35
                )
                grid_z = rbf(gx, gy)
            except Exception:
                # Fallback: intenta 'cubic' y luego 'linear'
                grid_z = griddata(points, values_vis, (gx, gy), method="cubic")
                if grid_z is None or np.all(np.isnan(grid_z)):
                    grid_z = griddata(points, values_vis, (gx, gy), method="linear")
                if grid_z is None:
                    return
            
            # Enmascarar fuera de la cabeza (aunque también hay clip_path)
            try:
                rnorm2 = np.sqrt((gx / r_x) ** 2 + (gy / r_y) ** 2)
                grid_z[rnorm2 > 1.0] = np.nan
            except Exception:
                pass
            try:
                # Si existe máscara precomputada, úsala también (equivalente a rnorm2>1)
                grid_z[self.topo_mask] = np.nan
            except Exception:
                pass
            

# Suavizado temporal (evita parpadeo)
            if self.topomap_grid_prev is not None:
                prev = self.topomap_grid_prev
                blended = np.where(np.isnan(grid_z), prev, grid_z)
                grid_z = 0.65 * prev + 0.35 * blended
            self.topomap_grid_prev = grid_z.copy()

            # Escala robusta basada en electrodos (no en la grilla)
            vmin, vmax = np.nanpercentile(values_vis, [5, 95])
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                vmin, vmax = np.nanmin(values_vis), np.nanmax(values_vis)
            if np.isclose(vmin, vmax):
                center = vmin if np.isfinite(vmin) else 0.0
                span = max(abs(center) * 0.25, 1e-3)
                vmin, vmax = center - span, center + span

            # Actualiza imagen y escala
            self.topomap_interp.set_data(grid_z)
            self.topomap_interp.set_extent((-r_x, r_x, -r_y, r_y))
            self.topomap_interp.set_clim(vmin=vmin, vmax=vmax)

            # Actualiza colorbar
            try:
                if self.topo_cbar is not None:
                    self.topo_cbar.update_normal(self.topomap_interp)
            except Exception:
                pass

            # Contornos ("líneas internas") — siempre blancas según requisito
            for cs in (self.topo_contours or []):
                try:
                    cs.remove()
                except Exception:
                    # compat
                    try:
                        for col in getattr(cs, "collections", []):
                            col.remove()
                    except Exception:
                        pass
            self.topo_contours = []
            try:
                levels = np.linspace(vmin, vmax, 9)
                z_masked = np.ma.masked_invalid(grid_z)

                # Limpia contornos previos para evitar que se encimen
                try:
                    if hasattr(self, "topo_contours") and self.topo_contours:
                        for old_cs in self.topo_contours:
                            try:
                                for col in getattr(old_cs, "collections", []):
                                    try:
                                        col.remove()
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                except Exception:
                    pass
                self.topo_contours = []
                self.topo_contour_collections = []

                cs = self.ax_topomap.contour(
                    gx, gy, z_masked, levels=levels,
                    colors="white", linewidths=1.1, alpha=0.90, linestyles="solid", zorder=5
                )
                # Clip de contornos al círculo para que no "salgan" fuera de la cabeza
                try:
                    for col in cs.collections:
                        col.set_clip_path(self.topo_clip_circle)
                except Exception:
                    pass
                self.topo_contours = [cs]
                try:
                    self.topo_contour_collections = list(cs.collections)
                except Exception:
                    self.topo_contour_collections = []
            except Exception:
                self.topo_contours = []

            # Actualiza sensores (electrodos)
            elec_mode = "Color"
            try:
                elec_mode = self.topomap_electrodes_dropdown.current.value or "Color"
            except Exception:
                elec_mode = "Color"

            if elec_mode == "Ocultar":
                self.topomap_scat.set_visible(False)
                self.topomap_scat_white.set_visible(False)
            elif elec_mode == "Blanco":
                self.topomap_scat.set_visible(False)
                self.topomap_scat_white.set_visible(True)
                self.topomap_scat_white.set_offsets(points)
            else:
                self.topomap_scat_white.set_visible(False)
                self.topomap_scat.set_visible(True)
                self.topomap_scat.set_offsets(points)
                self.topomap_scat.set_array(values_vis)

            # Ajusta estilo (contorno cabeza/nariz/orejas, contornos internos, colorbar) según tema
            self._set_topomap_outline_colors()

            self.topomap_last_ts = time.monotonic()
            self._refresh_topomap_canvas()

    def _draw_boxplots(self):
        ax = self.ax_boxplot
        ax.clear()
        if self.band_power_uv2_per_ch is None:
            self.boxplot_canvas.content = self._matplotlib_to_flet_image(self.fig_boxplot)
            return

        band_names = list(BANDS.keys())
        data = []
        for band in band_names:
            vals = [ch_data.get(band, 0.0) for ch_data in self.band_power_uv2_per_ch]
            vals = [v for v in vals if np.isfinite(v) and v > 0]
            if not vals:
                vals = [1e-6]
            data.append(np.log10(vals))

        bp = ax.boxplot(data, patch_artist=True, tick_labels=[b.capitalize() for b in band_names])
        for patch, band in zip(bp['boxes'], band_names):
            patch.set_facecolor(BAND_COLORS.get(band, 'gray'))
            patch.set_alpha(0.6)
        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.2)

        ax.set_title("Distribución por banda (log10 µV²)")
        ax.set_ylabel("log10 Power (µV²)")
        ax.grid(True, axis='y', alpha=0.3)
        self.boxplot_canvas.content = self._matplotlib_to_flet_image(self.fig_boxplot)

async def main(page: ft.Page):
    class Args: # Mocking argparse for Flet app
        def __init__(self):
            self.name = DEFAULT_STREAM_NAME
            self.fs = DEFAULT_FS
            self.update_hz = 10 # FPS de refresco de la UI

    args = Args()
    app = EEGViewerAppFlet(page, args)

if __name__ == "__main__":
    ft.app(target=main)
