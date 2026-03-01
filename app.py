"""
app.py - DuckCamASCII GUI Application
Full graphical interface with CustomTkinter.

Screens:
    1. Splash   — "Matrix rain" ASCII animation on startup
    2. Setup    — Choose local or remote camera + settings
    3. Viewer   — 3 panels: Original | YOLO | ASCII + sidebar
    4. Settings — Themes, charsets, density, export

Usage:
    python app.py
"""

import os
import sys
import time
import random
import threading
import datetime
import logging
import queue

import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

from ascii_engine import ASCIIEngine, CHARSETS, COLOR_THEMES, DENSITY_PRESETS
from camera_manager import CameraManager, CameraMode, CameraState, list_local_cameras

# ---------- Optional: YOLO ----------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

logging.basicConfig(level=logging.WARNING)

# ===================================================================
# Asynchronous YOLO Manager
# ===================================================================

class YOLOManager:
    """Runs YOLO prediction in an async background thread to avoid freezing the GUI."""
    
    def __init__(self, model_path="yolov8n-seg.pt"):
        self.model_path = model_path
        self._model = None
        self.is_loaded = False
        self.is_loading = False
        
        self._queue_in = queue.Queue(maxsize=1)
        self._queue_out = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread = None
        
        self._last_mask = None

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _worker(self):
        # 1. Load model
        self.is_loading = True
        try:
            self._model = YOLO(self.model_path)
            self.is_loaded = True
        except Exception as e:
            logging.warning("YOLO load error: %s", e)
            self.is_loaded = False
        finally:
            self.is_loading = False
            
        if not self.is_loaded:
            return

        # 2. Inference loop
        while not self._stop_event.is_set():
            try:
                frame_bgr = self._queue_in.get(timeout=0.1)
                
                # Run YOLO
                results = self._model.predict(frame_bgr, classes=[0], verbose=False, conf=0.45)
                
                mask_bin = None
                if results[0].masks is not None:
                    raw_mask = results[0].masks.data[0].cpu().numpy()
                    raw_mask = cv2.resize(raw_mask, (frame_bgr.shape[1], frame_bgr.shape[0]))
                    mask_bin = (raw_mask > 0.5).astype(np.uint8) * 255
                
                # Send result
                if self._queue_out.full():
                    try: self._queue_out.get_nowait()
                    except: pass
                self._queue_out.put_nowait(mask_bin)
                
            except queue.Empty:
                continue
            except Exception as e:
                logging.warning("YOLO pred error: %s", e)
                
    def push_frame(self, frame_bgr):
        """Sends frame for analysis if there's space (non-blocking)."""
        if not self.is_loaded: return
        if self._queue_in.full():
            try: self._queue_in.get_nowait()
            except: pass
        try:
            self._queue_in.put_nowait(frame_bgr)
        except queue.Full:
            pass
            
    def get_latest_mask(self):
        """Gets the latest processed mask. If there's no new one, returns the previous."""
        try:
            self._last_mask = self._queue_out.get_nowait()
        except queue.Empty:
            pass
        return self._last_mask


# ===================================================================
# Palette & Visual Theme
# ===================================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

BG_DARK    = "#0a0f0a"
BG_PANEL   = "#0d150d"
BG_CARD    = "#111c11"
BG_INPUT   = "#0e170e"
ACCENT     = "#00ff41"        # Matrix Green
ACCENT2    = "#00cc33"
BORDER     = "#1a3a1a"
TEXT_DIM   = "#4a7a4a"
TEXT_MAIN  = "#c8e6c8"
TEXT_TITLE = "#00ff41"
RED        = "#ff4444"
YELLOW     = "#ffcc00"

FONT_TITLE  = ("Courier New", 28, "bold")
FONT_MONO   = ("Courier New", 11)
FONT_SMALL  = ("Courier New", 10)
FONT_LABEL  = ("Courier New", 12)
FONT_BTN    = ("Courier New", 12, "bold")
FONT_STATUS = ("Courier New", 10)

# ===================================================================
# Frame utilities → CTkImage
# ===================================================================

def bgr_to_ctk(frame_bgr: np.ndarray, max_w: int, max_h: int) -> ctk.CTkImage:
    h, w = frame_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    return ctk.CTkImage(light_image=pil, dark_image=pil, size=(new_w, new_h))


# ===================================================================
# Splash Screen
# ===================================================================
DUCK_ASCII = r"""
    ___
   (o o)
  (  V  )--~  DuckCamASCII
   --m-m----
"""

RAIN_CHARS = "アイウエオカキクケコ01@#$%ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class SplashScreen(ctk.CTkToplevel):
    """Startup screen with Matrix rain and ASCII duck."""

    WIDTH, HEIGHT = 640, 420
    COLS   = 64
    SPEED  = 80   # ms per tick

    def __init__(self, parent, on_done):
        super().__init__(parent)
        self.on_done = on_done
        self.title("DuckCamASCII")
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}")
        self.resizable(False, False)
        self.configure(fg_color=BG_DARK)
        self.overrideredirect(True)  # No title bar
        self._center()
        self._build()
        self._running = True
        self._drops = [random.randint(0, 30) for _ in range(self.COLS)]
        self._after_id = None
        self._tick()
        # Closes automatically after 2.8s
        self.after(2800, self._close)

    def _center(self):
        self.update_idletasks()
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        x = (sw - self.WIDTH) // 2
        y = (sh - self.HEIGHT) // 2
        self.geometry(f"{self.WIDTH}x{self.HEIGHT}+{x}+{y}")

    def _build(self):
        self.canvas = ctk.CTkCanvas(
            self, width=self.WIDTH, height=self.HEIGHT,
            bg=BG_DARK, highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        # Center duck and title
        self.canvas.create_text(
            self.WIDTH // 2, self.HEIGHT // 2 - 30,
            text=DUCK_ASCII, fill=ACCENT, font=("Courier New", 13, "bold"),
            justify="center", tag="duck",
        )
        self.canvas.create_text(
            self.WIDTH // 2, self.HEIGHT // 2 + 80,
            text="Loading...", fill=TEXT_DIM, font=("Courier New", 10),
            tag="status",
        )

    def _tick(self):
        if not self._running:
            return
        c = self.canvas
        # Semi-transparent background for trail
        c.create_rectangle(0, 0, self.WIDTH, self.HEIGHT,
                            fill=BG_DARK, stipple="gray50", outline="")
        col_w = self.WIDTH // self.COLS
        for i, y in enumerate(self._drops):
            char = random.choice(RAIN_CHARS)
            x = i * col_w + col_w // 2
            cy = y * 14
            # Lighter character at the top of the drop
            c.create_text(x, cy, text=char, fill=ACCENT, font=("Courier New", 10), tag="rain")
            # Faded trail
            if cy > 20:
                c.create_text(x, cy - 14, text=random.choice(RAIN_CHARS),
                               fill=ACCENT2, font=("Courier New", 9), tag="rain")
            self._drops[i] = (y + 1) % (self.HEIGHT // 14 + 5)
            if random.random() < 0.02:
                self._drops[i] = 0

        # Bring center text over the rain
        c.tag_raise("duck")
        c.tag_raise("status")
        self._after_id = self.after(self.SPEED, self._tick)

    def _close(self):
        self._running = False
        if self._after_id:
            self.after_cancel(self._after_id)
        self.destroy()
        self.on_done()


# ===================================================================
# Setup Screen
# ===================================================================

class SetupScreen(ctk.CTkFrame):
    """Camera configuration screen."""

    def __init__(self, parent, on_connect, **kwargs):
        super().__init__(parent, fg_color=BG_DARK, **kwargs)
        self.on_connect = on_connect
        self._local_cams = []
        self._build()
        self._detect_cameras()

    # ------------------------------------------------------------------
    def _build(self):
        # --- Header ---
        header = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0, height=70)
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(
            header, text="◈  DuckCamASCII", font=FONT_TITLE, text_color=TEXT_TITLE
        ).pack(side="left", padx=24)
        ctk.CTkLabel(
            header, text="by MozzieGM", font=FONT_SMALL, text_color=TEXT_DIM
        ).pack(side="right", padx=24)

        # --- Body ---
        body = ctk.CTkFrame(self, fg_color=BG_DARK)
        body.pack(fill="both", expand=True, padx=40, pady=30)

        # --- Mode Card ---
        mode_card = self._card(body, "CAMERA SOURCE")
        mode_card.pack(fill="x", pady=(0, 20))

        self._mode_var = ctk.StringVar(value="local")
        row_mode = ctk.CTkFrame(mode_card, fg_color="transparent")
        row_mode.pack(fill="x", padx=16, pady=10)

        for text, val in [("📷  Local Camera (this PC)", "local"),
                           ("🌐  Remote Camera (another PC on network)", "remote")]:
            ctk.CTkRadioButton(
                row_mode, text=text, variable=self._mode_var, value=val,
                command=self._on_mode_change,
                font=FONT_LABEL, text_color=TEXT_MAIN,
                fg_color=ACCENT, hover_color=ACCENT2,
                border_color=BORDER,
            ).pack(anchor="w", pady=4)

        # --- Local Camera Card ---
        self._local_card = self._card(body, "LOCAL CAMERA")
        self._local_card.pack(fill="x", pady=(0, 10))
        inner_local = ctk.CTkFrame(self._local_card, fg_color="transparent")
        inner_local.pack(fill="x", padx=16, pady=10)

        ctk.CTkLabel(inner_local, text="Camera:", font=FONT_LABEL, text_color=TEXT_DIM).pack(side="left")
        self._cam_combo = ctk.CTkComboBox(
            inner_local, values=["Detecting..."], width=260,
            font=FONT_LABEL, fg_color=BG_INPUT, border_color=BORDER,
            button_color=ACCENT2, text_color=TEXT_MAIN,
        )
        self._cam_combo.pack(side="left", padx=10)
        ctk.CTkButton(
            inner_local, text="↻ Refresh", width=90, height=30,
            font=FONT_SMALL, fg_color=BG_CARD, hover_color=BORDER,
            text_color=ACCENT, border_color=BORDER, border_width=1,
            command=self._detect_cameras,
        ).pack(side="left", padx=4)

        # --- Remote Camera Card ---
        self._remote_card = self._card(body, "REMOTE CAMERA (TCP/IP)")
        # Starts hidden
        inner_remote = ctk.CTkFrame(self._remote_card, fg_color="transparent")
        inner_remote.pack(fill="x", padx=16, pady=10)

        ctk.CTkLabel(inner_remote, text="Server IP:", font=FONT_LABEL, text_color=TEXT_DIM).pack(side="left")
        self._ip_entry = ctk.CTkEntry(
            inner_remote, placeholder_text="e.g. 192.168.0.107",
            width=180, font=FONT_MONO, fg_color=BG_INPUT, border_color=BORDER,
            text_color=TEXT_MAIN,
        )
        self._ip_entry.pack(side="left", padx=10)
        ctk.CTkLabel(inner_remote, text="Port:", font=FONT_LABEL, text_color=TEXT_DIM).pack(side="left")
        self._port_entry = ctk.CTkEntry(
            inner_remote, placeholder_text="9999",
            width=70, font=FONT_MONO, fg_color=BG_INPUT, border_color=BORDER,
            text_color=TEXT_MAIN,
        )
        self._port_entry.insert(0, "9999")
        self._port_entry.pack(side="left", padx=6)

        # --- Remote Hint ---
        hint = ctk.CTkLabel(
            self._remote_card,
            text="  ℹ  Run  python server.py  on the laptop/PC with the camera.",
            font=FONT_SMALL, text_color=TEXT_DIM,
        )
        hint.pack(anchor="w", padx=16, pady=(0, 10))

        # --- YOLO Card ---
        yolo_card = self._card(body, "PERSON DETECTION (YOLO)")
        yolo_card.pack(fill="x", pady=(0, 20))
        yolo_inner = ctk.CTkFrame(yolo_card, fg_color="transparent")
        yolo_inner.pack(fill="x", padx=16, pady=10)

        self._yolo_var = ctk.BooleanVar(value=YOLO_AVAILABLE)
        ctk.CTkSwitch(
            yolo_inner, text="  Enable YOLO (segment person only)",
            variable=self._yolo_var, font=FONT_LABEL, text_color=TEXT_MAIN,
            fg_color=TEXT_DIM, progress_color=ACCENT, button_color=TEXT_MAIN,
        ).pack(side="left")

        if not YOLO_AVAILABLE:
            ctk.CTkLabel(
                yolo_inner, text="  ✗ ultralytics not installed",
                font=FONT_SMALL, text_color=RED,
            ).pack(side="left", padx=8)

        # --- Connect Button ---
        self._status_lbl = ctk.CTkLabel(
            body, text="", font=FONT_SMALL, text_color=TEXT_DIM
        )
        self._status_lbl.pack(pady=(0, 6))

        ctk.CTkButton(
            body, text="▶  CONNECT", font=FONT_BTN,
            fg_color=ACCENT2, hover_color=ACCENT, text_color="#000000",
            height=48, corner_radius=6,
            command=self._connect,
        ).pack(fill="x", pady=(0, 4))

        self._on_mode_change()

    # ------------------------------------------------------------------
    def _card(self, parent, title: str) -> ctk.CTkFrame:
        card = ctk.CTkFrame(parent, fg_color=BG_CARD, corner_radius=8,
                             border_color=BORDER, border_width=1)
        ctk.CTkLabel(
            card, text=f"  {title}", font=FONT_SMALL, text_color=TEXT_DIM,
            anchor="w",
        ).pack(fill="x", padx=8, pady=(8, 0))
        return card

    def _on_mode_change(self):
        mode = self._mode_var.get()
        if mode == "local":
            self._local_card.pack(fill="x", pady=(0, 10))
            self._remote_card.pack_forget()
        else:
            self._local_card.pack_forget()
            self._remote_card.pack(fill="x", pady=(0, 10))

    def _detect_cameras(self):
        self._cam_combo.configure(values=["Detecting..."])
        self._cam_combo.set("Detecting...")

        def _do():
            cams = list_local_cameras()
            self._local_cams = cams
            opts = [f"Camera {c['index']} — {c['name']}" for c in cams] or ["No cameras found"]
            self.after(0, lambda: self._cam_combo.configure(values=opts))
            self.after(0, lambda: self._cam_combo.set(opts[0]))

        threading.Thread(target=_do, daemon=True).start()

    def _connect(self):
        mode = self._mode_var.get()
        use_yolo = self._yolo_var.get() and YOLO_AVAILABLE
        self._status_lbl.configure(text="Connecting...", text_color=YELLOW)

        if mode == "local":
            idx_text = self._cam_combo.get()
            try:
                cam_index = int(idx_text.split()[1])
            except (IndexError, ValueError):
                cam_index = 0
            self.on_connect(mode="local", cam_index=cam_index, use_yolo=use_yolo)
        else:
            ip = self._ip_entry.get().strip()
            if not ip:
                self._status_lbl.configure(text="⚠ Enter the server IP.", text_color=RED)
                return
            try:
                port = int(self._port_entry.get().strip())
            except ValueError:
                port = 9999
            self.on_connect(mode="remote", host=ip, port=port, use_yolo=use_yolo)


# ===================================================================
# Viewer Screen — 3 panels + sidebar
# ===================================================================

class ViewerScreen(ctk.CTkFrame):
    """
    Main visualizer screen.
    Panels: Original | YOLO Mask | ASCII Output
    Sidebar: ASCII controls, YOLO toggle, themes, FPS, etc.
    """

    POLL_MS = 33  # ~30fps UI update rate

    def __init__(self, parent, camera_manager: CameraManager,
                 use_yolo: bool, on_disconnect, on_settings, **kwargs):
        super().__init__(parent, fg_color=BG_DARK, **kwargs)
        self._cam = camera_manager
        self._use_yolo = use_yolo
        self._yolo_on = use_yolo
        self._on_disconnect = on_disconnect
        self._on_settings = on_settings

        self._engine = ASCIIEngine()
        self._fps = 0.0
        self._frame_count = 0
        self._last_frame_time = time.monotonic()

        self._yolo_manager = YOLOManager()
        if use_yolo and YOLO_AVAILABLE:
            self._yolo_manager.start()

        # Processed frames queue for display
        self._display_queue: queue.Queue = queue.Queue(maxsize=2)

        self._build()
        self._start_poll()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self):
        # ---- TOPBAR ----
        topbar = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0, height=48)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        ctk.CTkLabel(
            topbar, text="◈ DuckCamASCII", font=("Courier New", 14, "bold"),
            text_color=TEXT_TITLE,
        ).pack(side="left", padx=16)

        ctk.CTkButton(
            topbar, text="⚙ Settings", width=100, height=30,
            font=FONT_SMALL, fg_color=BG_CARD, hover_color=BORDER,
            text_color=ACCENT, border_color=BORDER, border_width=1,
            command=self._on_settings,
        ).pack(side="right", padx=8)

        self._disconnect_btn = ctk.CTkButton(
            topbar, text="✕ Disconnect", width=120, height=30,
            font=FONT_SMALL, fg_color=BG_CARD, hover_color="#3a0000",
            text_color=RED, border_color=RED, border_width=1,
            command=self._disconnect,
        )
        self._disconnect_btn.pack(side="right", padx=4)

        # ---- MAIN AREA (panels + sidebar) ----
        main = ctk.CTkFrame(self, fg_color=BG_DARK)
        main.pack(fill="both", expand=True)

        # --- SIDEBAR ---
        sidebar = ctk.CTkScrollableFrame(
            main, fg_color=BG_PANEL, corner_radius=0, width=220,
            scrollbar_button_color=BORDER, scrollbar_button_hover_color=ACCENT2,
        )
        sidebar.pack(side="right", fill="y")
        self._build_sidebar(sidebar)

        # --- PANELS ---
        panel_area = ctk.CTkFrame(main, fg_color=BG_DARK)
        panel_area.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        # 3 inline panels
        self._panels: dict[str, ctk.CTkLabel] = {}
        titles = ["1 · Original", "2 · YOLO + Mask", "3 · ASCII Output"]
        keys   = ["original", "yolo", "ascii"]

        for title, key in zip(titles, keys):
            col = ctk.CTkFrame(panel_area, fg_color=BG_CARD, corner_radius=8,
                                border_color=BORDER, border_width=1)
            col.pack(side="left", fill="both", expand=True, padx=4)
            ctk.CTkLabel(
                col, text=title, font=FONT_SMALL, text_color=TEXT_DIM,
            ).pack(pady=(6, 2))
            lbl = ctk.CTkLabel(col, text="", image=None)
            lbl.pack(fill="both", expand=True, padx=4, pady=(0, 4))
            self._panels[key] = lbl

        # ---- STATUSBAR ----
        self._statusbar = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0, height=26)
        self._statusbar.pack(fill="x", side="bottom")
        self._statusbar.pack_propagate(False)
        self._status_lbl = ctk.CTkLabel(
            self._statusbar, text="Waiting for camera...",
            font=FONT_STATUS, text_color=TEXT_DIM, anchor="w",
        )
        self._status_lbl.pack(side="left", padx=12)
        self._fps_lbl = ctk.CTkLabel(
            self._statusbar, text="FPS: --",
            font=FONT_STATUS, text_color=ACCENT, anchor="e",
        )
        self._fps_lbl.pack(side="right", padx=12)

    def _build_sidebar(self, parent):
        def section(text):
            ctk.CTkLabel(
                parent, text=text, font=FONT_SMALL, text_color=TEXT_DIM, anchor="w",
            ).pack(fill="x", padx=10, pady=(14, 2))
            ctk.CTkFrame(parent, fg_color=BORDER, height=1).pack(fill="x", padx=6)

        # ----- YOLO -----
        section("DETECTION (YOLO)")
        self._yolo_switch_var = ctk.BooleanVar(value=self._yolo_on)
        ctk.CTkSwitch(
            parent, text="  YOLO ON/OFF",
            variable=self._yolo_switch_var, command=self._toggle_yolo,
            font=FONT_SMALL, text_color=TEXT_MAIN,
            fg_color=TEXT_DIM, progress_color=ACCENT, button_color=TEXT_MAIN,
        ).pack(anchor="w", padx=14, pady=6)

        # ----- THEME -----
        section("COLOR THEME")
        self._theme_var = ctk.StringVar(value="Matrix")
        for name in COLOR_THEMES:
            ctk.CTkRadioButton(
                parent, text=name, variable=self._theme_var, value=name,
                command=self._apply_theme,
                font=FONT_SMALL, text_color=TEXT_MAIN,
                fg_color=ACCENT, hover_color=ACCENT2, border_color=BORDER,
            ).pack(anchor="w", padx=14, pady=2)

        # ----- CHARSET -----
        section("CHARSET")
        self._charset_var = ctk.StringVar(value="Standard")
        ctk.CTkComboBox(
            parent, values=list(CHARSETS.keys()), variable=self._charset_var,
            command=self._apply_charset,
            font=FONT_SMALL, fg_color=BG_INPUT, border_color=BORDER,
            button_color=ACCENT2, text_color=TEXT_MAIN, width=180,
        ).pack(padx=10, pady=6)

        # ----- DENSITY -----
        section("DENSITY")
        self._density_var = ctk.StringVar(value="Medium")
        for name in DENSITY_PRESETS:
            ctk.CTkRadioButton(
                parent, text=name, variable=self._density_var, value=name,
                command=self._apply_density,
                font=FONT_SMALL, text_color=TEXT_MAIN,
                fg_color=ACCENT, hover_color=ACCENT2, border_color=BORDER,
            ).pack(anchor="w", padx=14, pady=2)

        # ----- ACTIONS -----
        section("ACTIONS")
        ctk.CTkButton(
            parent, text="📸  Save Frame", font=FONT_SMALL,
            fg_color=BG_CARD, hover_color=BORDER,
            text_color=ACCENT, border_color=BORDER, border_width=1,
            width=180, height=32, command=self._save_frame,
        ).pack(padx=10, pady=4)

        ctk.CTkButton(
            parent, text="📋  Copy ASCII (text)", font=FONT_SMALL,
            fg_color=BG_CARD, hover_color=BORDER,
            text_color=ACCENT, border_color=BORDER, border_width=1,
            width=180, height=32, command=self._copy_ascii_text,
        ).pack(padx=10, pady=4)

        self._save_label = ctk.CTkLabel(
            parent, text="", font=FONT_SMALL, text_color=TEXT_DIM, wraplength=180,
        )
        self._save_label.pack(padx=10, pady=(0, 8))

    # ------------------------------------------------------------------
    # Polling / Frame Update
    # ------------------------------------------------------------------

    def _start_poll(self):
        self._poll()

    def _poll(self):
        frame = self._cam.get_frame()
        if frame is not None:
            self._process_and_display(frame)

        # Update camera status
        state = self._cam.state
        state_text = {
            CameraState.STREAMING:   "● Streaming",
            CameraState.CONNECTING:  "◌ Connecting...",
            CameraState.ERROR:       "✗ Connection Error",
            CameraState.IDLE:        "◎ Idle",
            CameraState.STOPPED:     "■ Stopped",
        }.get(state, str(state))
        self._status_lbl.configure(text=f"  {state_text}")

        self.after(self.POLL_MS, self._poll)

    def _process_and_display(self, frame_bgr: np.ndarray):
        # Processing FPS counter
        now = time.monotonic()
        self._frame_count += 1
        if now - self._last_frame_time >= 1.0:
            self._fps = self._frame_count / (now - self._last_frame_time)
            self._frame_count = 0
            self._last_frame_time = now
            self._fps_lbl.configure(text=f"FPS: {self._fps:.1f}")

        # Size of each panel (~1/3 of the available area)
        pw = max(200, (self.winfo_width() - 240) // 3)
        ph = max(150, self.winfo_height() - 100)

        # --- Panel 1: Original ---
        img_orig = bgr_to_ctk(frame_bgr, pw, ph)
        self._panels["original"].configure(image=img_orig)
        self._panels["original"].image = img_orig

        # --- YOLO segmentation ---
        mask = None
        frame_person = frame_bgr.copy()

        if self._yolo_on and self._yolo_manager.is_loaded:
            # Send current frame to YOLO thread to run in background (non-blocking)
            self._yolo_manager.push_frame(frame_bgr.copy())
            
            # Get the latest ready mask (calculated from a recent frame)
            mask_bin = self._yolo_manager.get_latest_mask()
            
            if mask_bin is not None:
                # Handle possible resolution size differences if the camera changed
                if mask_bin.shape[:2] != frame_bgr.shape[:2]:
                    mask_bin = cv2.resize(mask_bin, (frame_bgr.shape[1], frame_bgr.shape[0]))
                
                mask = mask_bin
                frame_person = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask_bin)

                # Visual overlay: dark blue background + natural person
                yolo_viz = frame_bgr.copy()
                bg_overlay = np.full_like(frame_bgr, (20, 0, 40), dtype=np.uint8)
                yolo_viz = np.where(
                    np.stack([mask_bin, mask_bin, mask_bin], axis=2) > 0,
                    frame_bgr, bg_overlay,
                )
                # Green contour for the mask
                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(yolo_viz, contours, -1, (0, 255, 80), 2)
            else:
                frame_person = np.zeros_like(frame_bgr)
                yolo_viz = frame_bgr.copy()
                # Processing status for the first mask...
                cv2.putText(yolo_viz, "YOLO Background Processing...",
                            (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 200, 255), 1)
        else:
            yolo_viz = frame_bgr.copy()
            if self._yolo_on and self._yolo_manager.is_loading:
                cv2.putText(yolo_viz, "Loading YOLO...",
                            (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 200, 255), 1)

        # --- Panel 2: YOLO viz ---
        img_yolo = bgr_to_ctk(yolo_viz, pw, ph)
        self._panels["yolo"].configure(image=img_yolo)
        self._panels["yolo"].image = img_yolo

        # --- Panel 3: ASCII ---
        ascii_frame = self._engine.render(frame_person, mask=mask)
        img_ascii = bgr_to_ctk(ascii_frame, pw, ph)
        self._panels["ascii"].configure(image=img_ascii)
        self._panels["ascii"].image = img_ascii

        # Stores current frame for actions
        self._last_frame = frame_bgr
        self._last_ascii = ascii_frame

    # ------------------------------------------------------------------
    # Sidebar actions
    # ------------------------------------------------------------------

    def _toggle_yolo(self):
        self._yolo_on = self._yolo_switch_var.get()
        if self._yolo_on and not self._yolo_manager.is_loaded and YOLO_AVAILABLE:
            if not self._yolo_manager.is_loading:
                self._yolo_manager.start()

    def _apply_theme(self):
        self._engine.set_theme(self._theme_var.get())

    def _apply_charset(self, _=None):
        self._engine.set_charset(self._charset_var.get())

    def _apply_density(self):
        self._engine.set_density(self._density_var.get())

    def _save_frame(self):
        if not hasattr(self, "_last_ascii"):
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(os.getcwd(), f"duckcam_ascii_{ts}.png")
        cv2.imwrite(path, self._last_ascii)
        self._save_label.configure(text=f"✓ Saved:\n{os.path.basename(path)}", text_color=ACCENT)
        self.after(4000, lambda: self._save_label.configure(text=""))

    def _copy_ascii_text(self):
        if not hasattr(self, "_last_frame"):
            return
        text = self._engine.get_text_output(self._last_frame)
        self.clipboard_clear()
        self.clipboard_append(text)
        self._save_label.configure(text="✓ ASCII copied to\nthe clipboard!", text_color=ACCENT)
        self.after(3000, lambda: self._save_label.configure(text=""))

    def _disconnect(self):
        self._cam.stop()
        self._yolo_manager.stop()
        self._on_disconnect()
    
    def get_last_frames(self):
        """Returns (frame_original, frame_ascii) for Settings."""
        orig = getattr(self, "_last_frame", None)
        asc  = getattr(self, "_last_ascii", None)
        return orig, asc


# ===================================================================
# Settings Screen
# ===================================================================

class SettingsScreen(ctk.CTkFrame):
    """Advanced Settings Screen."""

    def __init__(self, parent, engine: ASCIIEngine, on_back, **kwargs):
        super().__init__(parent, fg_color=BG_DARK, **kwargs)
        self._engine = engine
        self._on_back = on_back
        self._build()

    def _build(self):
        # Header
        hdr = ctk.CTkFrame(self, fg_color=BG_PANEL, corner_radius=0, height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        ctk.CTkButton(
            hdr, text="◀ Back", width=90, height=32,
            font=FONT_SMALL, fg_color="transparent", hover_color=BG_CARD,
            text_color=ACCENT, command=self._on_back,
        ).pack(side="left", padx=12)
        ctk.CTkLabel(
            hdr, text="⚙  Advanced Settings",
            font=("Courier New", 14, "bold"), text_color=TEXT_TITLE,
        ).pack(side="left", padx=8)

        body = ctk.CTkScrollableFrame(self, fg_color=BG_DARK)
        body.pack(fill="both", expand=True, padx=40, pady=20)

        # --- Font scale ---
        self._card_section(body, "ASCII FONT SCALE")
        self._font_scale_var = ctk.DoubleVar(value=self._engine.font_scale)
        ctk.CTkSlider(
            body, from_=0.4, to=1.5, number_of_steps=22,
            variable=self._font_scale_var, command=self._update_font_scale,
            button_color=ACCENT, progress_color=ACCENT2, fg_color=BORDER,
        ).pack(fill="x", padx=20, pady=4)
        self._font_lbl = ctk.CTkLabel(body, text=f"Scale: {self._engine.font_scale:.2f}",
                                       font=FONT_SMALL, text_color=TEXT_DIM)
        self._font_lbl.pack()

        # --- Thickness ---
        self._card_section(body, "STROKE THICKNESS")
        self._thick_var = ctk.IntVar(value=self._engine.font_thickness)
        for v in [1, 2, 3]:
            ctk.CTkRadioButton(
                body, text=f"{'─' * v}  Thickness {v}", value=v,
                variable=self._thick_var, command=self._update_thickness,
                font=FONT_SMALL, text_color=TEXT_MAIN,
                fg_color=ACCENT, hover_color=ACCENT2, border_color=BORDER,
            ).pack(anchor="w", padx=24, pady=2)

        # --- Preview colors ---
        self._card_section(body, "COLOR PREVIEW")
        preview_grid = ctk.CTkFrame(body, fg_color="transparent")
        preview_grid.pack(fill="x", padx=20, pady=8)
        for i, (name, theme) in enumerate(COLOR_THEMES.items()):
            color = theme["color"] if theme["mode"] == "solid" else (0, 200, 200)
            r, g, b = color[2], color[1], color[0]
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            f = ctk.CTkFrame(
                preview_grid, fg_color=BG_CARD, corner_radius=4,
                border_color=BORDER, border_width=1, width=120, height=40,
            )
            f.grid(row=i // 4, column=i % 4, padx=4, pady=4)
            f.pack_propagate(False)
            ctk.CTkLabel(
                f, text=f"● {name}", font=FONT_SMALL, text_color=hex_color,
            ).pack(expand=True)

        # --- Info ---
        self._card_section(body, "ABOUT")
        about_text = (
            "DuckCamASCII v2.0\n"
            "Created by MozzieGM\n\n"
            "Technologies:\n"
            "  OpenCV · YOLO v8 · CustomTkinter\n"
            "  NumPy · Pillow · Python 3.x\n\n"
            "Local mode: direct camera on this PC\n"
            "Remote mode: server.py on another PC"
        )
        ctk.CTkLabel(
            body, text=about_text, font=FONT_MONO, text_color=TEXT_DIM,
            justify="left",
        ).pack(anchor="w", padx=24, pady=8)

    def _card_section(self, parent, title: str):
        ctk.CTkLabel(
            parent, text=f"\n  {title}", font=FONT_SMALL, text_color=TEXT_DIM, anchor="w",
        ).pack(fill="x", padx=10)
        ctk.CTkFrame(parent, fg_color=BORDER, height=1).pack(fill="x", padx=10, pady=(0, 6))

    def _update_font_scale(self, val):
        self._engine.font_scale = float(val)
        self._font_lbl.configure(text=f"Scale: {float(val):.2f}")

    def _update_thickness(self):
        self._engine.font_thickness = self._thick_var.get()


# ===================================================================
# Main App
# ===================================================================

class DuckCamApp(ctk.CTk):
    """Root Window — manages screen navigation."""

    def __init__(self):
        super().__init__()
        self.title("DuckCamASCII")
        self.geometry("1200x720")
        self.minsize(960, 600)
        self.configure(fg_color=BG_DARK)
        self._set_icon()

        self._cam = CameraManager()
        self._cam.on_state_change = self._on_cam_state
        self._cam.on_fps_update   = self._on_fps_update
        self._cam.on_error        = self._on_cam_error

        self._current_screen = None
        self._viewer: ViewerScreen | None = None

        # Start with splash
        self.withdraw()
        self.after(200, self._show_splash)

    def _set_icon(self):
        try:
            img = Image.new("RGB", (64, 64), color=(10, 20, 10))
            photo = ImageTk.PhotoImage(img)
            self.wm_iconphoto(True, photo)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _show_splash(self):
        splash = SplashScreen(self, on_done=self._show_setup)
        splash.grab_set()

    def _show_setup(self):
        self.deiconify()
        self._switch_screen(
            SetupScreen(self, on_connect=self._start_camera)
        )

    def _start_camera(self, mode: str, use_yolo: bool, **kwargs):
        if mode == "local":
            self._cam.configure_local(kwargs.get("cam_index", 0))
        else:
            self._cam.configure_remote(kwargs.get("host", ""), kwargs.get("port", 9999))

        self._cam.start()

        viewer = ViewerScreen(
            self,
            camera_manager=self._cam,
            use_yolo=use_yolo,
            on_disconnect=self._disconnect,
            on_settings=self._show_settings,
        )
        self._viewer = viewer
        self._switch_screen(viewer)

    def _disconnect(self):
        self._cam.stop()
        self._viewer = None
        self._show_setup()

    def _show_settings(self):
        if self._viewer is None:
            return
        settings = SettingsScreen(
            self,
            engine=self._viewer._engine,
            on_back=lambda: self._switch_screen(self._viewer),
        )
        self._switch_screen(settings)

    def _switch_screen(self, new_screen: ctk.CTkFrame):
        if self._current_screen:
            self._current_screen.pack_forget()
        self._current_screen = new_screen
        new_screen.pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    # Camera callbacks (threaded → enqueued to main thread)
    # ------------------------------------------------------------------

    def _on_cam_state(self, state: CameraState, message: str):
        self.after(0, lambda: self._handle_state(state, message))

    def _handle_state(self, state: CameraState, message: str):
        if state == CameraState.ERROR and self._viewer:
            pass  # ViewerScreen already displays the status

    def _on_fps_update(self, fps: float):
        pass  # ViewerScreen calculates its own display FPS

    def _on_cam_error(self, msg: str):
        self.after(0, lambda: self._show_error(msg))

    def _show_error(self, msg: str):
        # Only show dialog if not in viewer
        if self._viewer is None:
            dialog = ctk.CTkToplevel(self)
            dialog.title("Camera Error")
            dialog.geometry("440x160")
            dialog.configure(fg_color=BG_DARK)
            dialog.grab_set()
            ctk.CTkLabel(
                dialog,
                text=f"✗  {msg}",
                font=FONT_LABEL, text_color=RED, wraplength=380,
            ).pack(pady=24)
            ctk.CTkButton(
                dialog, text="OK", font=FONT_BTN, width=100,
                fg_color=BG_CARD, hover_color=BORDER,
                text_color=ACCENT, border_color=BORDER, border_width=1,
                command=dialog.destroy,
            ).pack()

    def on_closing(self):
        self._cam.stop()
        self.destroy()


# ===================================================================
# Entry point
# ===================================================================

if __name__ == "__main__":
    app = DuckCamApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
