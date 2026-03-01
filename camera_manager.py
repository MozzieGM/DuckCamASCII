"""
camera_manager.py - Camera manager for DuckCamASCII
Abstracts local cameras (OpenCV) and remote cameras (TCP socket) into a single object.
"""

import cv2
import socket
import struct
import pickle
import threading
import queue
import time
import logging
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class CameraMode(Enum):
    LOCAL = "local"
    REMOTE = "remote"


class CameraState(Enum):
    IDLE = "idle"
    CONNECTING = "connecting"
    STREAMING = "streaming"
    ERROR = "error"
    STOPPED = "stopped"


def list_local_cameras(max_index: int = 6) -> list[dict]:
    """Detects available local cameras. Returns a list of dicts with index and name."""
    cameras = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cameras.append({"index": i, "name": f"Camera {i}"})
            cap.release()
    return cameras


class CameraManager:
    """
    Manages capturing frames from a local or remote camera in a separate thread.

    Callbacks:
        on_state_change(state: CameraState, message: str)
        on_fps_update(fps: float)
        on_error(msg: str)
    """

    # Maximum queue size (discards oldest frames if full)
    QUEUE_SIZE = 3

    def __init__(self):
        self._mode = CameraMode.LOCAL
        self._local_index = 0
        self._remote_host = "192.168.0.107"
        self._remote_port = 9999

        self._frame_queue: queue.Queue = queue.Queue(maxsize=self.QUEUE_SIZE)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state = CameraState.IDLE

        # Callbacks
        self.on_state_change: Callable[[CameraState, str], None] | None = None
        self.on_fps_update: Callable[[float], None] | None = None
        self.on_error: Callable[[str], None] | None = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure_local(self, camera_index: int):
        self._mode = CameraMode.LOCAL
        self._local_index = camera_index

    def configure_remote(self, host: str, port: int):
        self._mode = CameraMode.REMOTE
        self._remote_host = host
        self._remote_port = port

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self):
        """Starts capture in a separate thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._flush_queue()
        self._set_state(CameraState.CONNECTING, "Starting connection...")
        target = (
            self._run_local if self._mode == CameraMode.LOCAL
            else self._run_remote
        )
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self):
        """Stops the capture."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=4.0)
        self._set_state(CameraState.STOPPED, "Capture stopped.")

    def get_frame(self):
        """Returns the most recent frame or None."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None

    @property
    def state(self) -> CameraState:
        return self._state

    @property
    def is_streaming(self) -> bool:
        return self._state == CameraState.STREAMING

    # ------------------------------------------------------------------
    # LOCAL Capture
    # ------------------------------------------------------------------

    def _run_local(self):
        cap = cv2.VideoCapture(self._local_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self._set_state(CameraState.ERROR, f"Could not open camera {self._local_index}.")
            self._call(self.on_error, f"Camera {self._local_index} not found.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._set_state(CameraState.STREAMING, "Local camera active.")

        fps_counter = _FPSCounter()

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    self._set_state(CameraState.ERROR, "Error reading frame.")
                    break
                self._push_frame(frame)
                fps = fps_counter.tick()
                if fps is not None:
                    self._call(self.on_fps_update, fps)
        except Exception as e:
            self._set_state(CameraState.ERROR, str(e))
            self._call(self.on_error, str(e))
        finally:
            cap.release()
            if not self._stop_event.is_set():
                self._set_state(CameraState.IDLE, "Camera released.")

    # ------------------------------------------------------------------
    # REMOTE Capture
    # ------------------------------------------------------------------

    def _run_remote(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(8.0)

        try:
            self._set_state(CameraState.CONNECTING, f"Connecting to {self._remote_host}:{self._remote_port}…")
            sock.connect((self._remote_host, self._remote_port))
            sock.settimeout(10.0)
            self._set_state(CameraState.STREAMING, f"Connected to {self._remote_host}:{self._remote_port}")
        except Exception as e:
            self._set_state(CameraState.ERROR, f"Failed to connect: {e}")
            self._call(self.on_error, str(e))
            sock.close()
            return

        payload_size = struct.calcsize("Q")
        data = b""
        fps_counter = _FPSCounter()

        try:
            while not self._stop_event.is_set():
                # Receive payload size
                while len(data) < payload_size:
                    packet = sock.recv(4096)
                    if not packet:
                        raise ConnectionResetError("Server disconnected.")
                    data += packet

                packed_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("Q", packed_size)[0]

                # Receive full payload
                while len(data) < msg_size:
                    data += sock.recv(4096)

                frame_data = data[:msg_size]
                data = data[msg_size:]

                buffer = pickle.loads(frame_data)
                frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
                if frame is not None:
                    self._push_frame(frame)
                    fps = fps_counter.tick()
                    if fps is not None:
                        self._call(self.on_fps_update, fps)

        except Exception as e:
            if not self._stop_event.is_set():
                self._set_state(CameraState.ERROR, f"Connection lost: {e}")
                self._call(self.on_error, str(e))
        finally:
            sock.close()
            if not self._stop_event.is_set():
                self._set_state(CameraState.IDLE, "Disconnected.")

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _push_frame(self, frame):
        """Pushes frame to queue, discarding the oldest if full."""
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def _flush_queue(self):
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    def _set_state(self, state: CameraState, message: str = ""):
        self._state = state
        self._call(self.on_state_change, state, message)

    def _call(self, cb, *args):
        if cb:
            try:
                cb(*args)
            except Exception as e:
                logger.warning("Callback error: %s", e)


class _FPSCounter:
    """Counts FPS using a sliding 1-second window."""

    def __init__(self, window: float = 1.0):
        self._window = window
        self._timestamps: list[float] = []
        self._last_reported: float | None = None

    def tick(self) -> float | None:
        now = time.monotonic()
        self._timestamps.append(now)
        cutoff = now - self._window
        self._timestamps = [t for t in self._timestamps if t >= cutoff]
        fps = len(self._timestamps) / self._window
        # Only report every 0.5s to avoid spamming
        if self._last_reported is None or now - self._last_reported >= 0.5:
            self._last_reported = now
            return fps
        return None
