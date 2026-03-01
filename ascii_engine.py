"""
ascii_engine.py - ASCII rendering engine for DuckCamASCII
Converts video frames (numpy arrays) into colored ASCII images.
"""

import cv2
import numpy as np

# ===== CHARACTER SETS =====
CHARSETS = {
    "Standard":  [" ", ".", ",", ":", ";", "+", "*", "?", "%", "S", "#", "@"],
    "Detailed":  [" ", ".", "'", "`", "^", "\"", ",", ":", ";", "I", "l", "!", "i", ">",
                  "<", "~", "+", "_", "-", "?", "]", "[", "}", "{", "1", ")", "(", "|",
                  "\\", "/", "t", "f", "j", "r", "x", "n", "u", "v", "c", "z", "X", "Y",
                  "U", "J", "C", "L", "Q", "0", "O", "Z", "m", "w", "q", "p", "d", "b",
                  "k", "h", "a", "o", "*", "#", "M", "W", "&", "8", "%", "B", "@", "$"],
    "Blocks":    [" ", "░", "▒", "▓", "█"],
    "Binary":    [" ", "0", "1"],
    "Minimal":   [" ", ".", ":", "|", "+", "X", "#", "@"],
}

# ===== COLOR THEMES (BGR) =====
COLOR_THEMES = {
    "Matrix":   {"mode": "solid", "color": (0, 255, 0)},       # Matrix Green
    "Amber":    {"mode": "solid", "color": (0, 165, 255)},     # Amber
    "Cyan":     {"mode": "solid", "color": (255, 220, 0)},     # Cyan
    "White":    {"mode": "solid", "color": (220, 220, 220)},   # Soft White
    "Red":      {"mode": "solid", "color": (50, 50, 255)},     # Red
    "Purple":   {"mode": "solid", "color": (220, 80, 180)},    # Purple
    "Rainbow":  {"mode": "rainbow", "color": None},            # Rainbow per row
    "Original": {"mode": "original", "color": None},           # Original pixel color
}

# ===== DENSITY PRESETS =====
DENSITY_PRESETS = {
    "Low":    (14, 20),   # (char_width, char_height)
    "Medium": (9, 14),
    "High":   (6, 10),
    "Ultra":  (4, 7),
}


class ASCIIEngine:
    """Renderer for ASCII art from video frames."""

    def __init__(
        self,
        charset_name: str = "Standard",
        theme_name: str = "Matrix",
        density_name: str = "Medium",
        font_scale: float = 0.75,
        font_thickness: int = 1,
    ):
        self.set_charset(charset_name)
        self.set_theme(theme_name)
        self.set_density(density_name)
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.font = cv2.FONT_HERSHEY_PLAIN

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_charset(self, name: str):
        self.charset_name = name
        self.chars = CHARSETS.get(name, CHARSETS["Standard"])

    def set_theme(self, name: str):
        self.theme_name = name
        self.theme = COLOR_THEMES.get(name, COLOR_THEMES["Matrix"])

    def set_density(self, name: str):
        self.density_name = name
        self.char_w, self.char_h = DENSITY_PRESETS.get(name, DENSITY_PRESETS["Medium"])

    # ------------------------------------------------------------------
    # Main Rendering
    # ------------------------------------------------------------------

    def render(self, frame_bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """
        Converts a color frame into a rendered ASCII image.

        Args:
            frame_bgr: OpenCV BGR frame.
            mask: Binary mask (0-255). Only white pixels are rendered.
                  If None, renders the entire frame.

        Returns:
            BGR image with ASCII characters drawn.
        """
        h, w = frame_bgr.shape[:2]
        cols = max(1, w // self.char_w)
        rows = max(1, h // self.char_h)

        # Grayscale for brightness → character mapping
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)

        # Small color frame (for "original" mode)
        color_small = cv2.resize(frame_bgr, (cols, rows), interpolation=cv2.INTER_AREA)

        # Small mask
        if mask is not None:
            mask_small = cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_NEAREST)
        else:
            mask_small = None

        # Black output canvas
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        mode = self.theme["mode"]
        base_color = self.theme["color"]
        n_chars = len(self.chars)

        # Row index cache for rainbow
        rainbow_colors = self._make_rainbow_palette(rows) if mode == "rainbow" else None

        for row in range(rows):
            y = row * self.char_h + self.char_h  # text baseline
            if y > h:
                break

            for col in range(cols):
                # Check mask
                if mask_small is not None and mask_small[row, col] < 128:
                    continue

                pixel = int(gray_small[row, col])
                if pixel < 8:  # skip near-black pixels
                    continue

                idx = min(int(pixel / 255 * (n_chars - 1)), n_chars - 1)
                char = self.chars[idx]
                if char == " ":
                    continue

                x = col * self.char_w

                # Character color
                if mode == "solid":
                    color = base_color
                elif mode == "rainbow":
                    color = rainbow_colors[row]
                else:  # original
                    b, g, r = color_small[row, col]
                    color = (int(b), int(g), int(r))

                cv2.putText(
                    canvas, char,
                    (x, y),
                    self.font,
                    self.font_scale,
                    color,
                    self.font_thickness,
                    cv2.LINE_AA,
                )

        return canvas

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_rainbow_palette(n_rows: int):
        """Generates a list of BGR colors for a per-row rainbow effect."""
        colors = []
        for i in range(n_rows):
            hue = int(i / max(n_rows - 1, 1) * 179)
            hsv = np.array([[[hue, 255, 220]]], dtype=np.uint8)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
        return colors

    def get_text_output(self, frame_bgr: np.ndarray, cols: int = 80) -> str:
        """
        Returns the ASCII art as pure text (for copying/exporting).

        Args:
            frame_bgr: BGR Frame.
            cols: Number of text columns.

        Returns:
            Newline-separated string.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        rows = int(cols * 0.45)  # monospace aspect ratio
        small = cv2.resize(gray, (cols, rows), interpolation=cv2.INTER_AREA)
        n_chars = len(self.chars)
        lines = []
        for row in range(rows):
            line = ""
            for col in range(cols):
                pixel = int(small[row, col])
                idx = min(int(pixel / 255 * (n_chars - 1)), n_chars - 1)
                line += self.chars[idx]
            lines.append(line)
        return "\n".join(lines)
