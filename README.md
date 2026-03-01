# 🦆 DuckCamASCII

Transform any camera into real-time ASCII art, with AI person segmentation (YOLO) and a modern graphical interface.

![Python](https://img.shields.io/badge/Python-3.10+-brightgreen?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue?style=flat-square)
![YOLO](https://img.shields.io/badge/YOLO-v8-orange?style=flat-square)
![CustomTkinter](https://img.shields.io/badge/GUI-CustomTkinter-purple?style=flat-square)

---

## ✨ Features

- **📷 Local Camera** — uses any camera connected to your PC (auto-detects)
- **🌐 Remote Camera** — receives video over the local network from another PC/laptop (TCP socket)
- **🤖 YOLO Segmentation** — segments only the person, removing the background (toggle on/off)
- **🎨 8 Color Themes** — Matrix, Amber, Cyan, White, Red, Purple, Rainbow, Original
- **🔡 5 Charsets** — Standard, Detailed, Blocks, Binary, Minimal
- **⚡ 4 Densities** — Low, Medium, High, Ultra
- **📸 Save Frame** — exports the ASCII frame as `.png`
- **📋 Copy ASCII** — copies the art as plain text to the clipboard
- **📊 FPS Counter** — real-time performance monitor

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/MozzieGM/DuckCamASCII.git
cd DuckCamASCII

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note:** The YOLO model (`yolov8n-seg.pt`) is downloaded automatically on the first run (~7MB).

---

## 🎮 Usage

### Local Mode (camera on the same PC)

```bash
python app.py
```

On the Setup screen:
1. Select **"Local Camera"**
2. Choose the camera from the dropdown
3. Click **"CONNECT"**

### Remote Mode (camera on another PC essentially acting as a server)

**On the laptop/PC WITH the camera:**
```bash
python server.py
```

The server will display its local IP automatically.

**On the PC WITHOUT the camera (where the app runs):**
```bash
python app.py
```

On the Setup screen:
1. Select **"Remote Camera"**
2. Enter the IP shown by `server.py`
3. Click **"CONNECT"**

---

## 🗂️ Project Structure

```
DuckCamASCII/
├── app.py              # Main GUI app (entry point)
├── ascii_engine.py     # ASCII rendering engine
├── camera_manager.py   # Local/remote camera manager
├── server.py           # Camera server (laptop)
├── requirements.txt
└── README.md
```

---

## 🛠️ Server Arguments

| Argument | Default | Description |
|-----------|--------|-----------|
| `--camera` | `0` | Camera index |
| `--port` | `9999` | TCP port |
| `--quality` | `70` | JPEG quality (1-100) |

---

## 📦 Dependencies

| Package | Usage |
|--------|-----|
| `opencv-python` | Video capture and processing |
| `ultralytics` | YOLO v8 for person segmentation |
| `customtkinter` | Modern graphical interface |
| `numpy` | Image array manipulation |
| `Pillow` | Frame conversion for GUI display |

---

## 🔑 Shortcuts

| Action | How to do it |
|------|------------|
| Save ASCII frame | "📸 Save Frame" button on the sidebar |
| Copy as text | "📋 Copy ASCII" button on the sidebar |
| Toggle YOLO | Switch on the sidebar |
| Change color theme | Radio buttons on the sidebar |
| Disconnect | "✕ Disconnect" button at the top |

---

## 📝 License

MIT License — feel free to use, modify, and distribute.

---

*Made with 💚 and lots of ASCII by MozzieGM*
