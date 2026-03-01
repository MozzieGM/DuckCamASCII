import cv2
import socket
import struct
import pickle
import time

# --- Settings ----------------------------------------------------------------
CAMERA_INDEX   = 0       # Change to 1, 2... if you have more than one camera
PORT           = 9999
MAX_FAIL_READ  = 30      # Consecutive failed frames before trying to restart camera
RECONNECT_WAIT = 3.0     # Wait time in seconds between camera open attempts
# -----------------------------------------------------------------------------


def open_camera(idx: int) -> cv2.VideoCapture:
    """
    Tries to open the camera first via DSHOW (DirectShow, more stable on Windows)
    and, if it fails, tries via MSMF (OpenCV default).
    Returns the opened VideoCapture object or raises RuntimeError.
    """
    print("[INFO] Attempting to open camera with DirectShow backend (DSHOW)...")
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    time.sleep(1.5)

    if cap.isOpened():
        ret, _ = cap.read()  # test read
        if ret:
            print("[OK] Camera online via DSHOW!")
            return cap
        cap.release()

    print("[WARN] DSHOW failed. Attempting default backend (MSMF)...")
    cap = cv2.VideoCapture(idx)
    time.sleep(2.0)

    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print("[OK] Camera online via MSMF!")
            return cap
        cap.release()

    raise RuntimeError("[ERROR] Could not open camera. "
                       "Make sure another app (Teams, OBS...) is not using it.")


def start_server():
    # -- Network --------------------------------------------------------------
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('0.0.0.0', PORT))
    server_socket.listen(1)

    print("====================================")
    print("  DUCKCAM - SERVER (LAPTOP/PC)      ")
    print("====================================")

    # -- Camera (global, not closed between clients) --------------------------
    try:
        cap = open_camera(CAMERA_INDEX)
    except RuntimeError as e:
        print(e)
        return

    # -- Resolution -----------------------------------------------------------
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # -- Main Loop ------------------------------------------------------------
    while True:
        try:
            print(f"\n[INFO] Waiting for PC to connect on port {PORT}...")
            client_socket, addr = server_socket.accept()
            print(f"[OK] Connected! IP: {addr}")
            print("[INFO] Transmitting video...")

            fail_count = 0

            while True:
                ret, frame = cap.read()

                # -- Handle invalid frame -------------------------------------
                if not ret or frame is None:
                    fail_count += 1
                    if fail_count >= MAX_FAIL_READ:
                        print(f"[WARN] {fail_count} consecutive failed frames. "
                              "Restarting camera...")
                        cap.release()
                        time.sleep(RECONNECT_WAIT)
                        try:
                            cap = open_camera(CAMERA_INDEX)
                            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                            fail_count = 0
                        except RuntimeError as e:
                            print(e)
                            break  # exit frame loop, back to waiting for client
                    time.sleep(0.03)
                    continue

                fail_count = 0  # reset upon successful frame capture

                # -- Encode and send ------------------------------------------
                _, buffer = cv2.imencode(
                    '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                )
                data    = pickle.dumps(buffer)
                message = struct.pack("Q", len(data)) + data
                client_socket.sendall(message)

        except (ConnectionResetError, BrokenPipeError):
            print(f"[INFO] Client ({addr[0]}) disconnected.")
        except Exception as e:
            print(f"[WARN] Error: {e}")
        finally:
            try:
                client_socket.close()
            except Exception:
                pass
            print("[INFO] Connection closed. Waiting for new client...")


if __name__ == "__main__":
    start_server()