import cv2
import socket
import struct
import pickle
import numpy as np
from ultralytics import YOLO

# === INICIALIZA A IA DEFINITIVA (YOLO) ===
# Ele baixa esse arquivo nanoscópico sozinho na 1ª vez que rodar
print("Carregando o modelo de IA...")
modelo_ia = YOLO('yolov8n-seg.pt')

# === CONFIGURAÇÃO DO ASCII ===
ASCII_CHARS = [" ", ".", ",", ":", ";", "+", "*", "?", "%", "S", "#", "@"]

def draw_ascii_image(gray_frame):
    height, width = gray_frame.shape
    char_width = 8
    char_height = 12
    
    cols = width // char_width
    rows = height // char_height
    
    resized_gray = cv2.resize(gray_frame, (cols, rows))
    ascii_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            pixel = resized_gray[i, j]
            
            # Pula os pixels pretos do fundo falso pra economizar CPU
            if pixel < 10:
                continue
                
            index = pixel // 22
            index = min(index, len(ASCII_CHARS) - 1)
            char = ASCII_CHARS[index]
            
            # Cor verde Matrix
            cv2.putText(ascii_img, char, (j * char_width, (i * char_height) + char_height),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
            
    return ascii_img

# === CONEXÃO COM O SERVIDOR ===
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '192.168.0.107' # Lembre-se de confirmar seu IP do notebook
port = 9999

print("Conectando ao notebook...")
client_socket.connect((host_ip, port))
print("Conectado! Abrindo a Matrix...")

data = b""
payload_size = struct.calcsize("Q") 

try:
    while True:
        # --- Recebimento do Vídeo ---
        while len(data) < payload_size:
            packet = client_socket.recv(4*1024)
            if not packet: break
            data += packet
        
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]
        
        while len(data) < msg_size:
            data += client_socket.recv(4*1024)
        
        frame_data = data[:msg_size]
        data = data[msg_size:]
        
        buffer = pickle.loads(frame_data)
        frame_colorido = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        
        # === A MÁGICA DA IA (YOLO) ===
        # classes=[0] manda a IA procurar apenas por "Pessoas"
        resultados = modelo_ia.predict(frame_colorido, classes=[0], verbose=False, conf=0.5)
        
        fundo_preto = np.zeros_like(frame_colorido)
        
        # Se a IA encontrou você e gerou a máscara de recorte...
        if resultados[0].masks is not None:
            # Pega a máscara da primeira pessoa que ele viu
            mascara = resultados[0].masks.data[0].cpu().numpy()
            
            # Ajusta o tamanho da máscara para bater certinho com o vídeo
            mascara = cv2.resize(mascara, (frame_colorido.shape[1], frame_colorido.shape[0]))
            
            # Converte para o formato correto (Branco = Você, Preto = Fundo)
            mascara_cv2 = (mascara > 0.5).astype(np.uint8) * 255
            
            # Usa a máscara como uma "tesoura" para recortar você
            frame_pessoa = cv2.bitwise_and(frame_colorido, frame_colorido, mask=mascara_cv2)
        else:
            # Se você sair da frente da câmera, a tela fica toda preta
            frame_pessoa = fundo_preto
            
        # === GERAÇÃO DO ASCII ===
        frame_cinza = cv2.cvtColor(frame_pessoa, cv2.COLOR_BGR2GRAY)
        frame_ascii = draw_ascii_image(frame_cinza)
        
        # === MOSTRAR JANELAS ===
        cv2.imshow("1. Câmera Original", frame_colorido)
        cv2.imshow("2. Recorte com YOLO", frame_pessoa)
        cv2.imshow("3. ASCII Isolado", frame_ascii)
        
        if cv2.waitKey(1) == ord('q'):
            break

except Exception as e:
    print(f"\nConexão encerrada: {e}")
finally:
    client_socket.close()
    cv2.destroyAllWindows()