import cv2
import socket
import struct
import pickle

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# === SUBSTITUA AQUI PELO IP DO SEU NOTEBOOK ===
host_ip = '192.168.0.107' 
port = 9999

print("Conectando ao notebook...")
client_socket.connect((host_ip, port))
print("Conectado! Recebendo vídeo...")

payload_size = struct.calcsize("Q") 

try:
    while True:
        packed_msg_size = recvall(client_socket, payload_size)
        if not packed_msg_size: break
            
        msg_size = struct.unpack("Q", packed_msg_size)[0]
        
        frame_data = recvall(client_socket, msg_size)
        if not frame_data: break
        
        buffer = pickle.loads(frame_data)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        
        cv2.imshow("Câmera do Notebook", frame)
        
        # Aperte 'q' para fechar
        if cv2.waitKey(1) == ord('q'):
            break
except Exception as e:
    print(f"Conexão encerrada: {e}")
finally:
    client_socket.close()
    cv2.destroyAllWindows()