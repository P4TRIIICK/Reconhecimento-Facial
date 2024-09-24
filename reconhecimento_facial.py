import cv2

# Inicializa a captura de vídeo da câmera (0 representa a câmera padrão)
camera = cv2.VideoCapture(0)

# Carrega o classificador de detecção de face
detector_face = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Captura o frame da câmera
    ret, frame = camera.read()
    
    if not ret:
        print("Falha ao capturar imagem")
        break
    
    # Converte o frame para escala de cinza
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecta faces no frame
    deteccoes = detector_face.detectMultiScale(frame_cinza, scaleFactor=1.3, minSize=(30, 30))
    
    # Desenha retângulos ao redor das faces detectadas
    for (x, y, l, a) in deteccoes:
        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)
    
    # Mostra o frame com as detecções
    cv2.imshow("Detecção Facial", frame)
    
    # Pressione 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
camera.release()
cv2.destroyAllWindows()
