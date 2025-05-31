import cv2
import cv2.data

# Crear el clasificador de cuerpo
detector_cuerpo = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Capturar video desde la camara
camara = cv2.VideoCapture(0)

if not camara.isOpened():
    print("Error: No se pudo abrir la camara")
    exit()

print("Usuario presione 'S' para salir")
while True:
    ret, frame = camara.read()
    if not ret:
        print("Error: No se pudo leer el Frame")
        break
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Mejorar el contraste de la imagen
    gray = cv2.equalizeHist(gray)
    
    # Detectar cuerpos completos
    cuerpos = detector_cuerpo.detectMultiScale(gray,scaleFactor=1.02,  minNeighbors=2 )# Tamaño máximo para un cuerpo)
    
    # Dibujar rectángulos alrededor de los cuerpos
    for (x, y, w, h) in cuerpos:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Rojo para cuerpos
        cv2.putText(frame, 'Cuerpo', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    # Mostrar la imagen
    cv2.imshow('Detector de Cuerpo', frame)
    
    # Salir con la tecla 'S'
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# Liberar recursos
camara.release()
cv2.destroyAllWindows()