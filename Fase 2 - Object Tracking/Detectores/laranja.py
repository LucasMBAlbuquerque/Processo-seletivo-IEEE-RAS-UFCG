import cv2
import imutils
import numpy as np

# Define a faixa de cor laranja em HSV
laranja_alto = np.array([10, 150, 150])
laranja_baixo = np.array([40, 255, 255])

# Inicia o objeto de captura de vídeo
# Caso queira usar sua webcam mude o valor do cv2.VideoCapture para 0 (cv2.VideoCapture(0))
cap = cv2.VideoCapture(r"Fase 2 - Object Tracking\Videos\laranja.mp4")

# delay para o rastro do objeto sumir
intervalo = 6
# Inicializa a variável que armazena a trajetória do objeto
trajectory = []

while True:
    # Captura um quadro do vídeo
    ret, frame = cap.read()
    
    # Redimensiona o quadro para facilitar o processamento
    frame = imutils.resize(frame, width=600)
    
    # Converte o quadro para o espaço de cor HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Cria uma máscara que filtra a cor laranja
    mask = cv2.inRange(hsv, laranja_alto, laranja_baixo)
    
    # Encontra os contornos dos objetos laranja na máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Se houver pelo menos um objeto laranja, encontra o contorno com a área máxima
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        
        # Encontra o centro do contorno com a área máxima
        M = cv2.moments(c)

        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            center = (None, None)

        trajectory.append(center)
        # Encontra o maior contorno laranja
        maior_contorno = None
        maior_area = 0
        for contorno in contours:
            area = cv2.contourArea(contorno)
            if area > maior_area:
                maior_area = area
                maior_contorno = contorno
        # Desenha o contorno e o centro do objeto laranja no quadro atual
        if maior_contorno is not None:
            x, y, w, h = cv2.boundingRect(maior_contorno)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
        # Desenha a trajetória no quadro atual
        for i in range(1, len(trajectory)):
            cv2.line(frame, trajectory[i - 1], trajectory[i], (0, 0, 255), 2)
        # Remove a trajetória após o intervalo de tempo definido
        if len(trajectory) > intervalo:
            trajectory.pop(0)    
    # Mostra o quadro atual na tela
    cv2.imshow("Frame", frame)
    # Verifica se a tecla 'q' foi pressionada para sair do loop. Também controla a velocidade do video caso não use webcam
    if cv2.waitKey(10) == ord('q'):
        break

# Libera os recursos utilizados
cap.release()
cv2.destroyAllWindows()