from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

# carrega modelo de detector de rosto serializado a partir do disco
print("[INFO] carregando modelo detector de rosto ...")
prototxtPath = os.path.sep.join(["classificadores/deploy.prototxt"])
weightsPath = os.path.sep.join(	["classificadores/res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# carregar o modelo do detector de máscara facial a partir do disco
print("[INFO] carregando modelo de detector de máscara facial ...")
maskNet = load_model("classificadores/mask_detector.model")

def detecta_e_predicao_de_mascara(frame):
	# pegue as dimensões do quadro e construa um blob a partir dele
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),(104.0, 177.0, 123.0))
	# passe o blob pela rede e obtenha as detecções de rosto
	faceNet.setInput(blob)
	detections = faceNet.forward()
	# inicialize nossa lista de rostos, seus locais correspondentes e a lista de previsões da nossa rede de máscaras faciais
	faces = []
	locs = []
	preds = []
	# loop nas detecções
	for i in range(0, detections.shape[2]):
		# extrair a confiança (ou seja, probabilidade) associada à detecção
		confidence = detections[0, 0, i, 2]
		# filtrar detecções fracas, garantindo que a confiança seja maior que a confiança mínima
		if confidence > 0.5:
			# calcular as coordenadas (x, y) da caixa delimitadora para o objeto
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# verifique se as caixas delimitadoras estão dentro das dimensões do quadro
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
			# extrai o ROI da face, converte-o de pedido de canal BGR para RGB, redimensione-o para 224x224 e pré-processe-o
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)
			# adicione as caixas de rosto e delimitadoras às respectivas listas
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# apenas faça previsões se pelo menos uma face for detectada
	if len(faces) > 0:
		# Em vez de fazer um loop for, fazemos uma predição de todos ao mesmo tempo, ou seja passamos uma lista de faces, ou um array numpy de shape = 4 
		preds = maskNet.predict(faces)
	# retorna duas tuplas dos locais de face e seus locais correspondentes
	return (locs, preds)

print("[INFO] Iniciando a camera ...")
captura_video = cv2.VideoCapture(0)


try:
    while(True):
        captura_ok, frame = captura_video.read()

        if captura_ok:
            frame = imutils.resize(frame, width=400)

            (locais, predicaos) = detecta_e_predicao_de_mascara(frame)

            for (box, predicao) in zip(locais, predicaos):
                # descompacte a caixa delimitadora e as previsões
                (startX, startY, endX, endY) = box
                (mascara, semmascara) = predicao

                label = "Com mascara" if mascara > semmascara else "Sem mascara"
                color = (0, 255, 0) if label == "Com mascara" else (0, 0, 255)

                label = "{}: {:.2f}%".format(label, max(mascara, semmascara) * 100)

                # exibe os textos da label no frame
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            cv2.imshow('janela',frame)

            k = cv2.waitKey(30) & 0xff #pega a tecla esc
            if k == 27: #caso esc for apertado para
                break

except KeyboardInterrupt:
    captura_video.release()
    print("Interrompido")