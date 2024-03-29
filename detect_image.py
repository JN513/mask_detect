#!/usr/bin/python
# -*- coding: UTF-8 -*-
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import sys


def make_help():
    print("\nArgumentos disponiveis:\n")
    print("    -s -> Para salvar a imagem")
    print(
        "    -i -> para passar a imagem, necessita do path da imagem como proximo argumento"
    )
    print("    -h or --help -> ajuda")


def salva_img(
    img, path
):  # Salvo a imagem com um .det antes da extenção para identificar que ela passou pelo detector
    paths = path.split("/")
    if paths != path:
        new_path = paths[-1].split(".")
        new_path[-2] += ".det"

        path_f = ""
        for s in new_path:
            path_f += s + "."

        path_f = path_f[:-1]

        diretorio = ""

        for i in range(0, len(paths) - 1):
            diretorio += paths[i] + "/"

        print(f"[ INFO ] Salvando imagem em: {diretorio+path_f}")
        cv2.imwrite(str(diretorio + path_f), img)  # Salva a imagem
    else:
        new_path = paths.split(".")
        new_path[-2] += ".det"

        path_f = ""
        for s in new_path:
            path_f += s

        cv2.imwrite(path_f, img)


os.system("clear || cls")

# carrega modelo de detector de rosto serializado a partir do disco
print("[ INFO ] carregando modelo detector de faces...")
prototxtPath = os.path.sep.join(["classificadores/deploy.prototxt"])
weightsPath = os.path.sep.join(
    ["classificadores/res10_300x300_ssd_iter_140000.caffemodel"]
)
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# carregar o modelo do detector de máscara facial a partir do disco
print("[ INFO ] carregando modelo...")
modelo = load_model("classificadores/mask_detector.model")

os.system("clear || cls")

# carrega a imagem

print("[ INFO ] carregando imagem...")

path = ""
save = False
help = False

if "-h" in sys.argv:
    sys.argv.remove("-h")
    help = True
if "--help" in sys.argv:
    sys.argv.remove("--help")
    help = True
if "-s" in sys.argv:
    save = True
    sys.argv.remove("-s")
if len(sys.argv) > 1:
    if sys.argv[1] == "-i":
        path = sys.argv[2]
    else:
        print("[ INFO ] Insira um argumento valido")
        sys.exit()
else:
    path = "testes/3.png"


imagem = cv2.imread(path)  # lendo imagem

try:
    origem = imagem.copy()  # copiando imagem
    print("[ INFO ] Imagem aberta")
except:  # Caso não caonsiga abrir a imagem o codigo morre
    print("[ INFO ] Imagem não encontrada")
    sys.exit()

(h, w) = imagem.shape[:2]

# construir um blob a partir da imagem
blob = cv2.dnn.blobFromImage(imagem, 1.0, (300, 300), (104.0, 177.0, 123.0))
# passe o blob pela rede e obtenha as detecções de rosto

print("[ INFO ] computando deteccoes faciais...")
net.setInput(blob)
deteccoes = net.forward()


# loop sobre as detecçoes
for i in range(0, deteccoes.shape[2]):
    # extrair a confiança (ou seja, probabilidade) associada à detecção
    confianca = deteccoes[0, 0, i, 2]
    # filtrar detecções fracas, garantindo que a confiança seja maior que a confiança mínima
    if confianca > 0.5:
        # calcular as coordenadas (x, y) da caixa delimitadora para o objeto
        box = deteccoes[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        # verifique se as caixas delimitadoras estão dentro das dimensões do

        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        # extrai o ROI da face, converte-o de pedido de canal BGR para RGB, redimensione-o para 224x224 e pré-processe
        face = imagem[startY:endY, startX:endX]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        # passa a imagem no modelo, para verificar a exixtencia de uma mascara ou não
        (mask, withoutMask) = modelo.predict(face)[0]

        # determinado a label e a cor do quadrado
        label = "Com mascara" if mask > withoutMask else "Sem mascara"
        color = (0, 255, 0) if label == "Com mascara" else (0, 0, 255)

        # inclui a probabilidade na label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        # exibir a label e o retângulo da caixa delimitadora no quadro de saída

        cv2.putText(
            imagem,
            label,
            (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
        )
        cv2.rectangle(imagem, (startX, startY), (endX, endY), color, 2)

# Mostra a imagem final

if save == True:
    salva_img(imagem, path)

if help == True:
    make_help()

cv2.imshow("Janela", imagem)
cv2.waitKey(0)
