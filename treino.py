import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os

print('[INFO] Carregando imagens ...')
#carregando imagens
caminho_imagens = list(paths.list_images('/dataset/'))

imagens = []
labels = []
#pegando o label e a imagem e add em listas
for path in caminho_imagens:
    label = path.split(os.path.sep)[-2]
    #abrindo e processando as imagens
    imagem = load_img(path, target_size=(224, 224))
    imagem = img_to_array(imagem)
    imagem = preprocess_input(imagem)

    imagens.append(imagem)
    labels.append(label)

#transformando em array numpy
imagens = np.array(imagens, dtype='float32')
labels = np.array(labels)

# executar codificação one-hot nas etiquetas
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# particione os dados em divisões de treinamento e teste usando 75% dos dados para treinamento e os 25% restantes para teste
(trainX, testX, trainY, testY) = train_test_split(imagens, labels, test_size=0.20, stratify=labels, random_state=42)

# construindo o gerador de imagens de treinamento para aumento de dados
gerador = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

# carregando a rede MobileNetV2, garantindo que os conjuntos de camadas FC principais sejam deixados de lado
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=keras.layers.Input(shape=(224, 224, 3)))

# construindo a cabeça do modelo que será colocado em cima do modelo base
modelo_cabeca = baseModel.output
modelo_cabeca = keras.layers.AveragePooling2D(pool_size=(7, 7))(modelo_cabeca)
modelo_cabeca = keras.layers.Flatten(name="flatten")(modelo_cabeca)
modelo_cabeca = keras.layers.Dense(128, activation="relu")(modelo_cabeca)
modelo_cabeca = keras.layers.Dropout(0.5)(modelo_cabeca)
modelo_cabeca = keras.layers.Dense(2, activation="softmax")(modelo_cabeca)

#criando o modelo principal sobre o base
modelo = Model(inputs=baseModel.input, outputs=modelo_cabeca)

# percorre todas as camadas no modelo base e as congela para que elas * não * sejam atualizadas durante o primeiro processo de treinamento
for layer in baseModel.layers:
	  layer.trainable = False

#compilando o modelo
print('[INFO] compilando modelo ...')

modelo.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

# inicialize a taxa de aprendizado inicial, número de épocas para treinamento e tamanho do lote
inicia_apredizado = 1e-4
quantidade_treinos = 30
tamanho_lote = 32

# treinar a cabeça
print("[INFO] treinando...")
cabeca = modelo.fit(gerador.flow(trainX, trainY, batch_size=tamanho_lote), steps_per_epoch=len(trainX) // tamanho_lote, validation_data=(testX, testY), validation_steps=len(testX) // tamanho_lote, epochs=quantidade_treinos)

# fazendo predições na rede neural
print("[INFO] avaliando a rede neural...")
predIdxs = modelo.predict(testX, batch_size=tamanho_lote)

# para cada imagem no conjunto de testes, precisamos encontrar o índice do rótulo com a maior probabilidade prevista correspondente
predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testY.argmax(axis=1), predIdxs,	target_names=lb.classes_))

# salvando o modelo
print("[INFO] salvando o modelo...")
modelo.save("mask_detector.model", save_format="h5")

# traçar a perda e a precisão do treinamento
N = quantidade_treinos
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), cabeca.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), cabeca.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), cabeca.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), cabeca.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")