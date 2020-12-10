# Mask_detect

Sistema feito para detecção facial e reconhecimento se a pessoa esta usando mascara ou não, atraves de Visão computacional, Machine larning e deep larning.

![Imagem de pessoas com e sem mascaras faciais](https://raw.githubusercontent.com/JN513/mask_detect/master/testes/teste.det.jpg)

## Requisitos

* Versão 3.0 ou superior do Python (testei na 3.8)
* Tensor Flow
* Open CV
* Numpy
* Cython
* Codecov
* Pytest-cov
* Pytesseract
* Wand
* Sklearn
* Imutils
* Matplotlib

## Instalação

Para instalar todas as Bibliotecas, basta istalar usando o arquivo **requirements.txt**, com o seguinte comando:

```
pip3 install -r requirements.txt
``` 

## Organização do projeto

O projeto conta com 3 arquivos: 

**Arquivo de treino:** Para treinar o modelo, basta rodar o arquivo **treino.py**. ( É válido destacar que o projeto conta com banco de imagens com 690 imagens de pessoas com máscaras e 686, sem elas. Para adicionar uma imagem, basta colocá-la na pasta “dataset”).

**Arquivo de detecção através de uma imagem:** Para realizar o processo de detecção, basta utilizar o arquivo detect_image.py e passar os argumentos **-i** e o **path da imagem**. Como opcional o argumento **-s** salva a imagem processada.

Exemplo: 

```
python3 detect_image.py -i testes/1.png
```

**Arquivo de detecção através de um vídeo ou câmera:** Para utilizar a câmera padrão de seu dispositivo, basta rodar o arquivo **detect_camera.py**.  

Exemplo: 

```
python3 detect_camera.py
```

Entretanto, caso queira usar uma câmera externa ou o aparelho eletrônico via droidcam, basta colocar o IP da câmera na linha 63.

Exemplo:

```python
captura_video = cv2.VideoCapture("192.168.0.102:4224")
```

Por fim, para fazer uso de um vídeo, basta passar o **path** dele na mesma linha já citada.

## Créditos

Esse projeto foi baseado no projeto de [Gorpo](https://github.com/gorpo/Face-Recognition-Detector-de-Mascara-Python-Covid-19) (Como não sei o nome dele, fica o nick mesmo haha).
