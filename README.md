
# Projeto de Reconhecimento Facial com YOLOv3 no Google Colab

Este projeto visa implementar um sistema de reconhecimento facial utilizando a rede neural YOLOv3 (You Only Look Once) para detectar faces em imagens. O treinamento é realizado no Google Colab com suporte para GPU, e o modelo é otimizado para funcionar com a biblioteca OpenCV.

## Tecnologias Utilizadas
- **YOLOv3**: Rede neural convolucional para detecção de objetos em tempo real.
- **Darknet**: Framework de código aberto para treinamento e inferência com YOLOv3.
- **Google Colab**: Ambiente de desenvolvimento baseado em Jupyter Notebook com suporte a GPU.
- **OpenCV**: Biblioteca de visão computacional para processamento de imagens.
- **Python**: Linguagem de programação para o desenvolvimento do código.

## Pré-requisitos
Para rodar o projeto, é necessário ter uma conta no Google e acesso ao Google Colab. O projeto foi desenvolvido com o suporte a GPU e OpenCV, sendo assim, é importante garantir que essas opções estejam habilitadas.

## Como Executar o Projeto

### 1. Preparar o Ambiente

Primeiro, clone o repositório do **Darknet** e altere o `Makefile` para habilitar o suporte a GPU e OpenCV:

```bash
%cd darknet
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
```

### 2. Compilação

Execute o seguinte comando para compilar o Darknet com o suporte necessário:

```bash
!make
```

### 3. Funções de Auxílio

Algumas funções de auxílio são definidas para exibir e processar imagens:

#### Exibir Imagens

A função `imShow` exibe imagens com redimensionamento adequado para visualização no Google Colab:

```python
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  %matplotlib inline

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image, (3*width, 3*height), interpolation=cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()
```

#### Upload de Arquivos

Para carregar imagens ou arquivos para o Colab:

```python
def upload():
  from google.colab import files
  uploaded = files.upload()
  for name, data in uploaded.items():
    with open(name, 'wb') as f:
      f.write(data)
      print('Arquivo salvo:', name)
```

#### Download de Arquivos

Para baixar arquivos gerados:

```python
def download(path):
  from google.colab import files
  files.download(path)
```

### 4. Treinamento do Modelo

Para iniciar o treinamento do modelo YOLOv3 com os dados e configuração personalizados, execute o seguinte comando:

```bash
!./darknet detector train data/obj.data cfg/yolov3_custom.cfg darknet53.conv.74 -dont_show
```

### 5. Teste do Modelo

Após o treinamento, você pode testar o modelo em uma imagem. Substitua o caminho da imagem e os pesos do modelo conforme necessário:

```bash
!./darknet detector test data/obj.data cfg/yolov3_custom.cfg /content/gdrive/MyDrive/backup/yolov3_custom_1000.weights buia2.jpg -thresh 0.3
imShow('predictions.jpg')
```

### 6. Resultados

O resultado da detecção será salvo em um arquivo de imagem (`predictions.jpg`), que pode ser visualizado diretamente no Colab.

## Considerações Finais

Este projeto utiliza a poderosa rede YOLOv3 para detectar rostos em imagens com alta precisão. Com o uso do Google Colab e da GPU, o treinamento é acelerado, tornando o processo mais eficiente. A integração com OpenCV permite a manipulação e exibição de imagens de forma simples e eficaz.


## Contato

Para mais informações, entre em contato através do email: [marcelopires.p@hotmail.com](mailto:marcelopires.p@hotmail.com).
