import argparse
import numpy as np
import cv2 as cv
import urllib.request

# Função de callback para os sliders (não faz nada, mas é necessária)
def on_change(value):
    pass

def load_image_from_url(url, **kwargs):
    """
    Loads an image from an Internet URL with optional arguments for OpenCV's cv.imdecode.
    
    Parameters:
    - url (str): URL of the image.
    - **kwargs: Additional keyword arguments for cv.imdecode (e.g., flags=cv.IMREAD_GRAYSCALE).
    
    Returns:
    - image: Loaded image as a NumPy array.
    """
    resp = urllib.request.urlopen(url)
    image_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    image = cv.imdecode(image_array, kwargs.get('flags', cv.IMREAD_COLOR))
    return image

# Argument parser
parser = argparse.ArgumentParser(description='Load and blend images from URLs.')
parser.add_argument('url1', type=str, help='URL of the first image')
parser.add_argument('url2', type=str, help='URL of the second image')
args = parser.parse_args()

# Carrega as imagens das URLs
f = load_image_from_url(args.url1)
g = load_image_from_url(args.url2)

# Verifica se as imagens foram carregadas corretamente
if f is None or g is None:
    print("Erro ao carregar as imagens. Verifique os URLs.")
    exit()

# Redimensiona a imagem g para o mesmo tamanho da imagem f
g = cv.resize(g, (f.shape[1], f.shape[0]))

# Normaliza as imagens para o intervalo [0, 1]
f = f.astype(np.float32) / 255.0
g = g.astype(np.float32) / 255.0

# Cria uma única janela para sliders e imagem
cv.namedWindow('Ajustes')

# Cria sliders na janela
cv.createTrackbar('a', 'Ajustes', 0, 100, on_change)
cv.createTrackbar('b', 'Ajustes', 0, 100, on_change)

while True:
    # Obtém os valores atuais dos sliders e normaliza para o intervalo [0, 1]
    a = cv.getTrackbarPos('a', 'Ajustes') / 100.0
    b = cv.getTrackbarPos('b', 'Ajustes') / 100.0

    # Aplica a convolução linear: h = a * f + b * g
    h = a * f + b * g
    h_display = (h * 255).astype(np.uint8)

    # Concatena a imagem gerada com os sliders
    sliders = np.zeros((50, f.shape[1], 3), dtype=np.uint8)  # Faixa preta para sliders
    display = np.vstack((sliders, h_display))

    # Mostra o resultado da convolução
    cv.imshow('Ajustes', display)

    # Espera por 1 ms e verifica se o usuário pressionou a tecla 'q' para sair
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Destroi todas as janelas
cv.destroyAllWindows()
