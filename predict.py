import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
import cv2

# Se carga el nombre de las carpetas donde se encuentran las imágenes, correspondiendo a las etiquetas que utiliza
# el modelo guardado
images_path = 'C:/Users/ACER/Desktop/Copia/IA/Pruebas opencv/Aguacates'
mylist = os.listdir(images_path)
# Se carga el modelo previamente entrenado (En este caso es un modelo con 50 épocas y con data augmentation)
keras_model_path = "C:/Users/ACER/Desktop/Copia/IA/Modelo2"
restored_keras_model = tf.keras.models.load_model(keras_model_path)


def show_image(text, img):
    cv2.imshow(text, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Se utiliza la misma función para manejar la entrada de datos de entrenamiento, principalmente para recortar el fondo
# y que la detección sea más acertadas.
def recortar(path):
    image = cv2.imread(path, 1)
    directory = 'D:/URL/Séptimo ciclo 2021/Inteligencia artificial/Phyton/predict_avocado'
    os.chdir(directory)
    width = 800
    height = 800
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (47, 47), 0)
    # blurred = cv2.GaussianBlur(blurred, (7, 7), 0)
    canny = cv2.Canny(blurred, 100, 300, apertureSize=5)     # apertura debe estar entre 3 y 7
    show_image('canny', canny)
    edges = cv2.dilate(canny, None)
    show_image('dilate', edges)
    for i in range(0, 3):
        edges = cv2.dilate(edges, None)
    show_image('dilate', edges)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        recortada = resized[y:y + h, x:x + w]
        show_image('imagen', recortada)
        cv2.imwrite('img_after_cv2.jpg', recortada)
        break


# Se carga la imagen y se muestra el proceso para quitar el fondo.
img_path = 'C:/Users/ACER/Desktop/Pruebas/avo1.jpg'
recortar(img_path)
img_height = 180
img_width = 180
# Se carga la imágen a analizar con un preprocesamiento que la estandariza al tamaño con el que se entrenó la CNN
img = keras.preprocessing.image.load_img(
    'D:/URL/Séptimo ciclo 2021/Inteligencia artificial/Phyton/predict_avocado/img_after_cv2.jpg', target_size=(img_height, img_width)
)
# Se utiliza este método para pasar la imágen a un array 3d que la red neuronal pueda interpretar
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch
# Se utiliza el método .predict(img) para que abtenga las probabilidades de pertenecer a cada clase, y almacenamos
# únicamente la más alta.
predictions = restored_keras_model.predict(img_array)
score = tf.nn.softmax(predictions[0])
# Devuelve al usuario la clase con mayor probabilidad.
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(mylist[np.argmax(score)], 100 * np.max(score))
)
