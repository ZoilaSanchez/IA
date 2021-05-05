import numpy as np
import tensorflow as tf
import os
from tensorflow import keras
import cv2

# Función para devolver el tiempo aproximado en el que se podrá consumir el aguacate.
def print_time_left(etiqueta):
    if etiqueta == 2:
        chain = "Tiempo de maduración restante aproximado: 4-5 días."
    elif etiqueta == 1:
        chain = "Está sobremadurado, no se recomienda para consumo."
    elif etiqueta == 0:
        chain = "¡Listo para comer! \nAún se puede almacenar entre 1-2 días."
    return chain


def show_image(text, img):
    cv2.imshow(text, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Se utiliza la misma función para manejar la entrada de datos de entrenamiento, principalmente para recortar el fondo
# y que la detección sea más acertadas.
def recortar(path,directorio):
    image = cv2.imread(path, 1)
    directory = directorio
    os.chdir(directory)
    width = 800
    height = 800
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (47, 47), 0)
    # blurred = cv2.GaussianBlur(blurred, (7, 7), 0)
    canny = cv2.Canny(blurred, 100, 300, apertureSize=5)  # apertura debe estar entre 3 y 7
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
    return 'img_after_cv2.jpg'
# FUNCIONES----------------------------

def tomarfoto():
    # camara
    captura = cv2.VideoCapture(0)
    while (True):
        # Caputramos la foto
        return_Varlor, image = captura.read();
        # visualizar
        if return_Varlor:
            cv2.imshow('visor', image)
        # salir
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('opencv1.png', image)
            del (captura)
            break
    return 'opencv1.PNG'

def recorte(carguardar, imgaleer, cargardatos, cargarmodelo, imgaen2):
    directory = carguardar
    img = cv2.imread(imgaleer, 1)
    dim = (760, 760)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    os.chdir(directory)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 8)
    blurred = cv2.GaussianBlur(blurred, (7, 7), 0)
    canny = cv2.Canny(blurred, 100, 300, apertureSize=3)  # apertura debe sesta entre 3 y 7
    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)

    i = 1
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        recortada = img[y:y + h, x:x + w]
        show_image('imagen no:' + str(i), recortada)
        cv2.imwrite('img' + str(i) + '.PNG', recortada)
        procesoreconocer(cargardatos, cargarmodelo, 'img'+str(i)+".PNG", imgaen2)
        i += 1
    return i-1
# FUNCIONES----------------------------
def procesoreconocer(cargardatos,cargarmodelo,imagendemuestra,imgaen2):
    # Se carga el nombre de las carpetas donde se encuentran las imágenes, correspondiendo a las etiquetas que utiliza
    # el modelo guardado
    images_path = cargardatos
    mylist = os.listdir(images_path)
    # Se carga el modelo previamente entrenado (En este caso es un modelo con 50 épocas y con data augmentation)
    keras_model_path = cargarmodelo
    restored_keras_model = tf.keras.models.load_model(keras_model_path)

    # Se carga la imagen y se muestra el proceso para quitar el fondo.
    img_path = imagendemuestra
    recortar(img_path, imgaen2)
    img_height = 180
    img_width = 180
    # Se carga la imágen a analizar con un preprocesamiento que la estandariza al tamaño con el que se entrenó la CNN
    pathimg = imgaen2 + '/img_after_cv2.jpg'
    img = keras.preprocessing.image.load_img(
        pathimg,
        target_size=(img_height, img_width)
    )
    # Se utiliza este método para pasar la imágen a un array 3d que la red neuronal pueda interpretar
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    # Se utiliza el método .predict(img) para que abtenga las probabilidades de pertenecer a cada clase, y almacenamos
    # únicamente la más alta.
    predictions = restored_keras_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    # Devuelve al usuario la clase con mayor probabilidad.
    print("--------  RESULTADOS   --------")
    print(
        "Este aguacate pertenece a la etiqueta {} con un {:.2f}% de certeza."
            .format(mylist[np.argmax(score)], 100 * np.max(score))
        )
    print(print_time_left(np.argmax(score)))

