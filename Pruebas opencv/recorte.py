import numpy as np
import cv2
import os


# Método utilizado para mostrar cada paso que se va haciendo.
def show_image(text, img):
    cv2.imshow(text, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def recortar(imgpath, num):
    img = cv2.imread(imgpath, 1)
    directory = 'C:/Users/ACER/Desktop/Salida/'
    # Se define la dirección de donde se escribirá cada imágen nueva
    os.chdir(directory)
    # Como una prueba, se estandariza el tamaño de las imágenes a 800x800 para que se pueda visualizar los cambios de
    # mejor manera, debido a que están tomadas en alta calidad.
    width = 800
    height = 800
    # Se sefine el ancho por alto
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # show_image('resized', resized)
    # Se pasa a escala de grises la imágen con el nuevo tamaño
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # show_image('original', gray)
    # Se hace uso de los desenfoques gaussianos, primero usando un kernel de 5x5 con un cv.BORDER_CONSTANT
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # show_image('grises gaussian blur', blurred)
    # Luego usamos otro kernel de 7x7, pero mantenemos el tipo de borde
    blurred = cv2.GaussianBlur(blurred, (7, 7), 0)
    # show_image('grises gaussian blur', blurred)
    # Se hace uso de el detector de bordes cv2.canny, en el que se mandan como parámetros un umbral por histéresis en el
    # cual si un pixel es mayor o está entre los dos valores se toma como parte del borde, caso contrario se descarta.
    # El valor de la apertura se aumentó para que detectara más detalles, aumentandolo a 5.
    canny = cv2.Canny(blurred, 100, 300, apertureSize=5)     # apertura debe estar entre 3 y 7
    # show_image('canny', canny)
    # Aunque ahora se detectan más detalles, necesitamos un contorno más grande que represente el borde del aguacate,
    # por lo tanto se hace uso de cv2.dilate, que lo que hace es aumentar la región blanca que se encuentra en la
    # máscara hecha con canny, lo que permite unir partes del contorno que puedan estar ligeramente separadas.
    edges = cv2.dilate(canny, None)
    # show_image('dilate', edges)
    # Ahora que se aumenta la probabilidad de tener un contorno que englobe al aguacate, se procede a crear el array que
    # contenga la información de los contornos hallados.
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Se ordena una lista en la que el primer elemento será el que almacene el área más grande dentro de su contorno.
    # Esto nos servirá para obtener el borde que nos interesa, pues generalmente será el del aguacate.
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)


    # Se recorre el vector donde se almacenan los contornos, y se procede a recortar en una nueva imágen el contorno con
    # el área más grande, teniendo así nuestro aguacate recortado lo mejor posible para evitar información innecesaria
    # pues las imágenes obtenidas servirán para el entrenamiento o validación.
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        recortada = resized[y:y + h, x:x + w]
        # show_image('imagen no:' + str(i), recortada)
        cv2.imwrite('img' + str(num) + '.jpg', recortada)
        break


# Se hace un ciclo que recorra todos los elementos de la carpeta elegida y obtener estas mismas imágenes pero sin la
# información innecesaria
images_path = 'C:/Users/ACER/Desktop/Copia/IA/Aguacates/MADURO'
mylist = os.listdir(images_path)
i = 1
for pos in mylist:
    full_path = images_path + "/" + pos
    recortar(full_path, i)
    i += 1

