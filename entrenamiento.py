import tensorflow as tf
import cv2
import pathlib

def show_image(text, img):
  cv2.imshow(text, img)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

datacd = "C:\\Users\\Lopez\\Downloads\\Aguacates"
ruta = pathlib.Path(datacd) # manejar la ruta
image_count = len(list(ruta.glob('*/*.PNG'))) # contar la cantidad de imagenes que existen en la carpeta aguacates
print(image_count)
batch_size = 32  # significa cuantas imagenes se pasaran en el entrenemiento, por ejemplo, que se tengan 1050 muestras
# y el batch_size es de 32, significa que tomara del 1 al 32 para el entrenanmiento, despues toma las 33 al 64 y asi susecivamente hasta terminar las 1050
img_height =180  #altura
img_width = 180  #ancho

train_ds = tf.keras.preprocessing.image_dataset_from_directory( # crear un conjunto de datos
  ruta, # Directorio donde se encuentran los datos
  validation_split=0.2, # Flotador opcional entre 0 y 1, fracción de datos para reservar, utilizaremos el 80% para entrenamiento
  subset="training", # Uno de "entrenamiento" o "validación".  ---  entrenamiento
  seed=123, # numero aleatorio para transformaciones.
  image_size=(img_height, img_width),
  batch_size=batch_size)  # tamaño del lote para cada entrenmienot predeterminado es 32
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  ruta,
  validation_split=0.2,
  subset="validation", # validacion 20% para validacion
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)