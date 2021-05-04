import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers

def show_image(text, img):
    cv2.imshow(text, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


datacd = "C:/Users/ACER/Desktop/Salida"
ruta = pathlib.Path(datacd)  # manejar la ruta
image_count = len(list(ruta.glob('*/*.PNG')))  # contar la cantidad de imagenes que existen en la carpeta aguacates
print(image_count)
batch_size = 32  # significa cuantas imagenes se pasaran en el entrenemiento, por ejemplo, que se tengan 1050 muestras
# y el batch_size es de 32, significa que tomara del 1 al 32 para el entrenanmiento, despues toma las 33 al 64 y asi susecivamente hasta terminar las 1050
img_height = 180  # altura
img_width = 180  # ancho

train_ds = tf.keras.preprocessing.image_dataset_from_directory(  # crear un conjunto de datos
    ruta,  # Directorio donde se encuentran los datos
    validation_split=0.3,
    # Flotador opcional entre 0 y 1, fracción de datos para reservar, utilizaremos el 70% para entrenamiento
    subset="training",  # Uno de "entrenamiento" o "validación".  ---  entrenamiento
    seed=123,  # numero aleatorio para transformaciones.
    image_size=(img_height, img_width),
    batch_size=batch_size)  # tamaño del lote para cada entrenmienot predeterminado es 32
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    ruta,
    validation_split=0.3,
    subset="validation",  # validacion 30% para validacion
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
# Se utiliza la captación previa del búfer para que no haya cuellos de botella
AUTOTUNE = tf.data.AUTOTUNE
# Se utiliza .cache() para mantener las imágenes en memoria después de cargarlas del disco en la primer época. Esto
# se hace para que el dataset no se convierta en un cuello de botella mientras se entrena el modelo.
# .prefetch se utiliza para dar más importancia al preprocesamiento y la ejecución del modelo mientras entrena.
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Debido a que se tiene un relativamente pequeño set de datos, se usa data augmentation que realiza transformaciones
# a los sets ya definidos para crear más variedad y que se generalice mejor.
data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal",
                                                     input_shape=(img_height,
                                                                  img_width,
                                                                  3)),
        # Se define una rotación y un zoom pequeños para que las imágenes sigan pareciendo "naturales".
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomZoom(0.1),
    ]
)

# Se define la cantidad de labels que hay, en el caso de clasificación de estados de aguacates está tierno,
# maduro y podrido
num_classes = 3
# Se utiliza el modelo secuencial, que son varias capas lineares apiladas Primero se estandarizan los datos,
# Puesto que los datos de los canales de colores están entre 0-255 es conveniente hacer los valores de entrada lo más
# pequeños posibles. Por lo tanto se aplica la función Rescaling para que queden entre valores de 0-1. También se
# define que entraran imágenes de 128x128 con los 3 canales de colores.
model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
    # El primer parámetro para conv2D es el número de filtos que la capa convolucional aprenderá. Este parámetro
    # determina el número de kernels que se convolucionarán con el volumen de entrada. Luego, como segundo parámetro
    # está el tamaño del kernel, que especifica el ancho y alto de la ventana convolucional 2D. Como tercer parámetro
    # está el padding, lo utilizamos en same puesto que se quiere que el volumen de salida sea el mismo que de entrada,
    # Finalmente se define que se utilizará la activación RELU, pues es la recomendada para capas escondidas.
    layers.Conv2D(16, (7, 7), padding='same', activation='relu'),
    # Luego se hace un max pooling para ayudar a ajustar la representación como para reducir el costo computacional
    # reduciendo el número de parámetros a aprender
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (5, 5), padding='same', activation='relu'),
    layers.MaxPooling2D((3, 3)),
    layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D((3, 3)),
    layers.Flatten(),
    # Se agregan dos capas ocultas intermedias de neuronas totalmente conectadas de 64 y 16 neuronas.
    # Se han añadido capas de dropout así como regularización para evitar el overfitting y que la CNN aprenda mejor.
    layers.Dropout(0.6),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.003)),
    layers.Dropout(0.5),
    layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.003)),
    # Se define la capa de nodos de salida, usando 3 nodos porque solo hay 3 labels.
    layers.Dense(num_classes)
])

# Se uilizará el optimizador adam, que es un método de descenso estocástico del gradiente que se basa en una
# estimación adaptativa de momentos de primer y segundo orden.  También se usará el modelo Cross-Entropy que compara
# la clase predecida con la clase deseada y con valores entre 0 y 1 calcula el cambio de los pesos durante el
# entrenamiento.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# Se muestra el resumen de las capas agregadas y la cantidad de parámetos a procesar.
model.summary()
# Serán 50 épocas y se entrenara en base al dataset de entrenamiento hecho anteriormente, validandose con el dataset
# de validación.
epochs = 150
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Mediante este método se puede guardar el resultado del entrenamiento para su uso en otro script para realizar
# únicamente la predicción con nuevos objetos.
keras_model_path = "C:/Users/ACER/Documents/GitHub/IA/Modelo5"
model.save(keras_model_path)
