import os
import predict as proceso
import cv2

def menu():
    os.system('cls')  # NOTA para windows tienes que cambiar clear por cls
    print("Selecciona una opción")
    print("\t1 - Multiples imagenes")
    print("\t2 - Predicción")
    print("\t3 - salir")


while True:
    # Mostramos el menu
    menu()
    opcionMenu = input("inserta un numero valor :: ")
    if opcionMenu == "1":
        print("Multiples Imagenes....\n")
        # proceso.recortar(path='C:/Users/ACER/Desktop/Pruebas/avotry.jpg', directorio='C:/Users/ACER/Desktop/Probar')
        save_img = input("Ruta Imagen....\n")
        proceso.recorte("C:\\Users\\Lopez\\Documents\\GitHub\\IA\\imagen",save_img)
        contenido = os.listdir('C:\\Users\\Lopez\\Documents\\GitHub\\IA\\imagen')
        for i in range(len(contenido)):
         print("C:\\Users\\Lopez\\Documents\\GitHub\\IA\\imagen\\img" + str(i+1) + '.PNG')
         imga="C:\\Users\\Lopez\\Documents\\GitHub\\IA\\imagen\\img" + str(i+1) + '.PNG'
         print("--------------------------------------------")
         print("\n  Imagen multiple   NO. "+ str(i+1) +"  \n")
         print("--------------------------------------------")
         img_path = imga
         save_path = input("Ruta para almacenar imágen recortada....\n")
         labels_path = input("Carpeta que contiene las carpetas de imágenes....\n")
         model = input("Modelo a utilizar ubicado en....\n")
         proceso.procesoreconocer(labels_path, model, img_path, save_path)

    elif opcionMenu == "2":
        print("")
        print("Predecir imágen....\n")
        img_path = input("Ruta de imágen....\n")
        save_path = input("Ruta para almacenar imágen recortada....\n")
        labels_path = input("Carpeta que contiene las carpetas de imágenes....\n")
        model = input("Modelo a utilizar ubicado en....\n")
        proceso.procesoreconocer(labels_path, model, img_path, save_path)
    elif opcionMenu == "3":
        break
    else:
        print("")
        input("No has pulsado ninguna opción correcta...\npulsa una tecla para continuar")






