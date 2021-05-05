import os
import predict as proceso

def menu():
    os.system('cls')  # NOTA para windows tienes que cambiar clear por cls
    print("Selecciona una opción")
    print("\t1 - Multiples imagenes")
    print("\t2 - Una imagen")
    print("\t3 - Predicción")
    print("\t4 - salir")


while True:
    # Mostramos el menu
    menu()
    opcionMenu = input("inserta un numero valor :: ")
    if opcionMenu == "1":
        print("")
        input("Multiples imagenes...\n")
        # proceso.recorte(carguardar='C:/Users/ACER/Desktop/outcome', imgaleer='C:/Users/ACER/Documents/GitHub/IA/MIXTAS/146d24f0-d5fb-4464-9400-1079031b2636.PNG' ,cargardatos=,cargarmodelo=,imgaen2=)
    elif opcionMenu == "2":
        print("")
        input("Una imagen....\n")
        # proceso.recortar(path='C:/Users/ACER/Desktop/Pruebas/avotry.jpg', directorio='C:/Users/ACER/Desktop/Probar')
    elif opcionMenu == "3":
        print("")
        print("Predecir imágen....\n")
        img_path = input("Ruta de imágen....\n")
        save_path = input("Ruta para almacenar imágen recortada....\n")
        labels_path = input("Carpeta que contiene las carpetas de imágenes....\n")
        model = input("Modelo a utilizar ubicado en....\n")
        proceso.procesoreconocer(labels_path, model, img_path, save_path)
    elif opcionMenu == "4":
        break
    else:
        print("")
        input("No has pulsado ninguna opción correcta...\npulsa una tecla para continuar")






