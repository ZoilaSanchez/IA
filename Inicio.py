import os
import predict as proceso

def menu():
    os.system('cls')  # NOTA para windows tienes que cambiar clear por cls
    print("Selecciona una opción")
    print("\t1 - Multiples imagenes")
    print("\t2 - Una imagen")
    print("\t3 - salir")


while True:
    # Mostramos el menu
    menu()
    opcionMenu = input("inserta un numero valor :: ")
    if opcionMenu == "1":
        print("")
        input("Multiples imagenes...\n")
       # proceso.recorte(carguardar=,imgaleer=,cargardatos=,cargarmodelo=,imgaen2=)
    elif opcionMenu == "2":
        print("")
        input("Una imagen....\n")
        #proceso.recortar(path=,directorio=)
    elif opcionMenu == "3":
        break
    else:
        print("")
        input("No has pulsado ninguna opción correcta...\npulsa una tecla para continuar")






