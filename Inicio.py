import os
import cv2

# FUNCIONES----------------------------
def show_image(text, img):
    cv2.imshow(text, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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

def recorte(carguardar, imgaleer):
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
        i += 1
    return i-1
# FUNCIONES----------------------------
def menu():
    os.system('cls')  # NOTA para windows tienes que cambiar clear por cls
    print("Selecciona una opción")
    print("\t1 - Multiples imagenes")
    print("\t2 - Una imagen")
    print("\t3 - Reducir imagen")
    print("\t4 - salir")


while True:
    # Mostramos el menu
    menu()
    opcionMenu = input("inserta un numero valor :: ")
    if opcionMenu == "1":
        print("")
        input("Multiples imagenes...\n")
        x=recorte("C:\\Users\\Lopez\\Documents\\GitHub\\IA\\guardar",tomarfoto())
        print("Realizado con extito ",x)
        print("\n\n\n")
    elif opcionMenu == "2":
        print("")
        input("Una imagen....\n")
    elif opcionMenu == "3":
        print("")
        input("Has pulsado la opción 3...\npulsa una tecla para continuar")
    elif opcionMenu == "4":
        break
    else:
        print("")
        input("No has pulsado ninguna opción correcta...\npulsa una tecla para continuar")






