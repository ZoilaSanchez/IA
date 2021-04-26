import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image

# camara
captura=cv2.VideoCapture(0)

while(True):
    #Caputramos la foto
    return_Varlor,image=captura.read();
    #visualizar
    if return_Varlor:
        cv2.imshow('visor',image)
    #salir
    if cv2.waitKey(1)&0xFF==ord('q'):
        cv2.imwrite('opencv1.png',image)
        del(captura)
        break

cv2.destroyAllWindows()

imagen=Image.open('opencv1.png')
plt.imshow(imagen)
plt.show()








