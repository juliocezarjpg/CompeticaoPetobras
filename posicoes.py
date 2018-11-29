import cv2
import numpy as np
import re

canvas = np.ones((3500, 3500, 3)) * 255
azul = (255, 0, 0)

file  = open('posicoes.txt', 'r')
str = file.read()
file.close()


while (not(re.search('99',str, re.IGNORECASE))):
    file  = open('posicoes.txt', 'r')
    str = file.read()
    file.close()
    #print re.search('99',str, re.IGNORECASE)

file  = open('posicoes.txt', 'r')
str = file.readline()
str = str.split()
posa = int(str[0])
posb = int(str[1])
    
while (int(str[0]) != 99):
    
    str = file.readline()
    str = str.split()

    try:
        if (int(str[0]) != 99):
            cv2.line(canvas, (posa, posb), (int(str[0]), int(str[1])), azul, 9)
            print (posa, posb, int(str[0]), int(str[1]))
        posa = int(str[0])
        posb = int(str[1])
    except:
        print ('deu barrado')

#cv2.namedWindow("Canvas",  cv2.WINDOW_NORMAL)

file.close()
canvas = cv2.flip(canvas, 1)
M = cv2.getRotationMatrix2D((canvas.shape[1]/2, canvas.shape[0]/2), 270, 1)
rotated = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]))
#cv2.imshow('mapa', canvas)
cv2.imwrite("mapa_pista.jpg", rotated)
#cv2.imshow("Canvas", canvas)
#cv2.waitKey(0)


