import cv2
import numpy as np

#arquivo = "mano.jpeg"

#camera = cv2.VideoCapture(0)

def ler(arquivo):
    frame = cv2.imread(arquivo)

    #cv2.imshow('Valvula',frame)
    #print frame
    saida = 'None'

    #cv2.imshow('frame',frame)
    #cv2.waitKey(0)
    
    aberto = cv2.imread("v_aberta.jpeg",0)
    fechado = cv2.imread("v_fecha.jpeg",0)
    wa, ha = aberto.shape[::-1]
    wf, hf = aberto.shape[::-1]

#    contim=0

    
    #img = cv2.imread(arquivo,0)
    #ret, frame = camera.read()

    #cv2.imshow("cam",frame)

    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    #cv2.imshow("Aberto", aberto)
    #cv2.imshow("Fechado",fechado)

    isopen = cv2.matchTemplate(img, aberto, cv2.TM_CCOEFF_NORMED)
    isclose = cv2.matchTemplate(img, fechado, cv2.TM_CCOEFF_NORMED)

    #print isopen, isclose
    threshold = 0.8
    loca = np.where( isopen >= threshold)
    locf = np.where( isclose >= threshold)

    if zip(*loca[::-1])!=[]:
        saida = 'Aberta'
        #print "achou A:"
        #for pt in zip(*loca[::-1]):
        #    cv2.rectangle(frame, pt, (pt[0] + wa, pt[1] + ha), (255,0,0), 2)
    if zip(*locf[::-1])!=[]:
        saida = 'Fechada'
        #print "acho F:"
        #for pt in zip(*locf[::-1]):
        #    cv2.rectangle(frame, pt, (pt[0] + wf, pt[1] + hf), (0,255,0), 2)

    return (saida)
    #cv2.imshow("cam",frame)

    #cv2.waitKey(0)
     
    
#cv2.destroyAllWindows()
#camera.release()

