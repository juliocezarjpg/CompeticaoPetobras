#Ordenacao: https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
import cv2
import numpy as np

def nothing(x):
    pass

frame = cv2.imread('painel1.jpg')
im = cv2.imread('painel1.jpg')

def lerArquivo (arquivo):
    a_read = open (arquivo, 'r')
    texto = a_read.readline()
    a_read.close()
    return texto

def lerValores (texto):
    n1 = ''
    values = []
    for i in range(len(texto)):
        if texto[i] != ',':
            n1 += texto[i]
        else:
            n1 = int(n1)
            values.append(n1)
            n1 = ''
    return values

def putThreshold (lower, upper):
    threshold = cv2.inRange(hsv, lower, upper)
    #Cria kernel para usar na erosao e dilatacao
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    #remove os ruídos da imagem usando a erosao
    threshold = cv2.erode (threshold, kernel, iterations = 5)
    #agrupa a imagem usando dilatacao
    threshold = cv2.dilate(threshold, kernel, iterations = 5)

    return threshold

def calcularMeio(cnts):
    meio = 0;
    for i in cnts:
        moments = cv2.moments(i)
        if moments['m00'] != 0:
            meio += (int(moments['m01']/moments['m00']))
    return int((meio/3))

def getOnOff(cnts, meio):
    for c in cnts:
        moments = cv2.moments(c)
        if moments['m00'] != 0:
            cy = (int(moments['m01']/moments['m00']))
            if (cy < meio):
                print ('desligado')
            else:
                print ('ligado')

def sort_contours(cnts, method="left-to-right"):
	reverse = False
	i = 0
 
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	return cnts

def desenharContornos (im, cnts):
    for c in cnts:
        (x,y),radius = cv2.minEnclosingCircle(c)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(im,center,radius,(255,0,0),2)
            
if __name__ == '__main__':
    texto = lerArquivo('values.txt')
    values = lerValores(texto)

    lower_collor = np.array([values[0], values[2] , values[1]], dtype = 'uint8')
    upper_collor = np.array([values[3], values[5], values[4]], dtype = 'uint8')


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    threshold = putThreshold (lower_collor, upper_collor)
    #Encontrar os contornos
    (_, cnts, _) = cv2.findContours(threshold.copy(),
                                    mode = cv2.RETR_TREE,
                                    method = cv2.CHAIN_APPROX_SIMPLE)
    #ordenar da esquerda para a direita
    cnts = sort_contours(cnts)

    #Caclular meio da imagem
    meio = calcularMeio(cnts)
    #Detectar os que estao ligados e desligados
    getOnOff(cnts, meio)   
    #Desenhar os contornos
    desenharContornos(im, cnts)
    #Desenhar linha do meio
    cv2.line(im, (0, meio), (im.shape[1], meio), (255, 0, 0), 2)
  

    cv2.imshow('Painel', im)
    im = cv2.imread('painel1.jpg')

    
