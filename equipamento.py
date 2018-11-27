#Ordenacao: https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
import cv2
import numpy as np

def nothing(x):
    pass

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

def putThreshold (lower, upper, hsv):
    threshold = cv2.inRange(hsv, lower, upper)
    #Cria kernel para usar na erosao e dilatacao
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    #remove os ruídos da imagem usando a erosao
    threshold = cv2.erode (threshold, kernel, iterations = 5)
    #agrupa a imagem usando dilatacao
    threshold = cv2.dilate(threshold, kernel, iterations = 5)

    return threshold

def calcularMeio(cnts, im, frame):
    cima = 0
    baixo = im.shape[0]

    texto = lerArquivo('valuesr.txt')
    values = lerValores(texto)

    lower_collor = np.array([values[0], values[2] , values[1]], dtype = 'uint8')
    upper_collor = np.array([values[3], values[5], values[4]], dtype = 'uint8')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    threshold = putThreshold (lower_collor, upper_collor, hsv)
    (_, cnts, _) = cv2.findContours(threshold.copy(),
                                    mode = cv2.RETR_TREE,
                                    method = cv2.CHAIN_APPROX_SIMPLE)
    if (len(cnts) > 0):
        cnts = sort_contours(cnts)
        moments = cv2.moments(cnts[0])
        cima = (int(moments['m01']/moments['m00']))

    texto = lerArquivo('valuesg.txt')
    values = lerValores(texto)

    lower_collor = np.array([values[0], values[2] , values[1]], dtype = 'uint8')
    upper_collor = np.array([values[3], values[5], values[4]], dtype = 'uint8')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    threshold = putThreshold (lower_collor, upper_collor, hsv)
    (_, cnts, _) = cv2.findContours(threshold.copy(),
                                    mode = cv2.RETR_TREE,
                                    method = cv2.CHAIN_APPROX_SIMPLE)
    if (len(cnts) > 0):
        cnts = sort_contours(cnts)
        moments = cv2.moments(cnts[0])
        baixo = (int(moments['m01']/moments['m00']))

    return int( (cima + baixo)/2 )

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

def procesarPainel():

    frame = cv2.imread('equipamento_2.jpg')
    im = cv2.imread('equipamento_2.jpg')

    texto = lerArquivo('values.txt')
    values = lerValores(texto)

    lower_collor = np.array([values[0], values[2] , values[1]], dtype = 'uint8')
    upper_collor = np.array([values[3], values[5], values[4]], dtype = 'uint8')


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    threshold = putThreshold (lower_collor, upper_collor, hsv)
    #Encontrar os contornos
    (_, cnts, _) = cv2.findContours(threshold.copy(),
                                    mode = cv2.RETR_TREE,
                                    method = cv2.CHAIN_APPROX_SIMPLE)
    #ordenar da esquerda para a direita
    cnts = sort_contours(cnts)

    #Caclular meio da imagem
    meio = calcularMeio(cnts, im, frame)
    #Detectar os que estao ligados e desligados
    getOnOff(cnts, meio)   
    #Desenhar os contornos
    desenharContornos(im, cnts)
    #Desenhar linha do meio
    cv2.line(im, (0, meio), (im.shape[1], meio), (255, 0, 0), 2)
  

    cv2.imshow('Painel', im)
    im = cv2.imread('painel1.jpg')

def detectarAgulhaManometro(frame):
    
    texto = lerArquivo('values_manometro.txt')
    values = lerValores(texto)

    lower_collor = np.array([values[0], values[2] , values[1]], dtype = 'uint8')
    upper_collor = np.array([values[3], values[5], values[4]], dtype = 'uint8')
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    threshold = cv2.inRange(hsv, lower_collor, upper_collor)

    #Cria kernel para usar na erosao e dilatacao
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    #remove os ruídos da imagem usando a erosao
    threshold = cv2.erode (threshold, kernel, iterations = 1)

    #agrupa a imagem usando dilatacao
    threshold = cv2.dilate(threshold, kernel, iterations = 6)

    (_, cnts, _) = cv2.findContours(threshold.copy(), mode = cv2.RETR_EXTERNAL,
                                    method = cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('threshold', threshold)


def processarManometro():
    
    img = cv2.imread('equipamento_1.jpg')
    im = cv2.imread('equipamento_1.jpg')

    img = cv2.medianBlur(img,5)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

    circles = np.uint16(np.around(circles))
    circle=sorted(circles[0],key=lambda x:x[2],reverse=1)[0] #ordenar por raio e pegar o maior

    height,width,depth = im.shape
    circle_img = np.zeros((height,width), np.uint8)
    #cv2.circle(circle_img,(width/2,height/2),280,1,thickness=-1)
    cv2.circle(circle_img,(circle[0],circle[1]),circle[2]-40,1,-1)

    masked_data = cv2.bitwise_and(im, im, mask=circle_img)
    #masked_data = cv2.cvtColor(masked_data,cv2.COLOR_BGR2GRAY)

    detectarAgulhaManometro(masked_data)
    
    cv2.imshow('Manometro', masked_data)
    
if __name__ == '__main__':
    processarManometro()    

    
