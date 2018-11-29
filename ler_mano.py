import numpy as np
import cv2
import math

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

def processarManometro():#entrada):
    
    img = cv2.imread('mano.jpeg')
    im = cv2.imread('mano.jpeg')
    #img = entrada
    #im = entrada
    
    masked_data = detectarManometro(img, im)
    agulha = detectarAgulhaManometro(masked_data)

    print agulha

    cv2.drawContours(img, [agulha], -11, (255, 0, 0), 2)

    pt1 = (agulha[2][0]+agulha[3][0])/2,(agulha[2][1]+agulha[3][1])/2
    pt2 = (agulha[0][0]+agulha[1][0])/2,(agulha[0][1]+agulha[1][1])/2
    
    #print pt1
    #print pt2
    
    deltax=pt1[0]-pt2[0]
    deltay=pt1[1]-pt2[1]
    
    #print math.atan2(deltay,deltax)

    cv2.line(img, pt1,pt2,(0,0,255),1)

    angulo=math.degrees(math.atan2(deltax,deltay))

    #print "Angulo: ",angulo
    
    #cv2.imshow('Manometro', img)
    #cv2.waitKey(0)
    return angulo

def detectarManometro(img, im):
    img = cv2.medianBlur(img,5)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=0,maxRadius=0)

    circles = np.uint16(np.around(circles))
    circle=sorted(circles[0],key=lambda x:x[2],reverse=1)[0] #ordenar por raio e pegar o maior

    height,width,depth = im.shape
    circle_img = np.zeros((height,width), np.uint8)
    #cv2.circle(circle_img,(width/2,height/2),280,1,thickness=-1)
    #cv2.circle(circle_img,(circle[0],circle[1]),circle[2]-40,1,-1)

    masked_data = cv2.bitwise_and(im, im, mask=circle_img)

    return masked_data

def detectarAgulhaManometro(frame):
    
    texto = lerArquivo('values_manometro.txt')
    values = lerValores(texto)

    lower_collor = np.array([values[0], values[2] , values[1]], dtype = 'uint8')
    upper_collor = np.array([values[3], values[5], values[4]], dtype = 'uint8')
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    threshold = cv2.inRange(hsv, lower_collor, upper_collor)

    #Cria kernel para usar na erosao e dilatacao
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    #remove os ruídos da imagem usando a erosao
    threshold = cv2.erode (threshold, kernel, iterations = 1)

    #agrupa a imagem usando dilatacao
    threshold = cv2.dilate(threshold, kernel, iterations = 7)

    (_, cnts, _) = cv2.findContours(threshold.copy(), mode = cv2.RETR_TREE,
                                    method = cv2.CHAIN_APPROX_SIMPLE)

    cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

    #cv2.drawContours(frame,cnts,-1,(0, 0, 255),2)
    #cv2.drawContours(frame,cnt,-1,(0, 255, 0),2)

    rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))

    #cv2.imshow('threshold', frame)

    return rect
