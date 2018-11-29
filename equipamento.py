#Ordenacao: https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
from __future__ import print_function
import pyzbar.pyzbar as pyzbar
import cv2
import numpy as np
import re
import math

resultados = open ('resultados.txt', 'w')

canvas = np.ones((3000, 3000, 3)) * 255
azul = (255, 0, 0)

file  = open('posicoes.txt', 'r')
string = file.read()
file.close()

qr = []
painel = []
painel2 = []
valvula = []
manometro = []

def decode(im) : 
  # Find barcodes and QR codes
  decodedObjects = pyzbar.decode(im)
 
  # Print results
  for obj in decodedObjects:
    #print('Type : ', obj.type)
    #print('Data : ', obj.data,'\n')
    return (str(obj.data))
 
 
# Display barcode and QR code location  
def display(im, decodedObjects):
 
  # Loop over all decoded objects
  for decodedObject in decodedObjects: 
    points = decodedObject.polygon
 
    # If the points do not form a quad, find convex hull
    if len(points) > 4 : 
      hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
      hull = list(map(tuple, np.squeeze(hull)))
    else : 
      hull = points;
     
    # Number of points in the convex hull
    n = len(hull)
 
    # Draw the convext hull
    for j in range(0,n):
      cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)
 
  # Display results 
  cv2.imshow("Results", im);
  cv2.waitKey(0);

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
    qt = 0;
    for c in cnts:
        qt += 1
        moments = cv2.moments(c)
        if moments['m00'] != 0:
            cy = (int(moments['m01']/moments['m00']))
            if (cy < meio):
                #resultados.write('\tCircuito ' + str(qt) + ': desligado\n')
                painel.append('D')
                #print ('desligado')
            else:
                #resultados.write('\tCircuito ' + str(qt) + ': ligado\n')
                painel.append('L')
                #print ('ligado')

def sort_contours(cnts, method="left-to-right"):
	reverse = False
	i = 0
 
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	return cnts

def desenharContornos (im, cnts, color = (255, 0, 0)):
    for c in cnts:
        (x,y),radius = cv2.minEnclosingCircle(c)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(im,center,radius,color, 2)

def getOnOff2(meio, img):

  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  texto = lerArquivo('valuesr.txt')
  values = lerValores(texto)

  lower_collor = np.array([values[0], values[2] , values[1]], dtype = 'uint8')
  upper_collor = np.array([values[3], values[5], values[4]], dtype = 'uint8')

  threshold = putThreshold (lower_collor, upper_collor, hsv.copy())
  #Encontrar os contornos
  (_, cntsr, _) = cv2.findContours(threshold.copy(),
                                  mode = cv2.RETR_TREE,
                                  method = cv2.CHAIN_APPROX_SIMPLE)

  texto = lerArquivo('valuesg.txt')
  values = lerValores(texto)

  lower_collor = np.array([values[0], values[2] , values[1]], dtype = 'uint8')
  upper_collor = np.array([values[3], values[5], values[4]], dtype = 'uint8')

  threshold = putThreshold (lower_collor, upper_collor, hsv.copy())
  #Encontrar os contornos
  (_, cntsg, _) = cv2.findContours(threshold.copy(),
                                  mode = cv2.RETR_TREE,
                                  method = cv2.CHAIN_APPROX_SIMPLE)

  desenharContornos(img, cntsr, (0, 255, 0))
  desenharContornos(img, cntsg, (0, 255, 0))

  cntsr = sort_contours(cntsr)
  cntsg = sort_contours(cntsg)

  #cv2.imshow('OnOff2', img)

  #Lista com posicao de todos os objetos (desligados)
  leituras = []
  for c in cntsg:
    moments = cv2.moments(c)
    if moments['m00'] != 0:
      pacote = []
      cx = int(moments['m10']/moments['m00'])
      cy = (int(moments['m01']/moments['m00']))   
      pacote.append(cx)
      pacote.append(cy)
      pacote.append('g')
      leituras.append(pacote)

  for c in cntsr:
    moments = cv2.moments(c)
    if moments['m00'] != 0:
      pacote = []
      cx = int(moments['m10']/moments['m00'])
      cy = (int(moments['m01']/moments['m00']))   
      pacote.append(cx)
      pacote.append(cy)
      pacote.append('r')
      leituras.append(pacote)

  #sort (I am shamed of this implementation)
  if (leituras[0][0] > leituras[1][0]):
    temp = leituras[0]
    leituras[0] = leituras[1]
    leituras[1] = temp
  if (leituras[1][0] > leituras[2][0]):
    temp = leituras[1]
    leituras[1] = leituras[2]
    leituras[2] = temp
  if (leituras[0][0] > leituras[1][0]):
    temp = leituras[0]
    leituras[0] = leituras[1]
    leituras[1] = temp

  #comparacao entre a posicao do objeto e o meio entre os botoes
  for l in leituras:
    if (l[1] > meio):
      painel2.append('D')
    else:
      painel2.append('L')
  

  print (meio)  
  print (leituras)  
  


def procesarPainel(imagem):

    frame = cv2.imread(imagem)
    im = cv2.imread(imagem)

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
    #getOnOff2(meio, im)
    #Desenhar os contornos
    desenharContornos(im, cnts)
    #Desenhar linha do meio
    cv2.line(im, (0, meio), (im.shape[1], meio), (255, 0, 0), 2)
  

    cv2.imshow('Painel', im)
    im = cv2.imread(imagem)


def detectarManometro(img, im):
    img = cv2.medianBlur(img,5)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,
                            param1=150,param2=30,minRadius=0,maxRadius=0)

    circles = np.uint16(np.around(circles))


    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('circulo', img)
    
    circle=sorted(circles[0],key=lambda x:x[2],reverse=1)[0] #ordenar por raio e pegar o
    
    height,width,depth = im.shape
    circle_img = np.zeros((height,width), np.uint8)
    #cv2.circle(circle_img,(width/2,height/2),280,1,thickness=-1)
    cv2.circle(circle_img,(circle[0],circle[1]),circle[2]-15,1,-1)

    masked_data = cv2.bitwise_and(im, im, mask=circle_img)

    cv2.imshow('circulo_com_mascara', masked_data)
    #cv2.imwrite('teste.jpg', masked_data)

    return masked_data

def trata(ang):
    angulo=[]
    #print ("Sinal de enrada:"),ang
    if (ang<=-45 or ang >= 45):
        lei=0
        if (ang<-45):
            angulo.insert(0, ang + 180.0)
        else:
            angulo.insert(0, ang)
    else:
        lei=1
        if (ang<0):
            angulo.insert(0,ang + 180.0)#[0] = ang+180.0
            angulo.insert(1,ang)#[1] = ang
        else:
            angulo.insert(1,ang + 180.0) #= ang+180.0
            angulo.insert(0,ang )#[0] = ang
            


    #print ("Leituras:"),lei,(" Angulos:"),angulo

    return (lei,angulo)

def converte(ang):
    real = ang
    manometro = 225 - real

    pressao = (14.0*(manometro-270))/270.0 + 14.0

    return(manometro, pressao)

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

    rows,cols = frame.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
    #print vx,vy,x,y
    lefty = ((-x*vy/vx) + y)
    righty = (((cols-x)*vy/vx)+y)
    #print ("cols"),cols
    cv2.line(frame,(cols-1,righty),(0,lefty),(0,255,0),1)

    #Calculo para deslocar a linha para passar no ponto (0,0) e pegar a angulacao dessa linha
    eixoy = -1.0*(righty-lefty)
    eixox = cols-1.0
    angulo=math.degrees(math.atan2(eixoy,eixox))

    ang_tratado = trata(angulo)

    print (ang_tratado)
  
    if (ang_tratado[0]==0):
      ang_considerado = ang_tratado[1][0]
    else:
      ang_considerado = ang_tratado[1][0] #0esq
    #print ("Angulo considerado:"), ang_considerado
    res = converte(ang_considerado)
    print ('O angulo e:' + str(res[0]))
    print ('A leitura e:' + str(res[1]) + 'bar')

    manometro.append(res[0])
    manometro.append(res[1])

    cv2.imshow('threshold', frame)

    #return rect
    return cv2.minAreaRect(cnt)

def processarManometro(imagem):
    
    img = cv2.imread(imagem)
    im = cv2.imread(imagem)
    
    masked_data = detectarManometro(img, im)
    agulha = detectarAgulhaManometro(masked_data)
    rect = np.int32(cv2.boxPoints(agulha))
    #print (int(agulha[2])*-1)
    print (agulha[2])
    #manometro.append(agulha[2])
    #resultados.write('\t Angulo da agulha: ' + str(int(agulha[2])*-1) + ' graus\n')

    cv2.drawContours(img, [rect], -11, (255, 0, 0), 2)
    
    cv2.imshow('Manometro', img)

def processarValvula(imagem):
  
  img = cv2.imread(imagem)
  im = cv2.imread(imagem)


  #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  
  #v_aberta = cv2.imread('v_aberta.jpeg',0)
  #wa, ha = v_aberta.shape[::-1]

  #v_fechada = cv2.imread('v_fechada.jpeg',0)
  #wf, hf = v_fechada.shape[::-1]

  #isopen = cv2.matchTemplate(img, v_aberta, cv2.TM_CCOEFF_NORMED)
  #isclose = cv2.matchTemplate(img, v_fechada, cv2.TM_CCOEFF_NORMED)
    
  #threshold = 0.6
  
  #loca = np.where( isopen >= threshold)
  #locf = np.where( isclose >= threshold)

  #if zip(*loca[::-1])!=[]:
  #  saida = 'Aberta'
  #  print ("achou A:")
  #  for pt in zip(*loca[::-1]):
  #    print ('Tudo bem?')
  #    cv2.rectangle(im, pt, (pt[0] + wa, pt[1] + ha), (255,0,0), 2)

  #if zip(*locf[::-1])!=[]:
  #  saida = 'Fechada'
  #  print ("acho F:") 
  #  for pt in zip(*locf[::-1]):
  #    print ('Opa')
  #    cv2.rectangle(im, pt, (pt[0] + wf, pt[1] + hf), (0,255,0), 2)

  #cv2.imshow('valvula', im)

    
  texto = lerArquivo('values_v.txt')
  values = lerValores(texto)

  lower_collor = np.array([values[0], values[2] , values[1]], dtype = 'uint8')
  upper_collor = np.array([values[3], values[5], values[4]], dtype = 'uint8')
    
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  threshold = cv2.inRange(hsv, lower_collor, upper_collor)

    #Cria kernel para usar na erosao e dilatacao
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

    #remove os ruídos da imagem usando a erosao
  threshold = cv2.erode (threshold, kernel, iterations = 5)

   #agrupa a imagem usando dilatacao
  threshold = cv2.dilate(threshold, kernel, iterations = 5)

  (_, cnts, _) = cv2.findContours(threshold.copy(), mode = cv2.RETR_TREE,
                                    method = cv2.CHAIN_APPROX_SIMPLE)

  cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

  cv2.drawContours(img,cnts,-1,(0, 0, 255),2)

  rect = np.int32(cv2.boxPoints(cv2.minAreaRect(cnt)))

  cv2.drawContours(img, [rect], -11, (255, 0, 0), 2)

  cv2.imshow('valvula', img)
  
  print(cv2.minAreaRect(cnt)[1][0], cv2.minAreaRect(cnt)[1][1])
  
  if (cv2.minAreaRect(cnt)[1][0] > cv2.minAreaRect(cnt)[1][1]):
    #resultados.write('\tValvula Fechada\n')
    valvula.append('F')
    print ('fechado')
  else:
  #  resultados.write('\tValvula Aberta\n')
    valvula.append('A')
    print ('aberto')

def processarQrCode(nome):
    im = cv2.imread(nome)
    decodedObjects = decode(im)
    return decodedObjects 

def lerPosicoes():
  pos = []
  file  = open('posicoes.txt', 'r')
  string = file.readline()
  string = string.split()
  posa = int(string[0])
  posb = int(string[1])
      
  while (int(string[0]) != 99):
      pos.append(posa)
      pos.append(posb)
      string = file.readline()
      string = string.split()

      try:
          if (int(string[0]) != 99):
              cv2.line(canvas, (posa, posb), (int(string[0]), int(string[1])), azul, 9)
              #print (posa, posb, int(string[0]), int(string[1]))
          posa = int(string[0])
          posb = int(string[1])
      except:
          print ('arquivo finalizado')
  file.close()

  return pos


if __name__ == '__main__':
    qt= 1
    while (not(re.search('99',string, re.IGNORECASE))):
      file  = open('posicoes.txt', 'r')
      string = file.read()
      file.close()

    pos = lerPosicoes()
    while 1:
        try:
            nome = 'qrcode'
            nome += str(qt)
            nome += '_0.jpeg'

            dado = processarQrCode(nome)
            #print (dado)
            qr.append(dado)
            #print (dado[2])
            if (dado[2] == 'P'):
                try:
                  procesarPainel('equipamento' + str(qt) + '_0.jpeg')
                except:
                  print ('Painel nao deu')
            elif (dado[2] == 'V'):
                try:
                  processarValvula('equipamento' + str(qt) + '_0.jpeg')
                except:
                  print ('Valvula nao deu')
            elif (dado[2] == 'M'):
                try:
                  processarManometro('equipamento' + str(qt) + '_0.jpeg')
                except:
                  print ('Manometro nao deu')
            qt = qt + 1
            
        except:
           print (qr, painel, valvula, manometro)
           for i in qr:
             resultados.write('QR: ' + i[2] + i[3] + i[4] + '\n')

           if (len(painel) > 0):
             resultados.write('\nP: ' + painel[0] + painel[1] + painel[2])
           if (len(valvula) > 0):
            resultados.write('\n\nV: ' + valvula[0])
           if (len(manometro) > 0):
             resultados.write('\n\nM: ' +
                            str(manometro[0]) + ' graus\n' + 'M: '+
                            str(manometro[1]) + ' bar')
             
           resultados.write('\n\nPosicao dos nichos (x y):\n\n'
                            + 'Nicho 1: (' + str(pos[1]) + ' ' + str(pos[0]) + ')\n'
                            + 'Nicho 2: (' + str(pos[3]) + ' ' + str(pos[2]) + ')\n'
                            + 'Nicho 3: (' + str(pos[5]) + ' ' + str(pos[4]) + ')\n')
           resultados.close()
           print ('acabou')
           break
