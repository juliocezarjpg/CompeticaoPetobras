import cv2
import numpy as np

def nothing(x):
    pass

calibrar = 1

imagem = 'equipamento2_0g.jpeg'

frame = cv2.imread(imagem)
im = cv2.imread(imagem)

a_read = open ('valuesg.txt', 'r')
texto = a_read.readline()
a_read.close()

n1 = ''
values = []
for i in range(len(texto)):
    if texto[i] != ',':
        n1 += texto[i]
    else:
        n1 = int(n1)
        values.append(n1)
        n1 = ''

lower_collor = np.array([values[0], values[2] , values[1]], dtype = 'uint8')
upper_collor = np.array([values[3], values[5], values[4]], dtype = 'uint8')


if calibrar == 1:
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

if calibrar == 1:    
    cv2.createTrackbar('LowHue','image',values[0],255,nothing)
    cv2.createTrackbar('LowSat','image',values[1],255,nothing)
    cv2.createTrackbar('LowValue','image',values[2],255,nothing)
    cv2.createTrackbar('HighHue','image',values[3],255,nothing)
    cv2.createTrackbar('HighSat','image',values[4],255,nothing)
    cv2.createTrackbar('HighValue','image',values[5],255,nothing)

while 1:

    if calibrar == 1:
            LowHue = cv2.getTrackbarPos('LowHue','image')
            LowSat = cv2.getTrackbarPos('LowSat','image')
            LowValue = cv2.getTrackbarPos('LowValue','image')
            HighHue = cv2.getTrackbarPos('HighHue','image')
            HighSat = cv2.getTrackbarPos('HighSat','image')
            HighValue = cv2.getTrackbarPos('HighValue','image')

            lower_collor = np.array([LowHue, LowValue, LowSat], dtype = 'uint8')
            upper_collor = np.array([HighHue, HighValue, HighSat], dtype = 'uint8')


    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    threshold = cv2.inRange(hsv, lower_collor, upper_collor)

    #Cria kernel para usar na erosao e dilatacao
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

    #remove os ruídos da imagem usando a erosao
    threshold = cv2.erode (threshold, kernel, iterations = 5)

    #agrupa a imagem usando dilatacao
    threshold = cv2.dilate(threshold, kernel, iterations = 5)

    (_, cnts, _) = cv2.findContours(threshold.copy(), mode = cv2.RETR_EXTERNAL,
                                    method = cv2.CHAIN_APPROX_SIMPLE)
    (_,contours, _) = cv2.findContours(threshold.copy(),cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(im,cnts,-1,(0, 150, 255),2)

    cv2.imshow('Threshold', threshold)
    cv2.imshow('Painel', im)
    im = cv2.imread(imagem)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

if calibrar == 1:
    values = ''
    values += (str(LowHue))
    values += (',')
    values += (str(LowSat))
    values += (',')
    values += (str(LowValue))
    values += (',')
    values += (str(HighHue))
    values += (',')
    values += (str(HighSat))
    values += (',')
    values += (str(HighValue))
    values += (',')

    a_write = open ('valuesg.txt', 'w')
    a_write.write(values)
    a_write.close()

