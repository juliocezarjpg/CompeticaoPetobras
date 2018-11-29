import time
import cv2
import numpy as np
#Modulos desenvolvidos
import ler_qr #ler qrCode
import ler_mano #Ler Manometro
import ler_valvula
import ler_painel
import ang2lei
import trata_angulo
#abrindo a camera
#camera = cv2.VideoCapture(0)

def escreve_arq(text):
    esc_arq = open ('relatorio.txt','a')
    esc_arq.write(text+"\n")
    esc_arq.close()
    

while (True):
    escreve_arq('//Relatorio EQUIPE JAGUAR')
    escreve_arq('//Desafio Petrobras de Robotica - 2018')
    escreve_arq(' ')
    #Agurada sinal robino
    print "Aguardando"
    conteudo = "False"
    while (conteudo == "False"):
        ler_arquivo = open("geral.txt", 'r')
        conteudo = ler_arquivo.readline()
        ler_arquivo.close()
        
        #time.sleep(1)
    QR=[]
    escreve_arq('//Leituras dos QR codes')
    for i in range(1,4):
        #print i
        im = cv2.imread("QR-"+str(i)+".jpeg",0)
        leitura = ler_qr.ler(im)
        QR.insert(i-1,leitura)
        #print leitura
        escreve_arq('QR: ' + leitura)
    print QR

    i=0
    for i in range(0,3):
        #im = cv2.imread("EQ"+str(i+1)+".jpeg")
        if (QR[i][0]=="M"): #indica que leu manomtro
            print "manometro"
            ang_real = ler_mano.processarManometro("EQ-"+str(i+1)+".jpeg")
            #print ("Angulo real:"), ang_real
            ang_tratado = trata_angulo.trata(ang_real)
            #print ("Angulo tratado 'e:"),ang_tratado
            if (ang_tratado[0]==0):
                ang_considerado = ang_tratado[1][0]
            else:
                ang_considerado = ang_tratado[1][0] #0esq
            #print ("Angulo considerado:"), ang_considerado
            res = ang2lei.converte(ang_considerado)
            print ("O angulo e:"),res[0]
            print ("A leitura e:"),res[1],("bar")
        if (QR[i][0]=="V"): #indica que leu valvula
            print "Valvula",i
            resultado = ler_valvula.ler("EQ-"+str(i+1)+".jpeg")
            print (resultado)
            #inflida = ler_valula.ler(img)
        if (QR[i][0]=="P"):
            print "Painel"
            resultado = ler_painel.ler("EQ-"+str(i+1)+".jpeg")
            print (resultado)

    print "Acabou"
    #time.sleep(5)
    break
        
'''    

    inflida = 'None'
    #toma a acao em funcao do QRCode
   

    #Grava as informacoes no relatori
    print "gravando"

    #Informa ao robotino
    GPIO.output(37,GPIO.HIGH)
    #Aguarda 2 seg para retirar o sinal
    time.sleep(2)
    GPIO.output(37,GPIO.LOW)
    
    

GPIO.cleanup()
'''
