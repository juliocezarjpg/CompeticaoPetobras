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
    
