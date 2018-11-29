
def converte(ang):
    real = ang
    manometro = 225 - real

    pressao = (14.0*(manometro-270))/270.0 + 14.0

    return(manometro, pressao)
