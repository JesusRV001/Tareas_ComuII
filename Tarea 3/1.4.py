import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import galois
GF=galois.GF(2**4)
def ceros(polinomio):
    largo=len(polinomio)
    for e in range(largo-1):
        for i in range(len(polinomio)-1):
             if(polinomio[0]==0):
                 del polinomio[0]
             else:
                break
    return polinomio

def RS(polinomio_g):
    polinomio=polinomio_g
    ss=polinomio
    polinomio.append(0)
    polinomio.append(0)
    polinomio.append(0)
    polinomio.append(0)
    suma=GF.arithmetic_table("+")
    multi=(GF.arithmetic_table("*"))
    poli=[1,13,12,8,7]
    largo=len(polinomio)
    r=[]
    indice=0
    polinomio=ceros(polinomio)
    while(len(polinomio)>=len(poli)):
        r.append(int(GF(poli[0]*polinomio[0])))
        resta=[]
        for i in range(len(poli)):
            resta.append(int(GF(r[-1])*GF(poli[i])))
        for k in range(len(poli)):
            polinomio[k]=int(GF(polinomio[k])+GF(resta[k]))
        for i in range(len(poli)-1):
            if(polinomio[i+1]==0):
                r.append(0)
        polinomio=ceros(polinomio)
    print("residuo=   ",polinomio,)
    print("cociente=  ",r)
    return polinomio

def deRS(polinomio_g):
    polinomio=polinomio_g
    ss=polinomio
    poli=[1,13,12,8,7]
    largo=len(polinomio)
    r=[]
    indice=0
    polinomio=ceros(polinomio)
    while(len(polinomio)>=len(poli)):
        r.append(int(GF(poli[0]*polinomio[0])))
        resta=[]
        for i in range(len(poli)):
            resta.append(int(GF(r[-1])*GF(poli[i])))
        for k in range(len(poli)):
            polinomio[k]=int(GF(polinomio[k])+GF(resta[k]))
        for i in range(len(poli)-1):
            if(polinomio[i+1]==0):
                r.append(0)
        polinomio=ceros(polinomio)
    print("residuo=   ",polinomio,)
    print("cociente=  ",r)
    return polinomio


mensaje=[7,5,2,13,6,3,4,7,11,6,0]
poli=mensaje.copy()
print("mensaje",mensaje)
residuo=RS(mensaje)
for i in range(len(residuo)):
    poli.append(residuo[i])
total=poli.copy()
print("mensaje con residuo",poli)
n_res=deRS(poli)
if(n_res==0):
    print("funciona")
else:
    print("valio madre")

    



    
