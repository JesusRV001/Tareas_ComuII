import galois 

GF=galois.GF(2**4)

print("-----------------------------------------------------------------------")
 
#------------- POLINOMIO IRREDUCIBLE -------------

print("Polinomio Irreducible")
irred = GF.properties
print(irred)
print("-----------------------------------------------------------------------")
#------------- SUMA Y MULTIPLICACION -------------

print("Matriz de suma:")
print(GF.arithmetic_table("+"))
print("Matriz de multiplicacion:")
print((GF.arithmetic_table("*")))
print("-----------------------------------------------------------------------")

#------------- POLINOMIO GENERADOR -------------
print("Polinomio generador:")
P_gen= galois.ReedSolomon(15,11)
print(P_gen)
print("-----------------------------------------------------------------------")

#------------- MULTIPLICACIONES SOLICITADAS -------------

print("Multiplicaciones del punto 2:")

m1=GF(2)*GF(3)
print("2*3 = ", m1)
m2=GF(3)*GF(4)
print("3*4 = ", m2)
m3=GF(6)*GF(11)
print("6*11 = ", m3)
m4=GF(8)*GF(12)
print("8*12 = ", m4)
m5=GF(5)*GF(15)
print("5*15 = ", m5)
