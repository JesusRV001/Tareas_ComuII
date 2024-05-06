import numpy as np
import random
import galois

GF = galois.GF(2**4)

def get_generator_poly():
    return np.array([1, 13, 12, 8, 7])

def calc_syndrome(error_vector, generator_poly):
    syndrome = np.zeros(len(generator_poly) - 1, dtype=int)
    for i in range(len(generator_poly) - 1):
        value = 0
        for j in range(len(error_vector)):
            value += error_vector[j] * generator_poly[(i - j) % len(generator_poly)]
        syndrome[i] = value % 15
    return tuple(syndrome)

def generate_error_vectors(n, t):
    error_vectors = {}
    for i in range(n):
        for j in range(i+1, n):
            for a in range(15):
                for b in range(15):
                    error_vector = np.zeros(n, dtype=int)
                    error_vector[[i, j]] = [a, b]
                    syndrome = calc_syndrome(error_vector, get_generator_poly())
                    error_vectors[syndrome] = error_vector.copy()
    return error_vectors

def codificar(mensaje):
    polinomio = ceros(mensaje)
    r = []  # Inicializar r aquí
    while len(polinomio) > len(get_generator_poly()):
        r.append(int(GF(get_generator_poly()[0] * polinomio[0])))
        resta = []
        for i in range(len(get_generator_poly())):
            resta.append(int(GF(r[-1]) * GF(get_generator_poly()[i])))
        for k in range(len(get_generator_poly())):
            polinomio[k] = int(GF(polinomio[k]) + GF(resta[k]))
        polinomio = ceros(polinomio)
    codigo_codificado = mensaje + polinomio
    return codigo_codificado

def ceros(polinomio):
    largo = len(polinomio)
    for e in range(largo-1):
        for i in range(len(polinomio)-1):
            if polinomio[0] == 0:
                del polinomio[0]
            else:
                break
    return polinomio

syndrome_dict = generate_error_vectors(15, 2)

# Generar un vector de mensaje de prueba
mensaje = [7, 5, 2, 13, 6, 3, 4, 7, 11, 6, 0]

# Codificar el vector de mensaje
codigo_codificado = codificar(mensaje)
print("Código original:", codigo_codificado)

# Introducir errores en el código codificado
vector_error_1 = [0] * 15
indice_error = random.randint(0, 14)
valor_error = random.randint(1, 15)
vector_error_1[indice_error] = valor_error

codigo_con_error_1 = [sum(x) for x in zip(codigo_codificado, vector_error_1)]

# Calcular el síndrome para el código con error
sindrome_1 = calc_syndrome(codigo_con_error_1, get_generator_poly())

# Buscar el vector de error correspondiente al síndrome calculado
vector_error_estimado_1 = syndrome_dict[sindrome_1]

# Corregir el código con error
codigo_corregido_1 = [sum(x) for x in zip(codigo_con_error_1, [-y for y in vector_error_estimado_1])]

# Mostrar el código corregido
print("Código corregido 1:", codigo_corregido_1)
