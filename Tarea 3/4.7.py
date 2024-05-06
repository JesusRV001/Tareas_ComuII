import numpy as np

def get_generator_poly():
    return np.array([1, 13, 12, 8, 7])

def calc_syndrome(error_vector, generator_poly):
    syndrome = np.zeros(len(generator_poly) - 1, dtype=int)
    for i in range(len(generator_poly) - 1):
        value = 0
        for j in range(len(error_vector)):
            value += error_vector[j] * generator_poly[(i - j) % len(generator_poly)]  
        syndrome[i] = value % 15  
    print("Síndrome calculado:", syndrome)  
    return syndrome


def generate_error_vectors(n, t):
    error_vectors = {}
    for i in range(n):
        for j in range(i+1, n):
            for a in range(15):  
                for b in range(15):
                    error_vector = np.zeros(n, dtype=int)
                    error_vector[[i, j]] = [a, b]
                    syndrome = tuple(calc_syndrome(error_vector, get_generator_poly()))
                    error_vectors[syndrome] = error_vector.copy() 
    return error_vectors


def main():
    syndrome_dict = generate_error_vectors(15, 2)

    print("Diccionario de síndromes y vectores de error:")
    for syndrome, error_vector in syndrome_dict.items():
        print(f"Síndrome: {syndrome}, Vector de error: {error_vector}")

    user_syndrome = input("Ingresa los coeficientes del síndrome correspondiente separado por un espacio: ")
    user_syndrome = list(map(int, user_syndrome.split()))

    if tuple(user_syndrome) in syndrome_dict:
        corresponding_error_vector = syndrome_dict[tuple(user_syndrome)]
        print("Vector de error correspondiente:", corresponding_error_vector)
    else:
        print("No se encontró un vector de error correspondiente para el síndrome ingresado.")

if __name__ == "__main__":
    main()
