import numpy as np
import matplotlib.pyplot as plt

# Definir una lista de valores de SNR
SNR_values =  [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  # Comenzando desde 10 dB, decrementando 1 dB a la vez
BER_valuesA = [0.0, 0.0, 0.0, 5e-5, 0.0001225, 0.0003825,0.0057375, 0.0183875, 0.0474625, 0.0981875, 0.1534125]
BER_valuesB = [0.0, 0.0, 0.0, 0.0, 1.25e-5, 0.0001375, 0.0004, 0.001075, 0.0058, 0.01545, 0.0364]

# Trazar el gráfico SNR vs BER
plt.figure(figsize=(10, 6))
plt.plot(SNR_values, BER_valuesA, marker='o', linestyle='-', label='BER for encoder A')
plt.plot(SNR_values, BER_valuesB, marker='o', linestyle='-', label='BER for encoder B')
plt.xlabel('SNR [dB]')
plt.yscale('log')
plt.ylabel('Bit Error Rate (BER)')
plt.legend()
plt.grid(True)
plt.show()