import numpy as np
import matplotlib.pyplot as plt
from rcosdesign import rcosdesign

Ts =    1                               # Duración de simbolo
L =     16                              # Numero de muestras por simbolo
a =     1e-6                            # Factor de rodamiento
t =     np.arange(-3, 3+Ts/L, Ts/L)     # Vector de tiempo para eje x
i =     0                               # Contador del ciclo
b =     0                               # Contador de referencia

for i in range(1, 6):
    pcan = rcosdesign(a, 6, L, 'normal')
    plt.figure(1)
    plt.plot(t, pcan, label = 'PCA FR={:.2f}'.format(b))

    plt.grid(True)
    plt.xlabel('Tiempo [T]')
    plt.ylabel('Amplitud [V]')
    plt.title('Pulso de Coseno Alzado (PCA normal)')
    plt.legend(loc='upper right')

    pcar = rcosdesign(a, 6, L, 'sqrt')
    plt.figure(2)
    plt.plot(t, pcar, label = 'PCA FR={:.2f}'.format(b))

    plt.grid(True)
    plt.xlabel('Tiempo [T]')
    plt.ylabel('Amplitud [V]')
    plt.title('Pulso de Coseno Alzado (PCA SCCR )')
    plt.legend(loc='upper right')

    a = a+0.249                         # Adicionador de factor de rodamiento
    b = b+0.25                          # Contador de referencia para labels
    
plt.show()




