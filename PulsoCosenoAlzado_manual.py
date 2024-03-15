import numpy as np
import matplotlib.pyplot as plt
from rcosdesign import rcosdesign

# Parametros de cambio
Ts =    1                               # Duración de simbolo
L =     16                              # Numero de muestras por simbolo
a0 =    1e-6                            # Factor de rodamiento
a1 =    0.25                            # Factor de rodamiento
a2 =    0.5                             # Factor de rodamiento
a3 =    0.75                            # Factor de rodamiento
a4 =    1                               # Factor de rodamiento
t =     np.arange(-3, 3+Ts/L, Ts/L)     # Vector de tiempo para eje x

# Muestreo del pulso de coseno alzado (normal)
ptn_0 = rcosdesign(a0, 6, L, 'normal')
ptn_1 = rcosdesign(a1, 6, L, 'normal')
ptn_2 = rcosdesign(a2, 6, L, 'normal')
ptn_3 = rcosdesign(a3, 6, L, 'normal')
ptn_4 = rcosdesign(a4, 6, L, 'normal')

# Muestro del pulso de coseno alzado (raiz)
ptr_0 = rcosdesign(a0, 6, L, 'sqrt')
ptr_1 = rcosdesign(a1, 6, L, 'sqrt')
ptr_2 = rcosdesign(a2, 6, L, 'sqrt')
ptr_3 = rcosdesign(a3, 6, L, 'sqrt')
ptr_4 = rcosdesign(a4, 6, L, 'sqrt')


# Visuales

plt.subplots()
plt.plot(t, ptn_0, label = 'PCA (FR=0)')
plt.plot(t, ptn_1, label = 'PCA (FR=0.25)')
plt.plot(t, ptn_2, label = 'PCA (FR=0.5)')
plt.plot(t, ptn_3, label = 'PCA (FR=0.75)')
plt.plot(t, ptn_4, label = 'PCA (FR=1)')

plt.grid(True)
plt.xlabel('Tiempo [T]')
plt.ylabel('Amplitud [V]')
plt.title('Pulso de Coseno Alzado (PCA normal) ')
plt.legend(loc='upper right')

# segunda
plt.subplots()
plt.plot(t, ptr_0, label = 'PCA (FR=0)')
plt.plot(t, ptr_1, label = 'PCA (FR=0.25)')
plt.plot(t, ptr_2, label = 'PCA (FR=0.5)')
plt.plot(t, ptr_3, label = 'PCA (FR=0.75)')
plt.plot(t, ptr_4, label = 'PCA (FR=1)')

plt.grid(True)
plt.xlabel('Tiempo [T]')
plt.ylabel('Amplitud [V]')
plt.title('Pulso de Coseno Alzado (PCA sqrt)')
plt.legend(loc='upper right')
plt.show()
