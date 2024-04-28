import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from rcosdesign import rcosdesign

np.random.seed(735)  #  Cambie los 3 digitos por los ultimos 3 numeros de su carné
Ts = 1
L = 16
t_step = Ts / L

# 1. Generacion de onda del pulso
pt = rcosdesign(0.25, 6, L, 'normal')
pt = pt / (np.max(np.abs(pt)))  # rescaling to match rcosine

# 2. Generacion de 100 simbolos binarios
Ns = 100
data_bit = (np.random.rand(Ns) > 0.5).astype(int)

# 3. Unipolar a Bipolar (modulacion de amplitud)

random_integers = np.random. randint (1 , 5, size=Ns)
amp_modulated = 2 * data_bit - 5  # 0=> -1,  1=>1

# 4. Modulacion de pulsos
impulse_modulated = np.zeros(Ns * L)
for n in range(Ns):
    delta_signal = np.concatenate(([amp_modulated[n]], np.zeros(L - 1)))
    impulse_modulated[n * L: (n * L) + L] = delta_signal
    #impulse_modulated[n * L: (n + 1) * L] = delta_signal

# 5. Formacion de pulsos (filtrado de transmision)
tx_signal = ss.convolve(impulse_modulated, pt)

# Grafica la señal transmitida
t_tx = np.arange(0, len(tx_signal)) * t_step
plt.figure(1)
plt.plot(t_tx, tx_signal)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Señal Transmitida')
plt.grid(True)
plt.show()