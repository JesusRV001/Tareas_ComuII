import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from rcosdesign import rcosdesign

np.random.seed(335)  #  Cambie los 3 digitos por los ultimos 3 numeros de su carné

Ts = 1
L = 16
t_step = Ts / L

# 1. Generacion de onda del pulso
pt = rcosdesign(0.25, 6, L, 'normal')
pt = pt / (np.max(np.abs(pt)))  # rescaling to match rcosine

# 2. Generacion de 100 simbolos binarios
Ns = 1335
data_bit = (np.random.rand(Ns) > 0.5).astype(int)

# 3. Unipolar a Bipolar (modulacion de amplitud)
amp_modulated = 2 * data_bit - 1  # 0=> -1,  1=>1

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

# Gráfica de pulso modulado y forma del pulso
plt.figure(2)
plt.subplot(2,1,1)
plt.stem(np.arange(t_step, Ns*Ts+t_step,t_step),impulse_modulated,markerfmt='.')
plt.axis([0,Ns*Ts,-2*np.max(impulse_modulated),2*np.max(impulse_modulated)])
plt.grid(True)
plt.title('Pulso modulado')

plt.subplot(2,1,2)
plt.plot(np.arange(t_step,t_step*len(tx_signal)+t_step,t_step),tx_signal)
plt.axis([0,Ns*Ts,-2*np.max(tx_signal),2*np.max(tx_signal)])
plt.grid()
plt.title('Forma de pulso')

plt.tight_layout()
plt.show()

# Diagrama de ojo
plt.figure(3)
plt.grid(True)
for k in range(2, Ns // 2):
    tmp = tx_signal[(k - 1)*2*L:k*2*L]
    plt.plot(t_step*np.arange(2*L),tmp)
    plt.axis([0,2,np.min(tx_signal),np.max(tx_signal)])
    #plt.pause(0.1)
plt.show()