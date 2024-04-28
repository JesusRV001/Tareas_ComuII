import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from rcosdesign import rcosdesign

np.random.seed(588)  #  Cambie los 3 digitos por los ultimos 3 numeros de su carné
k=2
Ts = 1
L = 16
t_step = Ts / L

# 1. Generacion de onda del pulso
pt = rcosdesign(0.5, 6, L, 'sqrt')
pt = pt / (np.max(np.abs(pt)))  # rescaling to match rcosine

# 2. Generacion de 100 simbolos binarios
Ns = 100
data_bit = (np.random.rand(100) > 0.5).astype(int)

# 3. Unipolar a Bipolar (modulacion de amplitud)
amp_modulated = 2 * data_bit - 1  # 0=> -1,  1=>1
##random_integers = np.random.randint(1, 5, size=Ns)## descomentar para 4_PAM
##
####Performamplitude modulation
##amp_modulated = 2 * random_integers - 5
# 4. Modulacion de pulsos
impulse_modulated = np.zeros(Ns * L)
for n in range(Ns):
    delta_signal = np.concatenate(([amp_modulated[n]], np.zeros(L - 1)))
    impulse_modulated[n * L: (n * L) + L] = delta_signal
    #impulse_modulated[n * L: (n + 1) * L] = delta_signal
#############




###############
# 5. Formacion de pulsos (filtrado de transmision)

tx_signal = ss.convolve(impulse_modulated, pt)
# Generar números aleatorios de baja magnitud como ruido
length_tx_signal = len(tx_signal)
randn_array = 0.25 * np.random.randn( length_tx_signal)
print(len(randn_array))
# Agregar ruido a la señal se realizo un cambio para observar de manera mejor el cambio en la senal de ruido 
for k in range(length_tx_signal):
    rx_signal = tx_signal[k] + randn_array[k]## a cada valor de tx_signal se le agrega una fuente de ruido diferente

pt = rcosdesign(1, 6, L, 'sqrt')
pt = pt/(np.max(np.abs(pt)))  # rescaling to match rcosine
matchedout=ss.convolve(tx_signal,pt)




# Grafica la señal transmitida
t_tx =np.arange(0, len(tx_signal)) * t_step
plt.figure(1)
plt.plot(t_tx, tx_signal)
plt.xlabel('Tiempo (s)')
plt.ylabel('Amplitud')
plt.title('Señal Transmitida')
plt.grid(True)


ttx = np.arange(0, len(tx_signal)) * t_step
plt.figure(2)
# Plotting the modulated pulse
plt.subplot(2, 1, 1)
plt.stem(np.arange(t_step, Ns * Ts + t_step, t_step), impulse_modulated, markerfmt='.')
plt.axis([0, Ns * Ts, -2 * np.max(impulse_modulated), 2 * np.max(impulse_modulated)])
plt.grid(True)
plt.title('Pulso Modulado')

# Plotting the signal waveform
plt.subplot(2, 1, 2)
plt.plot(np.arange(t_step, t_step * len(tx_signal) + t_step, t_step), tx_signal)
##plt.axis([0, Ns * Ts, -2 * np.max(matchedout), 2 * np.max(matchedout)])
plt.grid(True)
plt.title('Forma de Pulso')

plt.tight_layout()


# Plotting the signal waveform



tmp = matchedout[(k - 1) * 2 * L: k * 2 * L]


plt.figure(3)
plt.grid(True)

for k in range(2, Ns // 2):
    tmp = matchedout[(k-1) * 2 * L:k * 2 * L]
    plt.plot(t_step * np.arange(2 * L), tmp)
    #plt.axis([0, 2, np.min(matchedout), np.max(matchedout)])
    #plt.pause(0.1) ### se elimina esta linea para una mayor fluides en la graficacion y ejecucion
    plt.title('ojo de tx')



plt.show()



