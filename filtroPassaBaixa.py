import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import sounddevice as sd

# Parâmetros do filtro passa-baixa
fc = 2500  # Frequência de corte em Hz
fs = 44100  # Taxa de amostragem em Hz
w_c = 2 * np.pi * fc  # Frequência de corte em rad/s
zeta = 0.707  # Amortecimento do filtro

# Função de transferência contínua e discretização
num = [w_c**2]
den = [1, 2 * zeta * w_c, w_c**2]
system_s = signal.TransferFunction(num, den)
system_z = system_s.to_discrete(1 / fs, method='bilinear')
b, a = system_z.num, system_z.den

# Função para aplicar o filtro passa-baixa
def low_pass_filter(signal):
    filtered_signal = signal.copy()
    for k in range(2, len(signal)):
        filtered_signal[k] = (b[0] * signal[k] + b[1] * signal[k-1] + b[2] * signal[k-2]
                              - a[1] * filtered_signal[k-1] - a[2] * filtered_signal[k-2])
    return filtered_signal

# Geração de um sinal de teste (soma de 500 Hz e 5000 Hz)
duration = 2  # Duração em segundos
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
test_signal = np.sin(2 * np.pi * 500 * t) + 0.5 * np.sin(2 * np.pi * 5000 * t)

# Captura de áudio do microfone
print("Gravando áudio do microfone...")
audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float64')
sd.wait()  # Aguarda o fim da gravação
audio_data = audio_data.flatten()  # Converte para uma dimensão

# Aplicação do filtro
filtered_audio = low_pass_filter(audio_data)

# Transformada de Fourier
def plot_fft(signal, title, fs):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / fs)[:N//2]
    plt.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))
    plt.title(title)
    plt.xlabel('Frequência [Hz]')
    plt.ylabel('Amplitude')

# Plotagem do Fourier antes e depois do filtro
plt.figure(figsize=(12, 8))

# Sinal original do microfone
plt.subplot(2, 1, 1)
plot_fft(audio_data, 'Fourier do Sinal Captado (Antes do Filtro)', fs)

# Sinal filtrado
plt.subplot(2, 1, 2)
plot_fft(filtered_audio, 'Fourier do Sinal Filtrado (Depois do Filtro Passa-Baixa)', fs)

plt.tight_layout()
plt.show()