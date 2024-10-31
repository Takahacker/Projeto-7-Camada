from suaBibSignal import signalMeu
import peakutils
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import time
from scipy import signal

# Parâmetros do filtro passa-baixa
fc = 100  # Frequência de corte em Hz
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
def low_pass_filter(data):
    data = np.nan_to_num(data)  # Substitui NaNs e infs por zero
    filtered_data = signal.lfilter(b, a, data)
    return filtered_data

# Função para gerar o sinal de áudio com amplitude reduzida
def gerar_sinal(frequencia, duration=2, amplitude=0.5):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sinal = amplitude * np.sin(2 * np.pi * frequencia * t)
    return sinal

# Função principal do receptor
def main():
    # Cria um objeto da classe da biblioteca fornecida
    signal = signalMeu()
    
    # Define a frequência do sinal a ser emitido
    frequencia_sinal = 1200  
    duration = 2  # Duração do sinal em segundos
    numAmostras = int(duration * fs)
    
    # Iniciar a gravação antes de emitir o sinal
    print("Iniciando gravação do áudio...")
    audio = sd.rec(numAmostras, samplerate=fs, channels=1)
    time.sleep(0.5)  # Pequena pausa para garantir sincronização
    
    # Emissão do sinal enquanto grava
    print(f"Emitindo um sinal de {frequencia_sinal} Hz por {duration} segundos com amplitude reduzida...")
    sinal = gerar_sinal(frequencia_sinal, duration, amplitude=0.5)
    sd.play(sinal, samplerate=fs)
    sd.wait()  # Aguarda o final da reprodução
    
    # Aguarda o término da gravação
    sd.wait()
    print("Gravação finalizada.")
    
    # Dados de áudio capturados
    dados = audio[:, 0]  # Captura o canal 0 (mono)

    # Remover valores extremos (maior que um limite arbitrário)
    dados = np.where(np.abs(dados) > 1e5, 0, dados)  # Substitui valores > 1e5 por zero

    # Normalizar o sinal captado se houver valores significativos
    max_val = np.max(np.abs(dados))
    if max_val > 0:
        dados = dados / max_val
    
    # Exibir o sinal no domínio do tempo para verificar se o áudio foi captado
    plt.figure()
    plt.plot(dados)
    plt.title("Sinal Gravado - Domínio do Tempo")
    plt.xlabel("Amostras")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    # Aplicando o filtro passa-baixa
    dados_filtrados = low_pass_filter(dados)
    
    # Realiza a FFT do sinal original e filtrado
    print("Calculando a FFT do sinal gravado...")
    xf_original, yf_original = signal.calcFFT(dados, fs)
    xf_filtrado, yf_filtrado = signal.calcFFT(dados_filtrados, fs)
    
    # Plotando a FFT do sinal original e filtrado lado a lado
    plt.figure(figsize=(12, 6))

    # Sinal original
    plt.subplot(1, 2, 1)
    plt.plot(xf_original, np.abs(yf_original))
    plt.title("FFT do Sinal Gravado - Antes do Filtro")
    plt.xlabel("Frequência [Hz]")
    plt.ylabel("Magnitude")
    plt.grid(True)

    # Sinal filtrado
    plt.subplot(1, 2, 2)
    plt.plot(xf_filtrado, np.abs(yf_filtrado))
    plt.title("FFT do Sinal Filtrado - Depois do Filtro Passa-Baixa")
    plt.xlabel("Frequência [Hz]")
    plt.ylabel("Magnitude")
    plt.grid(True)
    
    # Exibindo os gráficos
    plt.tight_layout()
    plt.show()
    
    # Identificação dos picos na FFT do sinal filtrado
    indices = peakutils.indexes(np.abs(yf_filtrado), thres=0.1, min_dist=50)
    frequencias_pico = xf_filtrado[indices]
    print(f"Frequências dos picos detectados: {frequencias_pico}")

if __name__ == "__main__":
    main()