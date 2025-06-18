'''
OBJETIVO => armar una señal totalmente sintética que permita corroborar que el código del algoritmo de Juliana funciona
correctamente.
Se trata de una señal de 2 segundos con Fs = 250hz cuyos datos responden a una distribución normal cuya media cambia en forma abrupta
en la mitad de la señal


IMPORTANTE:

- Cambiar el valor de "n_partes" en la función generate_signal y en la construcción de la señal
para mover el punto en el que cambia la distribución

- Cambiar el nombre del archivo en el que se guarda la señal para no pisar señales anteriores
'''


#%%
#LIBRERÍA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
#FUNCIONES
def generate_signal(fs=250, duration=2.0, mu1=0.0, mu2=5.0, sigma=1.0, n_partes=4):
    """
    Genera una señal de duración 'duration' segundos con frecuencia de muestreo fs.
    - Primera parte: distribución normal N(mu1, sigma^2)
    - Segunda parte: distribución normal N(mu2, sigma^2)
    """
    n_samples = int(fs * duration)
    half = n_samples // n_partes
    # Generación de ambas mitades
    sig1 = np.random.normal(loc=mu1, scale=sigma, size=half)
    sig2 = np.random.normal(loc=mu2, scale=sigma, size=n_samples - half)
    signal = np.concatenate((sig1, sig2))
    # Vector de tiempo
    time = np.arange(n_samples) / fs
    return time, signal

#%%
#CONSTRUCCIÓN DE LA SEÑAL
if __name__ == "__main__":
    # Parámetros configurables
    fs = 250         # Frecuencia de muestreo (Hz)
    duration = 2.0   # Duración de la señal (segundos)
    mu1 = 5.0        # Media de la primera mitad
    mu2 = 10.0        # Media de la segunda mitad
    sigma = 1.0      # Desviación estándar
    n_partes=4
    
    # Generar señal
    time, signal = generate_signal(fs, duration, mu1, mu2, sigma)
    
    # Estadísticas
    print(f"Número de muestras: {len(signal)}")
    print(f"Media primera parte: {signal[:len(signal)//n_partes].mean():.2f}")
    print(f"Media segunda parte: {signal[len(signal)//n_partes:].mean():.2f}")
    
 #%%
 # Guardar en CSV
    df = pd.DataFrame({
        'time': time,
        'amplitude': signal
    })
    output_path = 'señal_sintetica/signal_2.csv'  # puedes cambiar el nombre o ruta aquí
    df.to_csv(output_path, index=False)
    
 
 #%%   
    # Graficar
    plt.figure(figsize=(8, 4))
    plt.plot(time, signal)
    plt.axvline(x=duration/n_partes, color='red', linestyle='--', label='Cambio de media')
    plt.xlabel
    
  