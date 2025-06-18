'''
OBJETIVO => probar el algoritmo de detección de bordes con Distribución Normal

IMPORTANTE => está simplificado => no hay filtros ni transformación de los datos ya que
se toma una base sintética creada especialmente para esta prueba

Cambiar el nombre del archivo con la señal que se quiere levantar
'''

#%%
#LIBRERÍAS
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

#%%

#FUNCIONES

# Paso 1: Dividir el segmento en ventanas
def dividir_en_ventanas(segmento, ventana_size):
    """Divide el segmento en ventanas consecutivas de tamaño especificado (sin solapamiento)."""
    num_ventanas = len(segmento) // ventana_size
    ventanas = [segmento[i*ventana_size: (i+1)*ventana_size] for i in range(num_ventanas)]
    print(len(segmento))
    print(num_ventanas)
    print(len(ventanas))
    return ventanas

# Calcula los parámetros de la distribución Normal para la ventana
def calcular_parametros_normal(ventana):
    """Calcula los parámetros de la distribución Normal para la ventana."""
    
    # Ajustamos la distribución normal a los datos de la ventana
    mu, std = norm.fit(ventana)  # Ajuste de la distribución normal (media y desviación estándar)
    return mu, std  # Retorna los parámetros de la distribución normal: media y desviación estándar


# Paso 2: Calcular la diferencia entre los parámetros alfa de ventanas consecutivas
def calcular_diferencia_alfa(ventanas):
    """Calcula la diferencia de alfa entre ventanas consecutivas y encuentra la mayor diferencia."""
    diferencias = []
    for i in range(len(ventanas)-1):
        alfa1, _ = calcular_parametros_normal(ventanas[i])
        alfa2, _ = calcular_parametros_normal(ventanas[i+1])
        diferencia = abs(alfa1 - alfa2)
        diferencias.append(diferencia)
        print(f"Parámetros Alfa 1= {alfa1}")
        print(f"Parámetros Alfa 2= {alfa2}")  
        print(f"Diferencia= {diferencia}")    
       
    
    # Encontrar el índice de la mayor diferencia
    idx_max_diff = np.argmax(diferencias)
    print(f"MaxDif= {idx_max_diff}")
    return idx_max_diff, diferencias


# Paso 3: Maximizar la función de verosimilitud para encontrar el punto de transición

def negativo_log_verosimilitud_normal(ventana, mu, std,epsilon=1e-10):
    """Calcula el negativo de la log-verosimilitud para una ventana de datos
    usando los parámetros alpha y gamma (scale) de la distribución Normal."""

    log_likelihood = 0
    for z in ventana:
        # Evaluamos la función de densidad de probabilidad Normal para el valor z
        pdf_value = norm.pdf(z, mu, std)

        # Si la PDF es cero, la corregimos para evitar logaritmos de cero
        pdf_value = max(pdf_value, epsilon)

        # Sumamos el logaritmo de la PDF a la log-verosimilitud total
        log_likelihood += np.log(pdf_value)

    # El negativo de la log-verosimilitud
    neg_log_likelihood = -log_likelihood

    return neg_log_likelihood

# Paso 4: Identificar el punto de cambio de verosimilitud

def encontrar_punto_cambio_verosimilitud(inicio_ventana1,ventana1, ventana2):
    """
    Encuentra el punto de cambio entre ventana1 y ventana2 maximizando la log-verosimilitud.

    Args:
        ventana1 (np.ndarray): Datos de la primera ventana.
        ventana2 (np.ndarray): Datos de la segunda ventana.
        n_looks_fijo (float, optional): Valor fijo para el número de looks (n). Defaults to None.

    Returns:
        int: El índice del punto de cambio en el segmento combinado (inicio en 0).
             Retorna None si ocurre un error en la optimización.
    """
    segmento_combinado = np.concatenate((ventana1, ventana2))
    n_total = len(segmento_combinado)
    mejor_log_verosimilitud = -np.inf
    mejor_punto_cambio = None

    # Iterar sobre todos los posibles puntos de cambio dentro del segmento combinado
    # Excluimos los extremos para asegurar que haya datos en ambas partes
    for punto_cambio in range(10, n_total-10):
        parte1 = segmento_combinado[:punto_cambio]
        parte2 = segmento_combinado[punto_cambio:]

        if len(parte1) > 0 and len(parte2) > 0:
            mu1, std1 = calcular_parametros_normal(parte1)
            mu2, std2 = calcular_parametros_normal(parte2)

            if mu1 is not None and std1 is not None and mu2 is not None and std2 is not None:
                log_vero1 = -negativo_log_verosimilitud_normal(parte1, mu1, std1)
                log_vero2 = -negativo_log_verosimilitud_normal(parte2, mu2, std2)
                 # Imprimir las log-verosimilitudes
                
                log_verosimilitud_total = log_vero1 + log_vero2
                print(f"Punto de cambio: {punto_cambio}, Log-vero1: {log_vero1:.4f}, Log-vero2: {log_vero2:.4f}, Log-vero Total: {log_verosimilitud_total:.4f}")
            
            
                if log_verosimilitud_total > mejor_log_verosimilitud:
                    mejor_log_verosimilitud = log_verosimilitud_total
                    mejor_punto_cambio = punto_cambio
            
            else:
                print(f"Fallo al estimar parámetros en el punto de cambio: {punto_cambio}")        
    
    punto_transicion=mejor_punto_cambio+inicio_ventana1
    
    print(f"Mejor Log Verosimilitud: {mejor_log_verosimilitud}")
    print(f"Punto de cambio: {mejor_punto_cambio}")
    print(f"Inicio Ventana 1: {inicio_ventana1}")
    print(f"Mejor punto de cambio: {punto_transicion}")
    return punto_transicion


    

#DEFINICIÓN DEL PUNTO DE TRANSICIÓN

# %%
#Levanto el segmento sintético
directorio_entrada = 'señal_sintetica/signal_2.csv'
segmento = pd.read_csv(directorio_entrada)
print(f"Número de muestras: {len(segmento)}")

#%%
# Función para graficar la señal original
plt.figure(figsize=(10, 6))
plt.plot(segmento['amplitude'], label='Señal')
plt.title("Señal Original")
plt.xlabel("Muestras")
plt.ylabel("Valor")
plt.legend()
plt.show()


# %%
#DIVIDIR EN VENTANAS

ventana_size=100
segmento=segmento['amplitude']
ventanas = dividir_en_ventanas(segmento, ventana_size)

for i, ventana in enumerate(ventanas):
       alpha, gamma_scale = calcular_parametros_normal(ventana)
       print(f"Ventana {i+1}: Media = {alpha}, Sigma = {gamma_scale}")

#%%
# Calcular la diferencia de alfa entre las ventanas consecutivas
idx_max_diff, _ = calcular_diferencia_alfa(ventanas)
    
# Tomar las dos ventanas consecutivas con el mayor salto en alfa
ventana1 = ventanas[idx_max_diff]
ventana2 = ventanas[idx_max_diff + 1]
    
# Calcular el índice inicial de la ventana 1 (esto puede ser dinámico)
inicio_ventana1 = idx_max_diff * ventana_size  # O cualquier cálculo que determine la posición de la ventana
print(f"Inicio Ventana 1 = {inicio_ventana1}") 
print(f"Ventana 1 = {ventana1}") 
print(f"Ventana 2 = {ventana2}") 


#%%
# Encontrar el punto de transición usando Gamma
punto_transicion = encontrar_punto_cambio_verosimilitud(inicio_ventana1, ventana1, ventana2)
print(f"Punto de Transición = {punto_transicion}")  




