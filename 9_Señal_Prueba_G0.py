'''
OBJETIVO => probar el algoritmo de detección de bordes con Distribución G0

IMPORTANTE => está simplificado => no hay filtros ni transformación de los datos ya que
se toma una base sintética creada especialmente para esta prueba

Cambiar el nombre del archivo con la señal que se quiere levantar
'''

#%%
#LIBRERÍAS
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import gamma
import matplotlib.pyplot as plt
import os
from scipy.special import gamma as gamma_func
from scipy.optimize import minimize

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

# Calcula los parámetros de la distribución G0 para la ventana
def calcular_parametros_g0(ventana, n_looks=1):
    # Definimos los límites fuera para poder referenciarlos
    bounds_2params = [(-np.inf, -1e-6), (1e-6, np.inf)] # Para (alpha, gamma)
    bounds_3params = [(-np.inf, -1e-6), (1e-6, np.inf), (1e-6, np.inf)] # Para (alpha, n, gamma)
    
    # Añadir una comprobación para ventanas vacías o con valores problemáticos
    if len(ventana) == 0 or np.all(ventana == 0): # Puedes ajustar esta condición
        print("Advertencia: Ventana vacía o con todos los valores en cero. Se omitirá.")
        return None
    
    # Si los datos pueden contener valores negativos o cero y esto no es esperado por G0
    # Asegúrate de filtrar o transformar tus datos antes de pasarlos aquí
    if np.any(ventana <= 0):
        print("Advertencia: Ventana contiene valores no positivos. Se omitirá.")
        return None
    
    
    if n_looks is None:
        def objective_function_3params(params):
            alpha, n, gamma = params
            return log_likelihood_g0((alpha, n, gamma), ventana)

        initial_guess = (-np.mean(ventana), 2.0, np.mean(ventana**2))
        
        result = minimize(objective_function_3params, initial_guess, bounds=bounds_3params, method='L-BFGS-B')
        if result.success:
            # Verificar si los parámetros resultantes están en los límites de los bounds
            alpha_res, n_res, gamma_res = result.x
            
            # Comprobar si alpha es igual al límite inferior del bound para alpha
            # Se usa una tolerancia para la comparación de floats
            
            tolerance_absolute = 1e-7
            if (np.isclose(alpha_res, bounds_3params[0][0], atol=tolerance_absolute) or np.isclose(alpha_res, bounds_3params[0][1], atol=tolerance_absolute) or
                np.isclose(n_res, bounds_3params[1][0], atol=tolerance_absolute) or np.isclose(n_res, bounds_3params[1][1], atol=tolerance_absolute) or
                np.isclose(gamma_res, bounds_3params[2][0], atol=tolerance_absolute) or np.isclose(gamma_res, bounds_3params[2][1], atol=tolerance_absolute)):
                print(f"Advertencia: Parámetros de G0 (3 params) en los límites para la ventana. Se omitirá. Parámetros: {result.x}")
                return None
            return result.x
        else:
            print("Error en la optimización (3 parámetros):", result.message)
            return None
    else: # Caso n_looks está definido (ej. n_looks=1)
        def objective_function_2params(params):
            alpha, gamma = params
            return log_likelihood_g0((alpha, n_looks, gamma), ventana)

        initial_guess = (-np.mean(ventana), np.mean(ventana**2))
        result = minimize(objective_function_2params, initial_guess, bounds=bounds_2params, method='L-BFGS-B')
        if result.success:
            # Verificar si los parámetros resultantes están en los límites de los bounds
            alpha_res, gamma_res = result.x
            tolerance_absolute = 1e-7 # Puedes ajustar esta tolerancia
            if (np.isclose(alpha_res, bounds_2params[0][0], atol=tolerance_absolute) or np.isclose(alpha_res, bounds_2params[0][1], atol=tolerance_absolute) or
                np.isclose(gamma_res, bounds_2params[1][0], atol=tolerance_absolute) or np.isclose(gamma_res, bounds_2params[1][1], atol=tolerance_absolute)):
                print(f"Advertencia: Parámetros de G0 (n={n_looks}) en los límites para la ventana. Se omitirá. Parámetros: {(alpha_res, n_looks, gamma_res)}")
                return None
            return (result.x[0], n_looks, result.x[1])
        else:
            print(f"Error en la optimización (n={n_looks}):", result.message)
            return None


'''  VIEJO CÓDIGO SIN CONTROL DE LÍMITES 
def calcular_parametros_g0(ventana, n_looks=1):
    if n_looks is None:
        def objective_function_3params(params):
            alpha, n, gamma = params
            return log_likelihood_g0((alpha, n, gamma), ventana)

        initial_guess = (-np.mean(ventana), 2.0, np.mean(ventana**2))
        bounds = [(-np.inf, -1e-6), (1e-6, np.inf), (1e-6, np.inf)]
        result = minimize(objective_function_3params, initial_guess, bounds=bounds, method='L-BFGS-B')
        if result.success:
            return result.x
        else:
            print("Error en la optimización (3 parámetros):", result.message)
            return None
    else:
        def objective_function_2params(params):
            alpha, gamma = params
            return log_likelihood_g0((alpha, n_looks, gamma), ventana)

        initial_guess = (-np.mean(ventana), np.mean(ventana**2))
        bounds = [(-np.inf, -1e-6), (1e-6, np.inf)]
        result = minimize(objective_function_2params, initial_guess, bounds=bounds, method='L-BFGS-B')
        if result.success:
            return (result.x[0], n_looks, result.x[1])
        else:
            print(f"Error en la optimización (n={n_looks}):", result.message)
            return None
'''



# Paso 2: Calcular la diferencia entre los parámetros alfa de ventanas consecutivas
def calcular_diferencia_alfa(ventanas):
    """Calcula la diferencia de alfa entre ventanas consecutivas y encuentra la mayor diferencia,
    excluyendo ventanas donde los parámetros G0 estén en los límites de los bounds."""
    diferencias = []
    ventanas_validas_indices = []

    # PASO CRÍTICO: Pre-calcular los parámetros para todas las ventanas una sola vez
    # y almacenar el resultado (que puede ser None)
    parametros_calculados = []
    for ventana in ventanas:
        parametros_calculados.append(calcular_parametros_g0(ventana, n_looks=1))

    for i in range(len(parametros_calculados) - 1):
        params1 = parametros_calculados[i]
        params2 = parametros_calculados[i+1]

        # AHORA SÍ: Verificar si los parámetros no son None ANTES de intentar desempaquetar
        if params1 is not None and params2 is not None:
            alfa1, _, _ = params1
            alfa2, _, _ = params2
            diferencia = abs(alfa1 - alfa2)
            diferencias.append(diferencia)
            ventanas_validas_indices.append(i) # Guardar el índice de la primera ventana válida de la pareja
            print(f"Parámetros Alfa 1= {alfa1}")
            print(f"Parámetros Alfa 2= {alfa2}")  
            print(f"Diferencia= {diferencia}")
        else:
            # Si alguno de los dos sets de parámetros es None, se salta la pareja
            print(f"Saltando la diferencia entre ventana {i} y {i+1} debido a parámetros no válidos en G0.")
            
    if not diferencias:
        print("No se pudieron calcular diferencias de alfa válidas.")
        return None, []

    idx_max_diff_en_diferencias = np.argmax(diferencias)
    idx_max_diff_original = ventanas_validas_indices[idx_max_diff_en_diferencias]
    
    print(f"MaxDif (índice de la primera ventana de la pareja con mayor diferencia)= {idx_max_diff_original}")
    return idx_max_diff_original, diferencias


''' VIEJO CÓDIGO
def calcular_diferencia_alfa(ventanas):
    """Calcula la diferencia de alfa entre ventanas consecutivas y encuentra la mayor diferencia."""
    diferencias = []
       
    for i in range(len(ventanas)-1):
        alfa1, n,_ = calcular_parametros_g0(ventanas[i],n_looks=1)
        alfa2, n,_ = calcular_parametros_g0(ventanas[i+1],n_looks=1)
        diferencia = abs(alfa1 - alfa2)
        diferencias.append(diferencia)
        print(f"Parámetros Alfa 1= {alfa1}")
        print(f"Parámetros Alfa 2= {alfa2}")  
        print(f"Diferencia= {diferencia}")    
        
    # Encontrar el índice de la mayor diferencia
    idx_max_diff = np.argmax(diferencias)
    print(f"MaxDif= {idx_max_diff}")
    return idx_max_diff, diferencias
'''

# Paso 3: Maximizar la función de verosimilitud para encontrar el punto de transición

#FUNCIÓN DE DISTRIBUCIÓN G0
def pdf_g0(z, alpha, n, gamma):
    """Función de densidad de probabilidad (PDF) de la distribución G0."""
    numerator = 2 * (n**n) * gamma_func(n - alpha) * (z**(2 * n - 1))
    denominator = (gamma**alpha) * gamma_func(-alpha) * gamma_func(n) * ((gamma + n*(z**2))**(n - alpha))
    return numerator / denominator


def log_likelihood_g0(params, ventana):
    alpha, n, gamma = params
    z = ventana
    if n <= 0 or gamma <= 0:
        return np.inf
    if np.any(z < 0):
        return np.inf
    log_pdf=np.log((2 * (n**n) * gamma_func(n - alpha) * (z**(2 * n - 1)))/((gamma**alpha) * gamma_func(-alpha) * gamma_func(n) * ((gamma + n*(z**2))**(n - alpha))))          
    return -np.sum(log_pdf)

# Paso 4: Identificar el punto de cambio de verosimilitud

def encontrar_punto_cambio_verosimilitud(inicio_ventana1,ventana1, ventana2, n_looks=1):
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
    for punto_cambio in range(1, n_total):
        parte1 = segmento_combinado[:punto_cambio]
        parte2 = segmento_combinado[punto_cambio:]

        if len(parte1) > 0 and len(parte2) > 0:
            params1 = calcular_parametros_g0(parte1, n_looks=1)
            params2 = calcular_parametros_g0(parte2, n_looks=1)

            if params1 is not None and params2 is not None:
                log_vero1 = -log_likelihood_g0(params1, parte1)
                log_vero2 = -log_likelihood_g0(params2, parte2)
                 # Imprimir las log-verosimilitudes
                
                log_verosimilitud_total = log_vero1 + log_vero2
                print(f"Punto de cambio: {punto_cambio}, Log-vero1: {log_vero1:.4f}, Log-vero2: {log_vero2:.4f}, Log-vero Total: {log_verosimilitud_total:.4f}")
            
            
                if log_verosimilitud_total > mejor_log_verosimilitud:
                    mejor_log_verosimilitud = log_verosimilitud_total
                    mejor_punto_cambio = punto_cambio
            
            else:
                print(f"Fallo al estimar parámetros en el punto de cambio: {punto_cambio}")        
    
    mejor_punto_cambio_def=mejor_punto_cambio+inicio_ventana1
    
    print(f"Mejor Log Verosimilitud: {mejor_log_verosimilitud}")
    print(f"Punto de cambio: {mejor_punto_cambio}")
    print(f"Inicio Ventana 1: {inicio_ventana1}")
    print(f"Mejor punto de cambio: {mejor_punto_cambio_def}")
    return mejor_punto_cambio_def


    

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
segmento_data=segmento['amplitude'].values
ventanas = dividir_en_ventanas(segmento_data, ventana_size)

for i, ventana in enumerate(ventanas):
    params = calcular_parametros_g0(ventana, n_looks=1)
    
    if params is not None:
        alpha, n, gamma_scale = params # Solo desempaqueta si params no es None
        print(f"Ventana {i+1}: alpha = {alpha:.4f}, n = {n:.4f}, gamma = {gamma_scale:.4f}")
    else:
        print(f"Ventana {i+1}: Parámetros G0 no válidos, se omitirá el cálculo para esta ventana.")
     

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





# %%
