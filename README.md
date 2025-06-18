# Tesis
Tesis Maestría Ciencia de Datos

INFORMACIÓN
=> señal_sintética => contiene tres archivos creados con señales sintéticas con cambios evidentes en los parámetros de la distribución para probar el algoritmo de Juliana 
Contiene tres archivos con señales de 2 segundos con Fs = 250hz cuyos datos responden a una distribución normal cuya media cambia en forma abrupta
en la mitad de la señal, en el primer tercio y el primer cuarto respectivamente.

SCRIPTS:

=> 9_Señal_Prueba => el objetivo es generar una señal 100% artificial en el que haya un cambio abrupto evidente en la mitad de la señal para probar el código de detección de bordes.

=> 9_Señal_Prueba_G0, 9_Señal_Prueba_Gamma, 9_Señal_Prueba_Normal => es el código que intenta replicar el método de detección de Juliana. Cada uno corresponde a una distribución diferente. 
OJO!!! fueron desarrollados para una señal sintética. Sirven para otras señales, pero habría que modificar el código a fin de agregar las funciones de filtrado y transformación de señales previo a la aplicación del método.
