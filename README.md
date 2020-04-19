Predictor de clases KNN, basado en Arrays Binarios de caracteristicas geometricas de imagenes segmentadas
Para correr, por favor utilice:

.\python main.py

Si borra los datasets CSV de ./src , el programa creara nuevamente esta informacion a partir de las imagenes originales
funciones de segmentacion de imagen basadas en metodo de contornos al final del script


El ejercicio consiste de una funcion predictor() que recibe una imagen y retorna prediccion sobre que letra es, 
basandose en metodos de KNN con N=1, y los 4 primeros Momentos de Hu.
Estos hiperparametros fueron los con mejor resultado, probando combinaciones de K=(1...40), 7 Momentos de Hu, 
Roundness,
Area y Perimetro