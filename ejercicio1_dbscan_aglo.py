# 01_aglomerativo_distancia_simple.py
# Ejercicio 1: Clustering jerárquico aglomerativo con enlace/distancia simple
# Datos tomados de la matriz de distancias entre A, B, C, D, E, F y G.

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

# Etiquetas de los 7 individuos
etiquetas = ["A", "B", "C", "D", "E", "F", "G"]

# Matriz completa de distancias.
# La matriz del enunciado está dada en forma triangular inferior;
# aquí se coloca de forma simétrica para que Python pueda procesarla.
matriz_distancias = np.array([
    [0.00, 2.15, 0.70, 1.07, 0.85, 1.16, 1.56],
    [2.15, 0.00, 1.53, 1.14, 1.38, 1.01, 2.83],
    [0.70, 1.53, 0.00, 0.43, 0.21, 0.55, 1.86],
    [1.07, 1.14, 0.43, 0.00, 0.29, 0.22, 2.04],
    [0.85, 1.38, 0.21, 0.29, 0.00, 0.41, 2.02],
    [1.16, 1.01, 0.55, 0.22, 0.41, 0.00, 2.05],
    [1.56, 2.83, 1.86, 2.04, 2.02, 2.05, 0.00]
])

# linkage no recibe una matriz cuadrada directamente, sino un vector condensado.
distancias_condensadas = squareform(matriz_distancias)

# Clustering jerárquico aglomerativo con enlace simple.
resultado = linkage(distancias_condensadas, method="single")

print("Matriz linkage:")
print("columna 1, columna 2, distancia de unión, número de elementos")
print(resultado)

print("\nPasos de fusión:")
clusters = {i: etiquetas[i] for i in range(len(etiquetas))}

for paso, fila in enumerate(resultado, start=1):
    i, j, distancia, cantidad = fila
    i = int(i)
    j = int(j)

    nuevo_cluster = f"({clusters[i]} + {clusters[j]})"
    clusters[len(etiquetas) + paso - 1] = nuevo_cluster

    print(f"Paso {paso}: unir {clusters[i]} con {clusters[j]} "
          f"a distancia {distancia:.2f}. Nuevo cluster: {nuevo_cluster}")

# Dendrograma
plt.figure(figsize=(8, 5))
dendrogram(resultado, labels=etiquetas)
plt.title("Dendrograma - Clustering jerárquico aglomerativo")
plt.xlabel("Individuos")
plt.ylabel("Distancia")
plt.grid(True)
plt.show()
