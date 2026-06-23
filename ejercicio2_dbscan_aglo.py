# 02_dbscan_8_puntos_figura.py
# Ejercicio 2: DBSCAN para los 8 puntos de la figura
# Parámetros del enunciado: min_samples = 2 y eps desde sqrt(2) hasta sqrt(10).
# Cambia valor_final_radical si quieres probar radios mayores, por ejemplo sqrt(45).

from math import ceil
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Puntos leídos de la figura.
# Se usan las etiquetas A1, A2, ..., A8.
etiquetas = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]

puntos = np.array([
    [2, 10],  # A1
    [2, 5],   # A2
    [8, 4],   # A3
    [5, 8],   # A4
    [7, 5],   # A5
    [6, 4],   # A6
    [1, 2],   # A7
    [4, 9]    # A8
], dtype=float)

min_samples = 2
valor_inicial_radical = 2
valor_final_radical = 10
valores_radical = list(range(valor_inicial_radical, valor_final_radical + 1))

print("Nota: en sklearn, min_samples incluye al propio punto.")
print("Por eso min_samples=2 significa: el punto + al menos 1 vecino dentro del radio.\n")

n_columnas = 3
n_filas = ceil(len(valores_radical) / n_columnas)
fig, axes = plt.subplots(n_filas, n_columnas,
                         figsize=(16, 4.3 * n_filas),
                         squeeze=False)
axes = axes.ravel()

# Probar eps = sqrt(valor_inicial_radical), ..., sqrt(valor_final_radical)
for indice, valor in enumerate(valores_radical):
    eps = np.sqrt(valor)
    modelo = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = modelo.fit_predict(puntos)

    core_indices = set(modelo.core_sample_indices_)
    frontera = [etiquetas[i] for i in range(len(puntos))
                if clusters[i] != -1 and i not in core_indices]
    centrales = [etiquetas[i] for i in core_indices]
    outliers = [etiquetas[i] for i in range(len(puntos)) if clusters[i] == -1]

    agrupacion = defaultdict(list)
    for etiqueta, cluster in zip(etiquetas, clusters):
        if cluster == -1:
            agrupacion["Ruido / outlier"].append(etiqueta)
        else:
            agrupacion[f"Cluster {cluster}"].append(etiqueta)

    print("=" * 60)
    print(f"eps = sqrt({valor}) = {eps:.4f}")
    print("Clusters encontrados:")
    for nombre_cluster, puntos_cluster in agrupacion.items():
        print(f"  {nombre_cluster}: {puntos_cluster}")

    print(f"Puntos centrales: {centrales}")
    print(f"Puntos frontera: {frontera}")
    print(f"Outliers: {outliers}")

    # Gráfico para este valor de eps
    ax = axes[indice]
    ax.scatter(puntos[:, 0], puntos[:, 1], c=clusters, cmap="viridis",
               marker="o", edgecolor="k", s=120)

    for i, etiqueta in enumerate(etiquetas):
        ax.annotate(etiqueta, (puntos[i, 0] + 0.1, puntos[i, 1] + 0.1))

    ax.set_title(f"eps = sqrt({valor})", pad=8)
    if indice // n_columnas == n_filas - 1:
        ax.set_xlabel("Coordenada X")
    if indice % n_columnas == 0:
        ax.set_ylabel("Coordenada Y")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.grid(True)

for ax in axes[len(valores_radical):]:
    ax.set_visible(False)

fig.suptitle("DBSCAN - 8 puntos de la figura", fontsize=16, y=0.995)
fig.subplots_adjust(left=0.06, right=0.98, bottom=0.07, top=0.93,
                    hspace=0.35, wspace=0.18)
plt.show()
