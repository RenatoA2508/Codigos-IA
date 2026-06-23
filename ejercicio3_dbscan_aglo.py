# 03_dbscan_conjunto_bidimensional.py
# Ejercicio 3: DBSCAN para el conjunto bidimensional dado
# Parámetros del enunciado: eps = 2 y min_samples = 2.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from collections import defaultdict

etiquetas = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]

puntos = np.array([
    [1, 2],  # P1
    [1, 3],  # P2
    [2, 2],  # P3
    [2, 3],  # P4
    [4, 6],  # P5
    [8, 8],  # P6
    [8, 9],  # P7
    [8, 5],  # P8
    [8, 7],  # P9
    [7, 6]   # P10
], dtype=float)

eps = 2
min_samples = 2

modelo = DBSCAN(eps=eps, min_samples=min_samples)
clusters = modelo.fit_predict(puntos)

core_indices = set(modelo.core_sample_indices_)
puntos_centrales = [etiquetas[i] for i in core_indices]
puntos_frontera = [etiquetas[i] for i in range(len(puntos))
                   if clusters[i] != -1 and i not in core_indices]
outliers = [etiquetas[i] for i in range(len(puntos)) if clusters[i] == -1]

agrupacion = defaultdict(list)
for etiqueta, cluster in zip(etiquetas, clusters):
    if cluster == -1:
        agrupacion["Ruido / outlier"].append(etiqueta)
    else:
        agrupacion[f"Cluster {cluster}"].append(etiqueta)

print("Resultado DBSCAN")
print(f"eps = {eps}")
print(f"min_samples = {min_samples}")
print("\nClusters encontrados:")
for nombre_cluster, puntos_cluster in agrupacion.items():
    print(f"  {nombre_cluster}: {puntos_cluster}")

print(f"\nPuntos centrales: {puntos_centrales}")
print(f"Puntos frontera: {puntos_frontera}")
print(f"Outliers: {outliers}")

# Gráfico y resumen en una sola ventana
fig, (ax_grafico, ax_resumen) = plt.subplots(
    1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [2, 1]}
)

ax_grafico.scatter(puntos[:, 0], puntos[:, 1], c=clusters, cmap="viridis",
                   marker="o", edgecolor="k", s=120)

for i, etiqueta in enumerate(etiquetas):
    tipo = "central" if i in core_indices else ("outlier" if clusters[i] == -1 else "frontera")
    ax_grafico.annotate(f"{etiqueta}\n{tipo}",
                        (puntos[i, 0] + 0.1, puntos[i, 1] + 0.1))

ax_grafico.set_title("DBSCAN - Conjunto bidimensional", pad=10)
ax_grafico.set_xlabel("Coordenada X")
ax_grafico.set_ylabel("Coordenada Y")
ax_grafico.grid(True)

texto_resultados = [
    "Resultado DBSCAN",
    f"eps = {eps}",
    f"min_samples = {min_samples}",
    "",
    "Clusters encontrados:",
]

for nombre_cluster, puntos_cluster in agrupacion.items():
    texto_resultados.append(f"{nombre_cluster}: {puntos_cluster}")

texto_resultados.extend([
    "",
    f"Puntos centrales: {puntos_centrales}",
    f"Puntos frontera: {puntos_frontera}",
    f"Outliers: {outliers}",
])

ax_resumen.axis("off")
ax_resumen.set_title("Resumen", pad=10)
ax_resumen.text(0, 1, "\n".join(texto_resultados), va="top", fontsize=10)

fig.tight_layout()
plt.show()
