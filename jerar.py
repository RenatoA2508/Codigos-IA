import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
# Generar datos aleatorios
np.random.seed(42)
data = np.random.randn(10, 2)
# Aplicar clustering jerárquico aglomerativo
linked = linkage(data, 'single') # Usamos enlace simple como ejemplo
# Visualizar los puntos de la matriz
plt.figure(figsize=(5, 3))
plt.scatter(data[:, 0], data[:, 1], c='blue', marker='o', label='Puntos de datos')
for i, point in enumerate(data):
    plt.annotate(f'Punto {i+1}', (point[0]+0.1, point[1]-0.1))
plt.title("Puntos de datos generados aleatoriamente")
plt.xlabel("Coordenada X")
plt.ylabel("Coordenada Y")
plt.legend()
plt.grid(True)
plt.show()
# Crear etiquetas "Punto 1", "Punto 2", etc. para el dendrograma
labels = [f"P{i+1}" for i in range(data.shape[0])]
# Visualizar con un dendrograma
plt.figure(figsize=(5, 3))
dendrogram(linked, orientation='top', labels=labels,
distance_sort='descending', show_leaf_counts=True)
plt.title("Dendrograma de Clustering Jerárquico Aglomerativo")
plt.xlabel("Puntos de datos")
plt.ylabel("Distancia")
plt.show()