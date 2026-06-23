import numpy as np
import matplotlib.pyplot as plt

#######################################################################################################
# Programa de clustering que utiliza el algoritmo K-Means con dos centroides.
# Centroides iniciales:
#   C1 = (2, 2)
#   C2 = (5, 6)
#######################################################################################################


def distancia_euclidiana(punto, centroide):
    return np.sqrt(np.sum((punto - centroide) ** 2))


def mostrar_centroides(titulo, centroides):
    print(f"\n>> {titulo}")
    for indice, centroide in enumerate(centroides, start=1):
        print(f"Centroide C{indice}: ({centroide[0]:.4f}, {centroide[1]:.4f})")


def mostrar_tabla(headers, filas):
    anchos = [
        max(len(str(valor)) for valor in [header] + [fila[indice] for fila in filas])
        for indice, header in enumerate(headers)
    ]

    encabezado = " ".join(str(header).rjust(anchos[indice]) for indice, header in enumerate(headers))
    separador = " ".join("-" * ancho for ancho in anchos)

    print(encabezado)
    print(separador)
    for fila in filas:
        print(" ".join(str(valor).rjust(anchos[indice]) for indice, valor in enumerate(fila)))


def calcular_k_means(datos, centroides_iniciales, max_iteraciones=100):
    centroides = centroides_iniciales.astype(float)
    etiquetas = np.zeros(len(datos), dtype=int)

    mostrar_centroides("Centroides iniciales", centroides)

    for iteracion in range(1, max_iteraciones + 1):
        print(f"\n{'=' * 80}")
        print(f">> Iteracion {iteracion}")
        print(f"{'=' * 80}")

        nuevas_etiquetas = []

        for indice, punto in enumerate(datos, start=1):
            distancias = [distancia_euclidiana(punto, centroide) for centroide in centroides]
            etiqueta = int(np.argmin(distancias))
            nuevas_etiquetas.append(etiqueta)

            calculos = []
            for centroide_indice, centroide in enumerate(centroides, start=1):
                calculos.append(
                    f"d(P{indice}, C{centroide_indice}) = "
                    f"sqrt(({punto[0]} - {centroide[0]:.4f})^2 + "
                    f"({punto[1]} - {centroide[1]:.4f})^2) = "
                    f"{distancias[centroide_indice - 1]:.4f}"
                )

            print(f"P{indice} = ({punto[0]}, {punto[1]})")
            print("  " + " | ".join(calculos))
            print(f"  Cluster asignado: C{etiqueta + 1}")

        nuevas_etiquetas = np.array(nuevas_etiquetas)
        nuevos_centroides = centroides.copy()

        for cluster in range(len(centroides)):
            puntos_cluster = datos[nuevas_etiquetas == cluster]
            if len(puntos_cluster) > 0:
                nuevos_centroides[cluster] = puntos_cluster.mean(axis=0)

        mostrar_centroides("Nuevos centroides", nuevos_centroides)

        if np.array_equal(etiquetas, nuevas_etiquetas) and np.allclose(centroides, nuevos_centroides):
            print("\n>> Los centroides no cambiaron. El algoritmo ha convergido.")
            etiquetas = nuevas_etiquetas
            centroides = nuevos_centroides
            break

        etiquetas = nuevas_etiquetas
        centroides = nuevos_centroides

    return etiquetas, centroides


#######################################################################################################
# 1.- DATOS INICIALES
#######################################################################################################
X1 = np.array([1, 2, 3, 3, 4, 4, 6, 6])
X2 = np.array([1, 4, 2, 5, 4, 7, 4, 6])
X = np.array(list(zip(X1, X2))).reshape(len(X1), 2)

print(f"\n>> Numero de filas: {X.shape[0]}, Numero de columnas: {X.shape[1]}\n")
print(">> Datos iniciales:\n")
mostrar_tabla(["X1", "X2"], X.tolist())

#######################################################################################################
# 2.- K-MEANS CON DOS CENTROIDES
#######################################################################################################
centroides_iniciales = np.array([[2, 2], [5, 6]])
labels, centroides = calcular_k_means(X, centroides_iniciales)

print("\n\n>> Resultado final con dos centroides:\n")
mostrar_centroides("Centroides finales", centroides)

print("\n>> Cluster asignado a cada punto (X1, X2):")
resultado_final = [[int(punto[0]), int(punto[1]), int(label + 1)] for punto, label in zip(X, labels)]
mostrar_tabla(["X1", "X2", "Cluster"], resultado_final)

#######################################################################################################
# 3.- VISUALIZACION DE DATOS AGRUPADOS
#######################################################################################################
colores = ["red", "green"]
asignar = [colores[label] for label in labels]

plt.scatter(X1, X2, c=asignar, s=40)
plt.scatter(centroides[:, 0], centroides[:, 1], marker="*", c="black", s=80)
plt.xlabel("X1")
plt.ylabel("X2")
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.grid()
plt.minorticks_on()
plt.grid(visible=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
plt.title("K-Means clustering con dos centroides")
plt.show()
