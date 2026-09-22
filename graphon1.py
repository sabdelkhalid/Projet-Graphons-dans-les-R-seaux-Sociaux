import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Définition des Graphons

def constant_graphon(x, y, p=0.5):
    return p

def block_graphon(x, y, blocks=[0.3, 0.7], values=[[0.8, 0.2], [0.2, 0.5]]):
    cumulative_blocks = np.cumsum([0] + blocks)
    block_i = next(i for i, b in enumerate(cumulative_blocks) if x < b) - 1
    block_j = next(j for j, b in enumerate(cumulative_blocks) if y < b) - 1
    return values[block_i][block_j]

def continuous_graphon(x, y):
    return x * y

# Génération de matrice d'adjacence à partir d'un Graphon

def generate_adjacency_matrix(graphon, n):
    points = np.random.uniform(0, 1, n)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            matrix[i, j] = matrix[j, i] = 1 if np.random.rand() < graphon(points[i], points[j]) else 0

    return matrix

# Exemple concret avec sauvegarde des visualisations matricielles

n = 100

# Exemple 1: Graphon constant
matrix1 = generate_adjacency_matrix(lambda x, y: constant_graphon(x, y, 0.3), n)
plt.imshow(matrix1, cmap='Greys', interpolation='none')
plt.title('Graphon Constant (p=0.3)')
plt.show()
plt.clf()

# Exemple 2: Graphon en blocs
matrix2 = generate_adjacency_matrix(lambda x, y: block_graphon(x, y), n)
plt.imshow(matrix2, cmap='Greys', interpolation='none')
plt.title('Graphon en Blocs')
plt.show()
plt.clf()

# Exemple 3: Graphon continu
matrix3 = generate_adjacency_matrix(continuous_graphon, n)
plt.imshow(matrix3, cmap='Greys', interpolation='none')
plt.title('Graphon Continu (x*y)')
plt.show()
plt.clf()
