import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.utils import from_networkx


class User:
    def __init__(self, id, age=None, city=None, interests=None, activity=1.0):
        self.id = id
        self.age = age if age is not None else np.random.randint(15, 55)
        self.city = city if city is not None else np.random.choice(['Paris', 'Lyon', 'Marseille'])
        self.interests = interests if interests is not None else np.random.uniform(0, 1, size=3)
        self.activity = activity
        self.friends = set()
    def add_friend(self, other):
        self.friends.add(other.id)
    def __repr__(self):
        return f"User(id={self.id}, age={self.age}, city={self.city}, act={self.activity:.2f})"

n = 40  # nombre d'utilisateurs
users = [User(i) for i in range(n)]

def affinity(u1, u2):
    sim = np.dot(u1.interests, u2.interests) / (np.linalg.norm(u1.interests) * np.linalg.norm(u2.interests))
    age_bonus = 0.2 if abs(u1.age - u2.age) < 5 else 0
    city_bonus = 0.3 if u1.city == u2.city else 0
    p = 0.15 * sim + 0.4 * (u1.activity + u2.activity)/2 + age_bonus + city_bonus
    return min(max(p, 0), 1)

A = np.zeros((n, n), dtype=int)
for i in range(n):
    for j in range(i+1, n):
        if np.random.rand() < affinity(users[i], users[j]):
            A[i, j] = 1
            A[j, i] = 1
            users[i].add_friend(users[j])
            users[j].add_friend(users[i])
np.fill_diagonal(A, 0)

# 2. Affichage de la matrice d’adjacence 
plt.figure(figsize=(5,5))
plt.imshow(A, cmap='Greys', interpolation='none')
plt.title("Matrice d'adjacence sociale (User <-> Noeud)")
plt.xlabel("Utilisateur (Noeud j)")
plt.ylabel("Utilisateur (Noeud i)")
plt.show()


# Suppose que A est ta matrice d'adjacence (numpy) et n le nombre de nœuds
G = nx.from_numpy_array(A)
data = from_networkx(G)

n = A.shape[0]
seed = 0

x = torch.zeros((n, 1))
x[seed] = 1.0
data.x = x

class RumorSpreadGNN(nn.Module):
    def __init__(self, layers=5):
        super().__init__()
        self.layers = layers

    def forward(self, data):
        x = data.x.clone()
        edge_index = data.edge_index
        all_states = [x.clone()]
        for l in range(self.layers):
            x_new = x.clone()
            for i in range(x.shape[0]):
                neighbors = edge_index[1][edge_index[0]==i]
                if len(neighbors) > 0:
                    x_new[i] = torch.max(x[neighbors])
                if x[i] > 0:
                    x_new[i] = x[i]
            # Test de convergence
            if torch.equal(x_new, x):
                print(f"\nPropagation terminée à l'étape {l+1} : plus aucun nouveau nœud atteint.")
                all_states.append(x_new.clone())
                break
            x = x_new
            all_states.append(x.clone())
        return x, all_states








def plot_gnn_propagation_newonly(A, all_states, pause=1):
    n = A.shape[0]
    for t in range(1, len(all_states)):
        prev = set(np.where(all_states[t-1].view(-1).numpy() > 0)[0])
        curr = set(np.where(all_states[t].view(-1).numpy() > 0)[0])
        new_nodes = curr - prev

        plt.figure(figsize=(5,5))
        plt.imshow(A, cmap="Greys", interpolation="none")
        if new_nodes:
            plt.scatter(list(new_nodes), list(new_nodes), color="red", s=70, marker="o", label="nouveaux atteints")
        plt.title(f"Étape {t} : nouveaux atteints (GNN)")
        plt.xlabel("Noeud j")
        plt.ylabel("Noeud i")
        plt.legend()
        plt.show()
        plt.pause(pause)
  
# Création et propagation GNN
model = RumorSpreadGNN(layers=10)
final_x, all_states = model(data)



n = A.shape[0]
already_reached = set(np.where(all_states[0].view(-1).numpy() > 0)[0])
print(f"Départ : {sorted(already_reached)}")

for t in range(1, len(all_states)):
    prev = set(np.where(all_states[t-1].view(-1).numpy() > 0)[0])
    curr = set(np.where(all_states[t].view(-1).numpy() > 0)[0])
    new_nodes = curr - prev
    if not new_nodes:
        print(f"Étape {t}: plus aucun nouveau atteint, propagation terminée.")
        break
    print(f"\nÉtape {t}:")
    print("Nouveaux atteints :", sorted(new_nodes))
    print("Total atteints    :", len(curr))
    print("Liste totale      :", sorted(curr))


# Visualisation étape par étape
plot_gnn_propagation_newonly(A, all_states, pause=1)


