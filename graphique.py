import pandas as pd
# 1. Imports, Classe User, Génération du réseau
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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

# 3. Simulation de propagation
def simulate_propagation(A, seed=0, steps=8):
    n = A.shape[0]
    reached = set([seed])
    newly_reached = set([seed])
    propagation_steps = [set([seed])]
    for t in range(steps):
        next_new = set()
        for node in newly_reached:
            neighbors = set(np.where(A[node] == 1)[0])
            next_new |= (neighbors - reached)
        if not next_new:
            break
        newly_reached = next_new
        reached |= newly_reached
        propagation_steps.append(set(newly_reached))
    return propagation_steps

propagation = simulate_propagation(A, seed=0, steps=8)

def plot_propagation_matrix(A, propagation_steps):
    n = A.shape[0]
    for t, nodes in enumerate(propagation_steps):
        plt.figure(figsize=(5,5))
        plt.imshow(A, cmap="Greys", interpolation="none")
        plt.scatter(list(nodes), list(nodes), color="red", s=70, marker="o")
        plt.title(f"Étape {t} : propagation info")
        plt.show()

plot_propagation_matrix(A, propagation)

# 4. Visualisation réseau (networkx)
def plot_network(users, A, propagation_steps, step=0):
    G = nx.Graph()
    for i in range(len(users)):
        G.add_node(i)
    for i in range(len(users)):
        for j in range(i+1, len(users)):
            if A[i, j]:
                G.add_edge(i, j)
    color_map = []
    reached = set()
    for t in range(min(step+1, len(propagation_steps))):
        reached |= propagation_steps[t]

    for i in range(len(users)):
        color_map.append('red' if i in reached else 'grey')
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6,6))
    nx.draw(G, pos, node_color=color_map, with_labels=True)
    plt.title(f"Réseau social, propagation jusqu'à l'étape {step}")
    plt.show()

# Exemple : propagation jusqu’à l’étape 3
plot_network(users, A, propagation, step=3)

# 5. Analyses réseau
G = nx.from_numpy_array(A)
degree_sequence = [d for n, d in G.degree()]
print("Degré moyen :", np.mean(degree_sequence))

centrality = nx.degree_centrality(G)
top_users = sorted(centrality.items(), key=lambda x: -x[1])[:5]
print("Top utilisateurs centraux :", top_users)

# Détection de communautés (algorithme de Girvan-Newman)
from networkx.algorithms.community import girvan_newman
communities = next(girvan_newman(G))
print("Premières communautés détectées :", [list(c) for c in communities])

# 6. Extension rapide : voir un utilisateur et ses attributs
for i in range(3):
    print(users[i])
    print(f"Amis de {i}:", users[i].friends)



def plot_network_by_activity(users, A):
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.from_numpy_array(A)
    activity = [u.activity for u in users]
    # Normaliser l’activité entre 0.2 et 1 pour une couleur plus visible
    norm_activity = [(a-0.2)/(1-0.2) for a in activity]
    # Mapper sur une palette de couleur (du bleu au rouge)
    colors = [plt.cm.coolwarm(a) for a in norm_activity]
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6,6))
    nx.draw(G, pos, node_color=colors, with_labels=True)
    plt.title("Graphe social (colorié par activité)")
    plt.show()

plot_network_by_activity(users, A)



def plot_network_by_distance(users, A, source=0):
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.from_numpy_array(A)
    # Distance de chaque noeud à la source
    lengths = nx.single_source_shortest_path_length(G, source)
    max_dist = max(lengths.values())
    colors = []
    for i in range(len(users)):
        d = lengths.get(i, max_dist+1)
        # plus c'est proche de la source, plus c'est rouge
        colors.append(plt.cm.Reds(1 - d/(max_dist+1)))
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6,6))
    nx.draw(G, pos, node_color=colors, with_labels=True)
    plt.title(f"Graphe social (distance à la source {source})")
    plt.show()


plot_network_by_distance(users, A, source=0)  # ou n'importe quel index de user



def plot_network_by_community(users, A, communities):
    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.from_numpy_array(A)
    # On donne une couleur différente à chaque communauté
    palette = plt.cm.tab10.colors
    node_colors = [None]*len(users)
    for k, com in enumerate(communities):
        for idx in com:
            node_colors[idx] = palette[k%len(palette)]
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(6,6))
    nx.draw(G, pos, node_color=node_colors, with_labels=True)
    plt.title("Graphe social (colorié par communauté)")
    plt.show()


from networkx.algorithms.community import girvan_newman
communities = next(girvan_newman(nx.from_numpy_array(A)))
plot_network_by_community(users, A, communities)

def get_suggestions(user_id, users, A, top_k=3):
    """Propose top_k suggestions pour user_id (non amis)"""
    n = len(users)
    friends = users[user_id].friends
    # Score par similarité d'intérêt (hors déjà amis et soi-même)
    suggestions = []
    for j in range(n):
        if j != user_id and j not in friends:
            sim = np.dot(users[user_id].interests, users[j].interests) / (
                np.linalg.norm(users[user_id].interests) * np.linalg.norm(users[j].interests)
            )
            common_friends = len(friends & users[j].friends)
            suggestions.append((j, sim, common_friends))
    suggestions.sort(key=lambda x: (x[2], x[1]), reverse=True)  # tri sur amis communs puis similarité
    return suggestions[:top_k]

# Exemple d’affichage suggestions façon “feed social”
def display_suggestions(user_id, users, A):
    sugg = get_suggestions(user_id, users, A, top_k=3)
    print(f"Suggestions pour {users[user_id]} :")
    for (j, sim, common) in sugg:
        print(f" - {users[j].id} | {users[j].city} | {common} amis en commun | Score intérêts: {sim:.2f}")

display_suggestions(0, users, A)



def simulate_propagation_terminal(A, seed=0, steps=8):
    n = A.shape[0]
    reached = set([seed])
    newly_reached = set([seed])
    propagation_steps = [set([seed])]
    print(f"Départ : noeud {seed}")
    for t in range(steps):
        next_new = set()
        for node in newly_reached:
            neighbors = set(np.where(A[node] == 1)[0])
            next_new |= (neighbors - reached)
        if not next_new:
            print(f"---\nÉtape {t+1} : plus aucun nouveau atteint, propagation terminée.")
            break
        newly_reached = next_new
        reached |= newly_reached
        propagation_steps.append(set(newly_reached))
        print(f"\nÉtape {t+1} :")
        print("Nouveaux atteints :", sorted(newly_reached))
        print("Total atteints    :", len(reached))
        print("Liste totale      :", sorted(reached))
    print("\nPropagation terminée.")
    return propagation_steps

# Exemple d’utilisation (A est la matrice d’adjacence)
simulate_propagation_terminal(A, seed=0, steps=10)




# Grille 2D
G_physique = nx.grid_2d_graph(6, 6)
mapping = {node: i for i, node in enumerate(G_physique.nodes())}
G_physique = nx.relabel_nodes(G_physique, mapping)
A_physique = nx.to_numpy_array(G_physique)
simulate_propagation_terminal(A_physique, seed=0, steps=15)
