import numpy as np
import matplotlib.pyplot as plt

# 1. Définir la classe User
class User:
    def __init__(self, id, interests=None, activity=1.0):
        self.id = id
        self.interests = interests if interests is not None else np.random.uniform(0, 1, size=3)
        self.activity = activity
        self.friends = set()
    def add_friend(self, other):
        self.friends.add(other.id)
    def __repr__(self):
        return f"User(id={self.id}, activity={self.activity:.2f}, interests={self.interests})"

# 2. Créer la population d'utilisateurs
n = 40
users = [User(i) for i in range(n)]
for u in users:
    u.interests = np.random.uniform(0, 1, size=3)
    u.activity = np.random.uniform(0.2, 1)

# 3. Générer la matrice d'adjacence selon une règle sociale (ici, affinité)
def affinity(u1, u2):
    sim = np.dot(u1.interests, u2.interests) / (np.linalg.norm(u1.interests) * np.linalg.norm(u2.interests))
    p = 0.25 * sim + 0.5 * (u1.activity + u2.activity) / 2
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

# 4. Représenter la matrice (noir = lien d’amitié)
plt.figure(figsize=(4,4))
plt.imshow(A, cmap='Greys', interpolation='none')
plt.title("Matrice d'adjacence sociale\n(User <-> Noeud)")
plt.xlabel("Utilisateur (Noeud j)")
plt.ylabel("Utilisateur (Noeud i)")
plt.show()

# 5. Pour avoir la correspondance User <-> Noeud (exemple)
for i in range(3):  # affiche juste les 3 premiers
    print(f"Noeud {i} = {users[i]}")
    print(f"Amis de {i}:", users[i].friends)
