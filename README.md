
# Prédictions via les Graphons dans les Réseaux Sociaux

**Projet ingénieur – Université Paris Cité, EIDD**  
Auteurs :  
- OUASSIL Mohammed Nassr Allah  
- ROBAII Oussama  
- Saad ABDELKHALID  

---

## 1. Sujet

Ce projet explore la modélisation et l’analyse des réseaux sociaux via les **graphons** :
- Génération de graphes à partir de modèles probabilistes (graphon constant, à blocs, continu, affinity)
- Simulation de la propagation d’informations/rumeurs
- Suggestions d’amis
- **Propagation via GNN** (Graph Neural Network)
- **Backtesting** sur un vrai réseau de citations scientifiques (Cora)

---

## 2. Organisation des codes

- `graphon1.py` : Génération de matrices d’adjacence à partir de différents graphons et visualisation.
- `graphique.py` : Génération d’un réseau social synthétique, propagation classique, suggestions d’amis, analyses, visualisation réseau.
- `graphon_users.py` : Génération d’un réseau basé sur la règle d’affinité utilisateur.
- `gnn.py` : Simulation de la propagation de rumeur via un GNN, visualisation étape par étape (terminal et matrice).
- `backtesting_gnn_cora.py` : Backtesting sur données réelles Cora (classification de nœuds par GCN).
- Présentation : `PROJET_GRAPHON1.pdf` (slides du projet)

---

## 3. Installation et exécution

### Dépendances

Installer les librairies nécessaires :

pip install numpy pandas matplotlib networkx torch torch-geometric scikit-learn
```
(Le script de backtesting télécharge automatiquement les données Cora.)

### Exécution des scripts

- **Génération, analyse et propagation classique :**
  
  python graphique.py
  ```
- **Propagation via GNN (avec visualisation avancée) :**
  
  python gnn.py
  ```
- **Graphons et visualisation de matrices :**
  
  python graphon1.py
  ```
- **Propagation sur réseau “affinity” :**
  
  python graphon_users.py
  ```
- **Backtesting (Cora) :**
  
  python backtesting_gnn_cora.py
  ```

---

## 4. Visualisation avancée de la propagation GNN

Après la soutenance, une visualisation avancée a été ajoutée :
- **Affichage terminal** à chaque étape : nouveaux atteints, total, propagation dynamique.
- **Affichage graphique** : matrice d’adjacence, nouveaux atteints en couleur (étape par étape).

---

## 5. Backtesting sur données réelles (Cora)

Ajout de `backtesting_gnn_cora.py`:
- Téléchargement et prétraitement automatiques du dataset
- Entraînement d’un GCN pour la classification de nœuds (articles scientifiques)
- Scores de validation/test (précision, rappel, F1-score) affichés dans le terminal
