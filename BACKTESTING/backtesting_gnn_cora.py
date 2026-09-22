
# Installer les dépendances (exécute cette ligne si nécessaire, sur Google Colab par exemple)
# !pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.metrics import classification_report
import pandas as pd

# Chargement du dataset Cora
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Résumé des données
summary = pd.DataFrame({
    "Nombre de nœuds": [data.num_nodes],
    "Nombre d’arêtes": [data.num_edges],
    "Nombre de classes": [dataset.num_classes],
    "Features par nœud": [data.num_node_features],
    "Nœuds en train": [int(data.train_mask.sum())],
    "Nœuds en validation": [int(data.val_mask.sum())],
    "Nœuds en test": [int(data.test_mask.sum())],
})
print("Résumé des données Cora:")
print(summary)

# Définition du modèle GCN
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

for epoch in range(200):
    train()

model.eval()
out = model(data)
pred = out.argmax(dim=1)

val_true = data.y[data.val_mask].cpu()
val_pred = pred[data.val_mask].cpu()
test_true = data.y[data.test_mask].cpu()
test_pred = pred[data.test_mask].cpu()

print("=== VALIDATION ===")
print(classification_report(val_true, val_pred, digits=3))
print("=== TEST FINAL ===")
print(classification_report(test_true, test_pred, digits=3))
