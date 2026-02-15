import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.metrics import classification_report


# -----------------------
# Load processed data
# -----------------------
base_dir = os.path.join("data", "processed")
labels = np.load(os.path.join(base_dir, "corr_labels.npy"))
edge_index = np.load(os.path.join(base_dir, "corr_edge_index.npy"))

num_nodes = len(labels)

# Node features فعلاً ساده: یک ویژگی ثابت (بعداً بهترش می‌کنیم)
features = np.load(os.path.join(base_dir, "corr_features.npy"))
x = torch.tensor(features, dtype=torch.float)

edge_index = torch.tensor(edge_index, dtype=torch.long)
y = torch.tensor(labels, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)


# -----------------------
# Define GAT Model (Professional)
# -----------------------
class GAT(torch.nn.Module):
    def __init__(self, in_channels=3, hidden=8, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(in_channels, hidden, heads=heads, dropout=dropout)
        # خروجی لایه اول: hidden * heads
        self.gat2 = GATConv(hidden * heads, 2, heads=1, concat=False, dropout=dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x


model = GAT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# -----------------------
# Class weights for imbalance
# -----------------------
class_counts = torch.bincount(data.y)
# جلوگیری از تقسیم بر صفر
class_counts = torch.clamp(class_counts, min=1)
weights = (1.0 / class_counts.float()).to(device)


# -----------------------
# Train Loop
# -----------------------
model.train()
for epoch in range(201):
    optimizer.zero_grad()
    out = model(data)

    loss = F.cross_entropy(out, data.y, weight=weights)

    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        pred = out.argmax(dim=1)
        correct = (pred == data.y).sum().item()
        acc = correct / num_nodes
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")


# -----------------------
# Evaluation
# -----------------------
model.eval()
with torch.no_grad():
    out = model(data)
    pred = out.argmax(dim=1).detach().cpu().numpy()
    true = data.y.detach().cpu().numpy()

print("\nClassification Report (GAT):")
print(classification_report(true, pred, digits=4))
