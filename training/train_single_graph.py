import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.metrics import classification_report
# -----------------------
# Load processed data
# -----------------------
base_dir = os.path.join("data", "processed")

labels = np.load(os.path.join(base_dir, "corr_labels.npy"))
edge_index = np.load(os.path.join(base_dir, "corr_edge_index.npy"))

# Node features ساده: فعلاً فقط یک ویژگی 1 ثابت
# (بعداً می‌تونیم ویژگی‌های بهتر اضافه کنیم)
num_nodes = len(labels)
x = torch.ones((num_nodes, 1), dtype=torch.float)

edge_index = torch.tensor(edge_index, dtype=torch.long)
y = torch.tensor(labels, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = data.to(device)

# -----------------------
# Define GCN Model
# -----------------------
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(1, 16)
        self.conv2 = GCNConv(16, 2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

model = GCN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -----------------------
# Train Loop
# -----------------------
model.train()
for epoch in range(201):
    optimizer.zero_grad()
    out = model(data)
    loss = F.cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        pred = out.argmax(dim=1)
        correct = (pred == data.y).sum().item()
        acc = correct / num_nodes
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")



model.eval()
with torch.no_grad():
    out = model(data)
    pred = out.argmax(dim=1).cpu().numpy()
    true = data.y.cpu().numpy()

print("\nClassification Report:")
print(classification_report(true, pred, digits=4))
