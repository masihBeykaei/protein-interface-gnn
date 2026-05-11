import os
import random
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import classification_report


# -----------------------
# Config
# -----------------------
SEED = 42
EPOCHS = 201
LR = 0.005
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 2
NEGATIVE_RATIO = 5

PROCESSED_DIR = os.path.join("data", "processed")

CASES = [
    "1BRS_A_B",
    "1FSS_A_B",
    "1AHW_AB_C",
    "1DQJ_AB_C",
    "1E6J_HL_P",
    "1JPS_HL_T",
    "1MLC_AB_E",
    "1WEJ_HL_F",
    "2FD6_HL_U",
    "2VIS_AB_C",
    "3HMX_LH_AB",
    "3MJ9_HL_A",
]


# -----------------------
# Reproducibility
# -----------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# -----------------------
# Load dataset
# -----------------------
def load_graph(case_name):
    features_path = os.path.join(PROCESSED_DIR, f"{case_name}_corr_features.npy")
    labels_path = os.path.join(PROCESSED_DIR, f"{case_name}_corr_labels.npy")
    edge_path = os.path.join(PROCESSED_DIR, f"{case_name}_corr_edge_index.npy")

    if not (
        os.path.exists(features_path)
        and os.path.exists(labels_path)
        and os.path.exists(edge_path)
    ):
        print(f"Skipping {case_name}: missing processed files.")
        return None

    x = np.load(features_path)
    y = np.load(labels_path)
    edge_index = np.load(edge_path)

    if len(y) == 0:
        print(f"Skipping {case_name}: empty graph.")
        return None

    positive = int(y.sum())

    if positive == 0:
        print(f"Skipping {case_name}: no positive nodes.")
        return None

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
    )

    data.case_name = case_name
    return data


dataset = []

for case in CASES:
    graph = load_graph(case)

    if graph is not None:
        dataset.append(graph)


print("\nLoaded graphs:")

for data in dataset:
    print(
        f"{data.case_name}: nodes={data.num_nodes}, "
        f"edges={data.edge_index.shape[1]}, "
        f"positive={int(data.y.sum())}, "
        f"negative={int(data.num_nodes - data.y.sum())}, "
        f"feature_dim={data.x.shape[1]}"
    )


if len(dataset) < 2:
    raise RuntimeError("Not enough graphs loaded for multi-graph training.")


# -----------------------
# Train/Test split by graph
# -----------------------
random.shuffle(dataset)

split_idx = int(0.8 * len(dataset))
train_dataset = dataset[:split_idx]
test_dataset = dataset[split_idx:]

print("\nTrain graphs:", [d.case_name for d in train_dataset])
print("Test graphs:", [d.case_name for d in test_dataset])


# -----------------------
# Normalize node features
# -----------------------
# Use train statistics only to avoid leaking test information.
all_train_x = torch.cat([data.x for data in train_dataset], dim=0)

mean = all_train_x.mean(dim=0)
std = all_train_x.std(dim=0)
std[std == 0] = 1.0

for data in train_dataset:
    data.x = (data.x - mean) / std

for data in test_dataset:
    data.x = (data.x - mean) / std


# -----------------------
# DataLoaders
# -----------------------
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# -----------------------
# Balanced loss mask
# -----------------------
def create_balanced_loss_mask(y, negative_ratio=5):
    """
    Keeps all positive nodes and randomly samples negative nodes.
    The full graph is still used for message passing.
    Only the loss is computed on a balanced subset.
    """
    device = y.device

    pos_idx = torch.where(y == 1)[0]
    neg_idx = torch.where(y == 0)[0]

    if len(pos_idx) == 0:
        return torch.arange(len(y), device=device)

    num_neg = min(len(neg_idx), len(pos_idx) * negative_ratio)

    perm = torch.randperm(len(neg_idx), device=device)
    sampled_neg_idx = neg_idx[perm[:num_neg]]

    mask_idx = torch.cat([pos_idx, sampled_neg_idx])

    return mask_idx


# -----------------------
# GAT Model
# -----------------------
class MultiGraphGAT(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=16,
        heads=4,
        out_channels=2,
        dropout=0.2,
    ):
        super().__init__()

        self.dropout = dropout

        self.gat1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
        )

        self.gat2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, edge_index)

        return x


# -----------------------
# Device + Model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("\nDevice:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


input_dim = train_dataset[0].x.shape[1]
print("Input feature dimension:", input_dim)

model = MultiGraphGAT(in_channels=input_dim).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)


# -----------------------
# Evaluation
# -----------------------
def evaluate(loader, name="Eval"):
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index)
            pred = out.argmax(dim=1)

            all_true.append(batch.y.cpu())
            all_pred.append(pred.cpu())

    true = torch.cat(all_true).numpy()
    pred = torch.cat(all_pred).numpy()

    print(f"\nClassification Report ({name}):")
    print(classification_report(true, pred, digits=4, zero_division=0))


# -----------------------
# Training loop
# -----------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()

        out = model(batch.x, batch.edge_index)

        loss_idx = create_balanced_loss_mask(
            batch.y,
            negative_ratio=NEGATIVE_RATIO,
        )

        loss = F.cross_entropy(out[loss_idx], batch.y[loss_idx])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if epoch % 20 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")


# -----------------------
# Final evaluation
# -----------------------
evaluate(train_loader, name="Train")
evaluate(test_loader, name="Test")