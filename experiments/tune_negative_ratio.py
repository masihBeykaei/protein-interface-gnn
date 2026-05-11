import os
import csv
import random
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# -----------------------
# Config
# -----------------------
SEED = 42
EPOCHS = 201
LR_GCN = 0.01
LR_GAT = 0.005
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 2

NEGATIVE_RATIOS = [2, 3, 5, 10]

PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_CSV = os.path.join("experiments", "negative_ratio_tuning_results.csv")
RESULTS_MD = os.path.join("experiments", "negative_ratio_tuning_results.md")

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
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    if len(y) == 0 or int(y.sum()) == 0:
        print(f"Skipping {case_name}: empty graph or no positive nodes.")
        return None

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
    )

    data.case_name = case_name
    return data


def load_dataset():
    dataset = []

    for case in CASES:
        graph = load_graph(case)
        if graph is not None:
            dataset.append(graph)

    if len(dataset) < 2:
        raise RuntimeError("Not enough graphs loaded.")

    random.shuffle(dataset)

    split_idx = int(0.8 * len(dataset))
    train_dataset = dataset[:split_idx]
    test_dataset = dataset[split_idx:]

    # Normalize using train-set statistics only
    all_train_x = torch.cat([data.x for data in train_dataset], dim=0)

    mean = all_train_x.mean(dim=0)
    std = all_train_x.std(dim=0)
    std[std == 0] = 1.0

    for data in train_dataset:
        data.x = (data.x - mean) / std

    for data in test_dataset:
        data.x = (data.x - mean) / std

    return train_dataset, test_dataset


# -----------------------
# Balanced loss mask
# -----------------------
def create_balanced_loss_mask(y, negative_ratio=5):
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
# Models
# -----------------------
class MultiGraphGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=2):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        return x


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
# Evaluation
# -----------------------
def evaluate(model, loader, device):
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index)
            pred = out.argmax(dim=1)

            all_true.append(batch.y.cpu().numpy())
            all_pred.append(pred.cpu().numpy())

    true = np.concatenate(all_true)
    pred = np.concatenate(all_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        pred,
        labels=[0, 1],
        zero_division=0,
    )

    accuracy = accuracy_score(true, pred)

    return {
        "accuracy": accuracy,
        "precision_0": precision[0],
        "recall_0": recall[0],
        "f1_0": f1[0],
        "support_0": support[0],
        "precision_1": precision[1],
        "recall_1": recall[1],
        "f1_1": f1[1],
        "support_1": support[1],
    }


# -----------------------
# Train one experiment
# -----------------------
def train_model(model_name, negative_ratio, train_loader, test_loader, device):
    set_seed(SEED)

    input_dim = train_loader.dataset[0].x.shape[1]
    print(f"{model_name} input feature dimension: {input_dim}")

    if model_name == "GCN":
        model = MultiGraphGCN(in_channels=input_dim).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR_GCN,
            weight_decay=WEIGHT_DECAY,
        )
    elif model_name == "GAT":
        model = MultiGraphGAT(in_channels=input_dim).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LR_GAT,
            weight_decay=WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()

            out = model(batch.x, batch.edge_index)

            loss_idx = create_balanced_loss_mask(
                batch.y,
                negative_ratio=negative_ratio,
            )

            loss = F.cross_entropy(out[loss_idx], batch.y[loss_idx])

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 50 == 0:
            avg_loss = total_loss / len(train_loader)
            print(
                f"{model_name} | NEG_RATIO={negative_ratio} | "
                f"Epoch {epoch} | Loss={avg_loss:.4f}"
            )

    train_metrics = evaluate(model, train_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    return train_metrics, test_metrics


# -----------------------
# Main
# -----------------------
def main():
    os.makedirs("experiments", exist_ok=True)

    set_seed(SEED)

    train_dataset, test_dataset = load_dataset()

    print("\nTrain graphs:")
    print([data.case_name for data in train_dataset])

    print("\nTest graphs:")
    print([data.case_name for data in test_dataset])

    print("\nFeature dimension:", train_dataset[0].x.shape[1])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    results = []

    for model_name in ["GCN", "GAT"]:
        for ratio in NEGATIVE_RATIOS:
            print("\n" + "=" * 70)
            print(f"Running {model_name} with NEGATIVE_RATIO={ratio}")
            print("=" * 70)

            train_metrics, test_metrics = train_model(
                model_name=model_name,
                negative_ratio=ratio,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
            )

            row = {
                "model": model_name,
                "negative_ratio": ratio,

                "train_precision_1": train_metrics["precision_1"],
                "train_recall_1": train_metrics["recall_1"],
                "train_f1_1": train_metrics["f1_1"],
                "train_accuracy": train_metrics["accuracy"],

                "test_precision_1": test_metrics["precision_1"],
                "test_recall_1": test_metrics["recall_1"],
                "test_f1_1": test_metrics["f1_1"],
                "test_accuracy": test_metrics["accuracy"],
            }

            results.append(row)

            print("\nTest Result:")
            print(
                f"{model_name} ratio={ratio} | "
                f"P1={row['test_precision_1']:.4f} | "
                f"R1={row['test_recall_1']:.4f} | "
                f"F1={row['test_f1_1']:.4f} | "
                f"Acc={row['test_accuracy']:.4f}"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save CSV
    fieldnames = list(results[0].keys())

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Save Markdown
    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write("# Negative Ratio Tuning Results\n\n")
        f.write("This experiment compares GCN and GAT under different negative sampling ratios.\n\n")
        f.write(f"Input feature dimension: `{train_dataset[0].x.shape[1]}`\n\n")

        f.write("| Model | Negative Ratio | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |\n")
        f.write("|-------|----------------|------------------|---------------|-----------|---------------|\n")

        for row in results:
            f.write(
                f"| {row['model']} "
                f"| {row['negative_ratio']} "
                f"| {row['test_precision_1']:.4f} "
                f"| {row['test_recall_1']:.4f} "
                f"| {row['test_f1_1']:.4f} "
                f"| {row['test_accuracy']:.4f} |\n"
            )

    print("\nSaved results:")
    print(RESULTS_CSV)
    print(RESULTS_MD)


if __name__ == "__main__":
    main()