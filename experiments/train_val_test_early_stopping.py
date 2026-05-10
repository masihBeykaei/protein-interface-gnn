import os
import csv
import copy
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
EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 2
NEGATIVE_RATIO = 5

LR_GCN = 0.01
LR_GAT = 0.005
WEIGHT_DECAY = 5e-4

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

PROCESSED_DIR = os.path.join("data", "processed")
RESULTS_CSV = os.path.join("experiments", "early_stopping_results.csv")
RESULTS_MD = os.path.join("experiments", "early_stopping_results.md")


# -----------------------
# Fixed graph split
# -----------------------
TRAIN_CASES = [
    "1WEJ_HL_F",
    "1JPS_HL_T",
    "1AHW_AB_C",
    "2FD6_HL_U",
    "2VIS_AB_C",
    "1MLC_AB_E",
    "3MJ9_HL_A",
]

VAL_CASES = [
    "1DQJ_AB_C",
    "1E6J_HL_P",
]

TEST_CASES = [
    "1BRS_A_B",
    "1FSS_A_B",
    "3HMX_LH_AB",
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
# Dataset loading
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
        raise FileNotFoundError(f"Missing processed files for {case_name}")

    x = np.load(features_path)
    y = np.load(labels_path)
    edge_index = np.load(edge_path)

    if len(y) == 0 or int(y.sum()) == 0:
        raise ValueError(f"Empty graph or no positive nodes for {case_name}")

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
    )

    data.case_name = case_name
    return data


def load_split(case_names):
    return [load_graph(case) for case in case_names]


def normalize_features(train_dataset, val_dataset, test_dataset):
    """
    Normalize using train-set statistics only.
    This avoids leaking validation/test information into preprocessing.
    """
    all_train_x = torch.cat([data.x for data in train_dataset], dim=0)

    mean = all_train_x.mean(dim=0)
    std = all_train_x.std(dim=0)
    std[std == 0] = 1.0

    for dataset in [train_dataset, val_dataset, test_dataset]:
        for data in dataset:
            data.x = (data.x - mean) / std

    return train_dataset, val_dataset, test_dataset


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
    def __init__(self, in_channels=3, hidden_channels=32, out_channels=2):
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
        in_channels=3,
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


def build_model(model_name):
    if model_name == "GCN":
        return MultiGraphGCN()
    if model_name == "GAT":
        return MultiGraphGAT()
    raise ValueError(f"Unknown model: {model_name}")


def build_optimizer(model_name, model):
    if model_name == "GCN":
        lr = LR_GCN
    elif model_name == "GAT":
        lr = LR_GAT
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )


# -----------------------
# Metrics
# -----------------------
def collect_probabilities(model, loader, device):
    model.eval()

    all_true = []
    all_prob_1 = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            out = model(batch.x, batch.edge_index)
            probs = F.softmax(out, dim=1)
            prob_1 = probs[:, 1]

            all_true.append(batch.y.cpu().numpy())
            all_prob_1.append(prob_1.cpu().numpy())

    true = np.concatenate(all_true)
    prob_1 = np.concatenate(all_prob_1)

    return true, prob_1


def evaluate_at_threshold(true, prob_1, threshold):
    pred = (prob_1 >= threshold).astype(np.int64)

    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        pred,
        labels=[0, 1],
        zero_division=0,
    )

    accuracy = accuracy_score(true, pred)

    return {
        "threshold": threshold,
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


def find_best_threshold(true, prob_1):
    best_metrics = None

    for threshold in THRESHOLDS:
        metrics = evaluate_at_threshold(true, prob_1, threshold)

        if best_metrics is None or metrics["f1_1"] > best_metrics["f1_1"]:
            best_metrics = metrics

    return best_metrics


# -----------------------
# Training with early stopping
# -----------------------
def train_with_early_stopping(model_name, train_loader, val_loader, test_loader, device):
    set_seed(SEED)

    model = build_model(model_name).to(device)
    optimizer = build_optimizer(model_name, model)

    best_val_f1 = -1.0
    best_epoch = -1
    best_state = None
    best_threshold = 0.50
    patience_counter = 0

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

        # Validation: choose best threshold on validation set
        val_true, val_prob_1 = collect_probabilities(model, val_loader, device)
        val_best_metrics = find_best_threshold(val_true, val_prob_1)

        val_f1 = val_best_metrics["f1_1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_threshold = val_best_metrics["threshold"]
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0:
            avg_loss = total_loss / len(train_loader)
            print(
                f"{model_name} | Epoch {epoch} | "
                f"Loss={avg_loss:.4f} | "
                f"Val_F1_1={val_f1:.4f} | "
                f"Best_Val_F1_1={best_val_f1:.4f} | "
                f"Best_Threshold={best_threshold:.2f}"
            )

        if patience_counter >= PATIENCE:
            print(f"{model_name} early stopping at epoch {epoch}")
            break

    # Load best model
    model.load_state_dict(best_state)

    # Final validation metrics at selected threshold
    val_true, val_prob_1 = collect_probabilities(model, val_loader, device)
    final_val_metrics = evaluate_at_threshold(
        val_true,
        val_prob_1,
        best_threshold,
    )

    # Final test metrics at selected validation threshold
    test_true, test_prob_1 = collect_probabilities(model, test_loader, device)
    final_test_metrics = evaluate_at_threshold(
        test_true,
        test_prob_1,
        best_threshold,
    )

    return {
        "model": model_name,
        "best_epoch": best_epoch,
        "best_threshold": best_threshold,
        "val_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
    }


# -----------------------
# Save results
# -----------------------
def save_results(results):
    rows = []

    for item in results:
        val = item["val_metrics"]
        test = item["test_metrics"]

        row = {
            "model": item["model"],
            "best_epoch": item["best_epoch"],
            "best_threshold": item["best_threshold"],

            "val_precision_1": val["precision_1"],
            "val_recall_1": val["recall_1"],
            "val_f1_1": val["f1_1"],
            "val_accuracy": val["accuracy"],

            "test_precision_1": test["precision_1"],
            "test_recall_1": test["recall_1"],
            "test_f1_1": test["f1_1"],
            "test_accuracy": test["accuracy"],
        }

        rows.append(row)

    csv_path = os.path.join("experiments", "early_stopping_results.csv")
    md_path = os.path.join("experiments", "early_stopping_results.md")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Train/Validation/Test Early Stopping Results\n\n")
        f.write("This experiment uses a graph-level train/validation/test split.\n\n")
        f.write("The validation set is used for:\n\n")
        f.write("- early stopping\n")
        f.write("- selecting the probability threshold for class 1\n\n")
        f.write("The test set is used only for final evaluation.\n\n")

        f.write("## Split\n\n")

        f.write("### Train Graphs\n\n")
        for case in TRAIN_CASES:
            f.write(f"- {case}\n")

        f.write("\n### Validation Graphs\n\n")
        for case in VAL_CASES:
            f.write(f"- {case}\n")

        f.write("\n### Test Graphs\n\n")
        for case in TEST_CASES:
            f.write(f"- {case}\n")

        f.write("\n## Results\n\n")
        f.write("| Model | Best Epoch | Best Threshold | Val Precision 1 | Val Recall 1 | Val F1 1 | Test Precision 1 | Test Recall 1 | Test F1 1 | Test Accuracy |\n")
        f.write("|-------|------------|----------------|-----------------|--------------|----------|------------------|---------------|-----------|---------------|\n")

        for row in rows:
            f.write(
                f"| {row['model']} "
                f"| {row['best_epoch']} "
                f"| {row['best_threshold']:.2f} "
                f"| {row['val_precision_1']:.4f} "
                f"| {row['val_recall_1']:.4f} "
                f"| {row['val_f1_1']:.4f} "
                f"| {row['test_precision_1']:.4f} "
                f"| {row['test_recall_1']:.4f} "
                f"| {row['test_f1_1']:.4f} "
                f"| {row['test_accuracy']:.4f} |\n"
            )

    print("\nSaved results:")
    print(csv_path)
    print(md_path)


# -----------------------
# Main
# -----------------------
def main():
    os.makedirs("experiments", exist_ok=True)

    set_seed(SEED)

    train_dataset = load_split(TRAIN_CASES)
    val_dataset = load_split(VAL_CASES)
    test_dataset = load_split(TEST_CASES)

    train_dataset, val_dataset, test_dataset = normalize_features(
        train_dataset,
        val_dataset,
        test_dataset,
    )

    print("\nTrain graphs:", TRAIN_CASES)
    print("Validation graphs:", VAL_CASES)
    print("Test graphs:", TEST_CASES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    results = []

    for model_name in ["GCN", "GAT"]:
        print("\n" + "=" * 70)
        print(f"Training {model_name} with validation-based early stopping")
        print("=" * 70)

        result = train_with_early_stopping(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
        )

        results.append(result)

        test = result["test_metrics"]

        print(
            f"\n{model_name} Final Test Result | "
            f"Best Epoch={result['best_epoch']} | "
            f"Threshold={result['best_threshold']:.2f} | "
            f"P1={test['precision_1']:.4f} | "
            f"R1={test['recall_1']:.4f} | "
            f"F1={test['f1_1']:.4f} | "
            f"Acc={test['accuracy']:.4f}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_results(results)


if __name__ == "__main__":
    main()