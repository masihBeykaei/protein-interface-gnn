import os
import csv
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# -----------------------
# Config
# -----------------------
SEED = 42
EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 2
NEGATIVE_RATIO = 5

LR = 0.005
WEIGHT_DECAY = 5e-4

PROCESSED_DIR = os.path.join("data", "processed")

OUTPUT_CSV = os.path.join("experiments", "gat_hyperparameter_tuning_results.csv")
OUTPUT_MD = os.path.join("experiments", "gat_hyperparameter_tuning_results.md")

THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
]

# Small, practical grid to avoid very long runtime
GAT_CONFIGS = [
    {"hidden_channels": 16, "heads": 4, "dropout": 0.2},  # current baseline
    {"hidden_channels": 32, "heads": 4, "dropout": 0.2},
    {"hidden_channels": 16, "heads": 8, "dropout": 0.2},
    {"hidden_channels": 32, "heads": 8, "dropout": 0.2},
    {"hidden_channels": 16, "heads": 4, "dropout": 0.3},
]


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

    for path in [features_path, labels_path, edge_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    x = np.load(features_path)
    y = np.load(labels_path)
    edge_index = np.load(edge_path)

    if len(y) == 0 or int(y.sum()) == 0:
        raise ValueError(f"Empty graph or no positive nodes for {case_name}")

    # Important:
    # All feature modes start with:
    # [CA_distance, degree_A, degree_B]
    # The current best strict model uses only these basic 3 features.
    x = x[:, :3]

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
    )

    data.case_name = case_name

    return data


def load_split(case_names):
    return [load_graph(case_name) for case_name in case_names]


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
# GAT model
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
# Training
# -----------------------
def train_one_config(
    config,
    train_loader,
    val_loader,
    test_loader,
    device,
    input_dim,
):
    set_seed(SEED)

    model = MultiGraphGAT(
        in_channels=input_dim,
        hidden_channels=config["hidden_channels"],
        heads=config["heads"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = -1.0
    best_epoch = -1
    best_threshold = 0.50
    best_state = None
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
                f"GAT hidden={config['hidden_channels']} "
                f"heads={config['heads']} "
                f"dropout={config['dropout']} | "
                f"Epoch {epoch} | "
                f"Loss={avg_loss:.4f} | "
                f"Val_F1_1={val_f1:.4f} | "
                f"Best_Val_F1_1={best_val_f1:.4f} | "
                f"Best_Threshold={best_threshold:.2f}"
            )

        if patience_counter >= PATIENCE:
            print(
                f"Early stopping at epoch {epoch} "
                f"for hidden={config['hidden_channels']}, "
                f"heads={config['heads']}, "
                f"dropout={config['dropout']}"
            )
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    val_true, val_prob_1 = collect_probabilities(model, val_loader, device)
    val_metrics = evaluate_at_threshold(val_true, val_prob_1, best_threshold)

    test_true, test_prob_1 = collect_probabilities(model, test_loader, device)
    test_metrics = evaluate_at_threshold(test_true, test_prob_1, best_threshold)

    result = {
        "hidden_channels": config["hidden_channels"],
        "heads": config["heads"],
        "dropout": config["dropout"],

        "best_epoch": best_epoch,
        "best_threshold": best_threshold,

        "val_precision_1": val_metrics["precision_1"],
        "val_recall_1": val_metrics["recall_1"],
        "val_f1_1": val_metrics["f1_1"],
        "val_accuracy": val_metrics["accuracy"],

        "test_precision_1": test_metrics["precision_1"],
        "test_recall_1": test_metrics["recall_1"],
        "test_f1_1": test_metrics["f1_1"],
        "test_accuracy": test_metrics["accuracy"],
    }

    return result


# -----------------------
# Save outputs
# -----------------------
def save_csv(results):
    os.makedirs("experiments", exist_ok=True)

    fieldnames = [
        "hidden_channels",
        "heads",
        "dropout",
        "best_epoch",
        "best_threshold",

        "val_precision_1",
        "val_recall_1",
        "val_f1_1",
        "val_accuracy",

        "test_precision_1",
        "test_recall_1",
        "test_f1_1",
        "test_accuracy",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved: {OUTPUT_CSV}")


def save_markdown(results):
    sorted_by_test_f1 = sorted(
        results,
        key=lambda row: row["test_f1_1"],
        reverse=True,
    )

    sorted_by_val_f1 = sorted(
        results,
        key=lambda row: row["val_f1_1"],
        reverse=True,
    )

    best_by_test = sorted_by_test_f1[0]
    best_by_val = sorted_by_val_f1[0]

    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("# GAT Hyperparameter Tuning Results\n\n")

        f.write("This experiment tunes GAT hyperparameters using the strict graph-level train/validation/test protocol.\n\n")

        f.write("```text\n")
        f.write("Model: GAT\n")
        f.write("Feature set: basic 3 features\n")
        f.write("Input dimension: 3\n")
        f.write("Negative ratio: 5\n")
        f.write("Threshold selection: validation set\n")
        f.write("Early stopping: validation positive-class F1\n")
        f.write("```\n\n")

        f.write("## Search Space\n\n")
        f.write("| hidden_channels | heads | dropout |\n")
        f.write("|-----------------|-------|---------|\n")

        for config in GAT_CONFIGS:
            f.write(
                f"| {config['hidden_channels']} "
                f"| {config['heads']} "
                f"| {config['dropout']} |\n"
            )

        f.write("\n## Results\n\n")
        f.write(
            "| Hidden | Heads | Dropout | Best Epoch | Threshold | "
            "Val P1 | Val R1 | Val F1 | Val Acc | "
            "Test P1 | Test R1 | Test F1 | Test Acc |\n"
        )
        f.write(
            "|--------|-------|---------|------------|-----------|"
            "--------|--------|--------|---------|"
            "---------|---------|---------|----------|\n"
        )

        for row in results:
            f.write(
                f"| {row['hidden_channels']} "
                f"| {row['heads']} "
                f"| {row['dropout']} "
                f"| {row['best_epoch']} "
                f"| {row['best_threshold']:.2f} "
                f"| {row['val_precision_1']:.4f} "
                f"| {row['val_recall_1']:.4f} "
                f"| {row['val_f1_1']:.4f} "
                f"| {row['val_accuracy']:.4f} "
                f"| {row['test_precision_1']:.4f} "
                f"| {row['test_recall_1']:.4f} "
                f"| {row['test_f1_1']:.4f} "
                f"| {row['test_accuracy']:.4f} |\n"
            )

        f.write("\n## Best Configuration by Validation F1\n\n")
        f.write("| Hidden | Heads | Dropout | Best Epoch | Threshold | Val F1 | Test F1 |\n")
        f.write("|--------|-------|---------|------------|-----------|--------|---------|\n")
        f.write(
            f"| {best_by_val['hidden_channels']} "
            f"| {best_by_val['heads']} "
            f"| {best_by_val['dropout']} "
            f"| {best_by_val['best_epoch']} "
            f"| {best_by_val['best_threshold']:.2f} "
            f"| {best_by_val['val_f1_1']:.4f} "
            f"| {best_by_val['test_f1_1']:.4f} |\n"
        )

        f.write("\n## Best Configuration by Test F1\n\n")
        f.write(
            "This is reported for analysis only. "
            "Model selection should be based on validation performance.\n\n"
        )
        f.write("| Hidden | Heads | Dropout | Best Epoch | Threshold | Val F1 | Test F1 |\n")
        f.write("|--------|-------|---------|------------|-----------|--------|---------|\n")
        f.write(
            f"| {best_by_test['hidden_channels']} "
            f"| {best_by_test['heads']} "
            f"| {best_by_test['dropout']} "
            f"| {best_by_test['best_epoch']} "
            f"| {best_by_test['best_threshold']:.2f} "
            f"| {best_by_test['val_f1_1']:.4f} "
            f"| {best_by_test['test_f1_1']:.4f} |\n"
        )

        f.write("\n## Interpretation Notes\n\n")
        f.write("- The validation set is used for early stopping and threshold selection.\n")
        f.write("- The test set is used only for final evaluation.\n")
        f.write("- The current baseline configuration is `hidden_channels=16`, `heads=4`, `dropout=0.2`.\n")
        f.write("- If a larger configuration improves validation F1 but not test F1, it may indicate overfitting or limited data size.\n")

        f.write("\n## Output Files\n\n")
        f.write(f"- CSV results: `{OUTPUT_CSV}`\n")
        f.write(f"- Markdown summary: `{OUTPUT_MD}`\n")

    print(f"Saved: {OUTPUT_MD}")


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

    input_dim = train_dataset[0].x.shape[1]

    print("\nGAT hyperparameter tuning")
    print("Feature set: basic 3 features")
    print("Input feature dimension:", input_dim)

    print("\nTrain graphs:", TRAIN_CASES)
    print("Validation graphs:", VAL_CASES)
    print("Test graphs:", TEST_CASES)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    results = []

    for idx, config in enumerate(GAT_CONFIGS, start=1):
        print("\n" + "=" * 70)
        print(
            f"Running config {idx}/{len(GAT_CONFIGS)}: "
            f"hidden={config['hidden_channels']}, "
            f"heads={config['heads']}, "
            f"dropout={config['dropout']}"
        )
        print("=" * 70)

        result = train_one_config(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            input_dim=input_dim,
        )

        results.append(result)

        print("\nResult:")
        print(
            f"hidden={result['hidden_channels']} | "
            f"heads={result['heads']} | "
            f"dropout={result['dropout']} | "
            f"Val_F1_1={result['val_f1_1']:.4f} | "
            f"Test_P1={result['test_precision_1']:.4f} | "
            f"Test_R1={result['test_recall_1']:.4f} | "
            f"Test_F1_1={result['test_f1_1']:.4f} | "
            f"Test_Acc={result['test_accuracy']:.4f}"
        )

    save_csv(results)
    save_markdown(results)

    best_by_val = max(results, key=lambda row: row["val_f1_1"])

    print("\nBest configuration by validation F1:")
    print(
        f"hidden={best_by_val['hidden_channels']} | "
        f"heads={best_by_val['heads']} | "
        f"dropout={best_by_val['dropout']} | "
        f"threshold={best_by_val['best_threshold']:.2f} | "
        f"Val_F1_1={best_by_val['val_f1_1']:.4f} | "
        f"Test_F1_1={best_by_val['test_f1_1']:.4f}"
    )

    print("\nGAT hyperparameter tuning completed successfully.")


if __name__ == "__main__":
    main()