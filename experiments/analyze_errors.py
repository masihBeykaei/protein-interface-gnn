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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


# -----------------------
# Config
# -----------------------
SEED = 42
EPOCHS = 300
PATIENCE = 40
BATCH_SIZE = 2
NEGATIVE_RATIO = 5

LR_GAT = 0.005
WEIGHT_DECAY = 5e-4

THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.40, 0.50, 0.60, 0.70, 0.80, 0.90
]

PROCESSED_DIR = os.path.join("data", "processed")

OUTPUT_CSV = os.path.join("experiments", "error_analysis_gat_basic.csv")
OUTPUT_SUMMARY_MD = os.path.join("experiments", "error_analysis_gat_basic_summary.md")


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
def load_graph(case_name, use_basic_features=True):
    features_path = os.path.join(PROCESSED_DIR, f"{case_name}_corr_features.npy")
    labels_path = os.path.join(PROCESSED_DIR, f"{case_name}_corr_labels.npy")
    edge_path = os.path.join(PROCESSED_DIR, f"{case_name}_corr_edge_index.npy")
    pairs_path = os.path.join(PROCESSED_DIR, f"{case_name}_corr_pairs.npy")

    required_files = [
        features_path,
        labels_path,
        edge_path,
        pairs_path,
    ]

    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    x = np.load(features_path)
    y = np.load(labels_path)
    edge_index = np.load(edge_path)
    pairs = np.load(pairs_path)

    if len(y) == 0 or int(y.sum()) == 0:
        raise ValueError(f"Empty graph or no positive nodes for {case_name}")

    # Important:
    # All feature modes start with:
    # [CA_distance, degree_A, degree_B]
    # For the best strict model, we use only these basic 3 features.
    if use_basic_features:
        x = x[:, :3]

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
    )

    data.case_name = case_name
    data.pairs = torch.tensor(pairs, dtype=torch.long)

    return data


def load_split(case_names):
    return [load_graph(case, use_basic_features=True) for case in case_names]


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
# Training with early stopping
# -----------------------
def train_gat_with_early_stopping(train_loader, val_loader, device, input_dim):
    set_seed(SEED)

    model = MultiGraphGAT(in_channels=input_dim).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR_GAT,
        weight_decay=WEIGHT_DECAY,
    )

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
                f"GAT | Epoch {epoch} | "
                f"Loss={avg_loss:.4f} | "
                f"Val_F1_1={val_f1:.4f} | "
                f"Best_Val_F1_1={best_val_f1:.4f} | "
                f"Best_Threshold={best_threshold:.2f}"
            )

        if patience_counter >= PATIENCE:
            print(f"GAT early stopping at epoch {epoch}")
            break

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    return model, best_epoch, best_threshold, best_val_f1


# -----------------------
# Error analysis
# -----------------------
def analyze_test_errors(model, test_dataset, device, threshold):
    model.eval()

    all_rows = []
    per_case_summary = {}

    all_true = []
    all_pred = []

    with torch.no_grad():
        for data in test_dataset:
            case_name = data.case_name

            graph = data.to(device)

            out = model(graph.x, graph.edge_index)
            probs = F.softmax(out, dim=1)
            prob_1 = probs[:, 1].cpu().numpy()

            true = graph.y.cpu().numpy()
            pred = (prob_1 >= threshold).astype(np.int64)

            pairs = data.pairs.cpu().numpy()
            x_basic = data.x.cpu().numpy()

            all_true.append(true)
            all_pred.append(pred)

            tp = int(((true == 1) & (pred == 1)).sum())
            tn = int(((true == 0) & (pred == 0)).sum())
            fp = int(((true == 0) & (pred == 1)).sum())
            fn = int(((true == 1) & (pred == 0)).sum())

            per_case_summary[case_name] = {
                "nodes": len(true),
                "positive": int((true == 1).sum()),
                "negative": int((true == 0).sum()),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }

            for node_id in range(len(true)):
                y_true = int(true[node_id])
                y_pred = int(pred[node_id])

                if y_true == y_pred:
                    continue

                if y_true == 0 and y_pred == 1:
                    error_type = "FP"
                elif y_true == 1 and y_pred == 0:
                    error_type = "FN"
                else:
                    error_type = "UNKNOWN"

                residue_a_idx = int(pairs[node_id][0])
                residue_b_idx = int(pairs[node_id][1])

                ca_distance = float(x_basic[node_id][0])
                degree_a = float(x_basic[node_id][1])
                degree_b = float(x_basic[node_id][2])

                all_rows.append({
                    "case": case_name,
                    "node_id": node_id,
                    "residue_a_idx": residue_a_idx,
                    "residue_b_idx": residue_b_idx,
                    "true_label": y_true,
                    "pred_label": y_pred,
                    "prob_class_1": float(prob_1[node_id]),
                    "error_type": error_type,
                    "ca_distance_basic_feature": ca_distance,
                    "degree_a_basic_feature": degree_a,
                    "degree_b_basic_feature": degree_b,
                })

    true_all = np.concatenate(all_true)
    pred_all = np.concatenate(all_pred)

    return all_rows, per_case_summary, true_all, pred_all


def save_error_csv(rows):
    os.makedirs("experiments", exist_ok=True)

    fieldnames = [
        "case",
        "node_id",
        "residue_a_idx",
        "residue_b_idx",
        "true_label",
        "pred_label",
        "prob_class_1",
        "error_type",
        "ca_distance_basic_feature",
        "degree_a_basic_feature",
        "degree_b_basic_feature",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {OUTPUT_CSV}")


def save_summary_md(
    best_epoch,
    best_threshold,
    best_val_f1,
    metrics,
    cm,
    per_case_summary,
    rows,
):
    tn, fp, fn, tp = cm.ravel()

    fp_rows = [row for row in rows if row["error_type"] == "FP"]
    fn_rows = [row for row in rows if row["error_type"] == "FN"]

    # Sort false positives by highest predicted probability
    fp_top = sorted(
        fp_rows,
        key=lambda row: row["prob_class_1"],
        reverse=True,
    )[:10]

    # Sort false negatives by lowest predicted probability
    fn_top = sorted(
        fn_rows,
        key=lambda row: row["prob_class_1"],
    )[:10]

    with open(OUTPUT_SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("# Error Analysis: GAT with Basic 3 Features\n\n")

        f.write("This analysis uses the current best strict protocol model:\n\n")
        f.write("```text\n")
        f.write("Model: GAT\n")
        f.write("Features: basic 3 features\n")
        f.write("Input dimension: 3\n")
        f.write("Split: train/validation/test\n")
        f.write("Threshold selected on validation set\n")
        f.write("```\n\n")

        f.write("## Training Selection\n\n")
        f.write(f"- Best epoch: `{best_epoch}`\n")
        f.write(f"- Best validation threshold: `{best_threshold:.2f}`\n")
        f.write(f"- Best validation F1 for class 1: `{best_val_f1:.4f}`\n\n")

        f.write("## Test Metrics\n\n")
        f.write("| Precision 1 | Recall 1 | F1 1 | Accuracy |\n")
        f.write("|-------------|----------|------|----------|\n")
        f.write(
            f"| {metrics['precision_1']:.4f} "
            f"| {metrics['recall_1']:.4f} "
            f"| {metrics['f1_1']:.4f} "
            f"| {metrics['accuracy']:.4f} |\n\n"
        )

        f.write("## Confusion Matrix on Test Set\n\n")
        f.write("| True / Pred | Pred 0 | Pred 1 |\n")
        f.write("|-------------|--------|--------|\n")
        f.write(f"| True 0 | {tn} | {fp} |\n")
        f.write(f"| True 1 | {fn} | {tp} |\n\n")

        f.write("Definitions:\n\n")
        f.write("- FP: model predicted interface/contact, but the true label is non-contact\n")
        f.write("- FN: model missed a true interface/contact pair\n\n")

        f.write("## Per-Test-Graph Error Summary\n\n")
        f.write("| Case | Nodes | Positive | Negative | TP | TN | FP | FN |\n")
        f.write("|------|-------|----------|----------|----|----|----|----|\n")

        for case_name, item in per_case_summary.items():
            f.write(
                f"| {case_name} "
                f"| {item['nodes']} "
                f"| {item['positive']} "
                f"| {item['negative']} "
                f"| {item['tp']} "
                f"| {item['tn']} "
                f"| {item['fp']} "
                f"| {item['fn']} |\n"
            )

        f.write("\n## Top False Positives by Predicted Probability\n\n")
        f.write("| Case | Node ID | Residue A | Residue B | P(class 1) | CA Distance | Degree A | Degree B |\n")
        f.write("|------|---------|-----------|-----------|------------|-------------|----------|----------|\n")

        for row in fp_top:
            f.write(
                f"| {row['case']} "
                f"| {row['node_id']} "
                f"| {row['residue_a_idx']} "
                f"| {row['residue_b_idx']} "
                f"| {row['prob_class_1']:.4f} "
                f"| {row['ca_distance_basic_feature']:.4f} "
                f"| {row['degree_a_basic_feature']:.4f} "
                f"| {row['degree_b_basic_feature']:.4f} |\n"
            )

        f.write("\n## Top False Negatives by Lowest Predicted Probability\n\n")
        f.write("| Case | Node ID | Residue A | Residue B | P(class 1) | CA Distance | Degree A | Degree B |\n")
        f.write("|------|---------|-----------|-----------|------------|-------------|----------|----------|\n")

        for row in fn_top:
            f.write(
                f"| {row['case']} "
                f"| {row['node_id']} "
                f"| {row['residue_a_idx']} "
                f"| {row['residue_b_idx']} "
                f"| {row['prob_class_1']:.4f} "
                f"| {row['ca_distance_basic_feature']:.4f} "
                f"| {row['degree_a_basic_feature']:.4f} "
                f"| {row['degree_b_basic_feature']:.4f} |\n"
            )

        f.write("\n## Output Files\n\n")
        f.write(f"- Full error table: `{OUTPUT_CSV}`\n")
        f.write(f"- Summary: `{OUTPUT_SUMMARY_MD}`\n")

    print(f"Saved: {OUTPUT_SUMMARY_MD}")


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

    print("\nError analysis model:")
    print("Model: GAT")
    print("Feature set: basic 3 features")
    print("Input feature dimension:", input_dim)

    print("\nTrain graphs:", TRAIN_CASES)
    print("Validation graphs:", VAL_CASES)
    print("Test graphs:", TEST_CASES)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model, best_epoch, best_threshold, best_val_f1 = train_gat_with_early_stopping(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        input_dim=input_dim,
    )

    rows, per_case_summary, true_all, pred_all = analyze_test_errors(
        model=model,
        test_dataset=test_dataset,
        device=device,
        threshold=best_threshold,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        true_all,
        pred_all,
        labels=[0, 1],
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy_score(true_all, pred_all),
        "precision_0": precision[0],
        "recall_0": recall[0],
        "f1_0": f1[0],
        "support_0": support[0],
        "precision_1": precision[1],
        "recall_1": recall[1],
        "f1_1": f1[1],
        "support_1": support[1],
    }

    cm = confusion_matrix(true_all, pred_all, labels=[0, 1])

    print("\nFinal Test Metrics:")
    print(
        f"P1={metrics['precision_1']:.4f} | "
        f"R1={metrics['recall_1']:.4f} | "
        f"F1={metrics['f1_1']:.4f} | "
        f"Acc={metrics['accuracy']:.4f}"
    )

    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    print(f"\nTotal errors: {len(rows)}")
    print(f"False positives: {sum(row['error_type'] == 'FP' for row in rows)}")
    print(f"False negatives: {sum(row['error_type'] == 'FN' for row in rows)}")

    save_error_csv(rows)

    save_summary_md(
        best_epoch=best_epoch,
        best_threshold=best_threshold,
        best_val_f1=best_val_f1,
        metrics=metrics,
        cm=cm,
        per_case_summary=per_case_summary,
        rows=rows,
    )


if __name__ == "__main__":
    main()