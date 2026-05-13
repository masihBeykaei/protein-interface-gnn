import os
import csv
import copy
import heapq
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

LR_GAT = 0.005
WEIGHT_DECAY = 5e-4

THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.40, 0.50, 0.60, 0.70, 0.80, 0.90
]

PROCESSED_DIR = os.path.join("data", "processed")
FIGURE_DIR = os.path.join("experiments", "figures")

OUTPUT_ATTENTION_CSV = os.path.join("experiments", "gat_attention_weights.csv")
OUTPUT_TOP_CSV = os.path.join("experiments", "gat_attention_top_edges.csv")
OUTPUT_SUMMARY_MD = os.path.join("experiments", "gat_attention_summary.md")
OUTPUT_FIGURE = os.path.join(FIGURE_DIR, "gat_attention_distribution.png")

TOP_K = 100


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
    # The best strict model uses only these 3 basic features.
    if use_basic_features:
        x = x[:, :3]

    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(y, dtype=torch.long),
    )

    data.case_name = case_name
    data.pairs = torch.tensor(pairs, dtype=torch.long)

    # Keep a copy of raw basic features before normalization.
    # These are useful for interpreting top attention edges.
    data.raw_basic_features = torch.tensor(x[:, :3], dtype=torch.float)

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
# GAT model with attention extraction
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
        self.heads = heads

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

    def forward(self, x, edge_index, return_attention=False):
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention:
            x, attention_data = self.gat1(
                x,
                edge_index,
                return_attention_weights=True,
            )
            attention_edge_index, attention_weights = attention_data
        else:
            x = self.gat1(x, edge_index)
            attention_edge_index = None
            attention_weights = None

        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.gat2(x, edge_index)

        if return_attention:
            return x, attention_edge_index, attention_weights

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
# Final test evaluation
# -----------------------
def evaluate_test_dataset(model, test_dataset, device, threshold):
    model.eval()

    all_true = []
    all_prob_1 = []

    with torch.no_grad():
        for data in test_dataset:
            graph = data.to(device)

            out = model(graph.x, graph.edge_index)
            probs = F.softmax(out, dim=1)
            prob_1 = probs[:, 1]

            all_true.append(graph.y.cpu().numpy())
            all_prob_1.append(prob_1.cpu().numpy())

    true = np.concatenate(all_true)
    prob_1 = np.concatenate(all_prob_1)

    metrics = evaluate_at_threshold(true, prob_1, threshold)

    return metrics


# -----------------------
# Attention extraction
# -----------------------
def build_attention_row(
    case_name,
    edge_id,
    src_node,
    dst_node,
    attention_values,
    mean_attention,
    max_attention,
    src_pair,
    dst_pair,
    src_true,
    dst_true,
    src_prob_1,
    dst_prob_1,
    src_pred,
    dst_pred,
    src_raw_features,
    dst_raw_features,
):
    row = {
        "case": case_name,
        "edge_id": edge_id,
        "src_node": src_node,
        "dst_node": dst_node,
        "is_self_loop": int(src_node == dst_node),

        "src_residue_a_idx": int(src_pair[0]),
        "src_residue_b_idx": int(src_pair[1]),
        "dst_residue_a_idx": int(dst_pair[0]),
        "dst_residue_b_idx": int(dst_pair[1]),

        "src_true_label": int(src_true),
        "dst_true_label": int(dst_true),
        "src_pred_label": int(src_pred),
        "dst_pred_label": int(dst_pred),

        "src_prob_class_1": float(src_prob_1),
        "dst_prob_class_1": float(dst_prob_1),

        "mean_attention": float(mean_attention),
        "max_attention": float(max_attention),

        "src_ca_distance": float(src_raw_features[0]),
        "src_degree_a": float(src_raw_features[1]),
        "src_degree_b": float(src_raw_features[2]),

        "dst_ca_distance": float(dst_raw_features[0]),
        "dst_degree_a": float(dst_raw_features[1]),
        "dst_degree_b": float(dst_raw_features[2]),
    }

    for head_idx, value in enumerate(attention_values):
        row[f"head_{head_idx}"] = float(value)

    return row


def extract_and_save_attention(
    model,
    test_dataset,
    device,
    threshold,
    top_k=100,
):
    os.makedirs("experiments", exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

    model.eval()

    top_heap = []
    heap_counter = 0

    all_mean_attention = []
    per_case_stats = {}

    base_fieldnames = [
        "case",
        "edge_id",
        "src_node",
        "dst_node",
        "is_self_loop",

        "src_residue_a_idx",
        "src_residue_b_idx",
        "dst_residue_a_idx",
        "dst_residue_b_idx",

        "src_true_label",
        "dst_true_label",
        "src_pred_label",
        "dst_pred_label",

        "src_prob_class_1",
        "dst_prob_class_1",

        "mean_attention",
        "max_attention",

        "src_ca_distance",
        "src_degree_a",
        "src_degree_b",

        "dst_ca_distance",
        "dst_degree_a",
        "dst_degree_b",
    ]

    head_fieldnames = [f"head_{i}" for i in range(4)]
    fieldnames = base_fieldnames + head_fieldnames

    with open(OUTPUT_ATTENTION_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for data in test_dataset:
                case_name = data.case_name

                graph = data.to(device)

                out, attention_edge_index, attention_weights = model(
                    graph.x,
                    graph.edge_index,
                    return_attention=True,
                )

                probs = F.softmax(out, dim=1)
                prob_1 = probs[:, 1].cpu().numpy()
                pred = (prob_1 >= threshold).astype(np.int64)

                true = graph.y.cpu().numpy()
                pairs = graph.pairs.cpu().numpy()
                raw_basic_features = graph.raw_basic_features.cpu().numpy()

                attention_edge_index = attention_edge_index.cpu().numpy()
                attention_weights = attention_weights.cpu().numpy()

                edge_count = attention_edge_index.shape[1]

                case_attention_values = []

                for edge_id in range(edge_count):
                    src_node = int(attention_edge_index[0, edge_id])
                    dst_node = int(attention_edge_index[1, edge_id])

                    attention_values = attention_weights[edge_id]
                    mean_attention = float(np.mean(attention_values))
                    max_attention = float(np.max(attention_values))

                    all_mean_attention.append(mean_attention)
                    case_attention_values.append(mean_attention)

                    row = build_attention_row(
                        case_name=case_name,
                        edge_id=edge_id,
                        src_node=src_node,
                        dst_node=dst_node,
                        attention_values=attention_values,
                        mean_attention=mean_attention,
                        max_attention=max_attention,

                        src_pair=pairs[src_node],
                        dst_pair=pairs[dst_node],

                        src_true=true[src_node],
                        dst_true=true[dst_node],

                        src_prob_1=prob_1[src_node],
                        dst_prob_1=prob_1[dst_node],

                        src_pred=pred[src_node],
                        dst_pred=pred[dst_node],

                        src_raw_features=raw_basic_features[src_node],
                        dst_raw_features=raw_basic_features[dst_node],
                    )

                    writer.writerow(row)

                    heap_item = (mean_attention, heap_counter, row)
                    heap_counter += 1

                    if len(top_heap) < top_k:
                        heapq.heappush(top_heap, heap_item)
                    else:
                        heapq.heappushpop(top_heap, heap_item)

                case_attention_values = np.array(case_attention_values)

                per_case_stats[case_name] = {
                    "attention_edges": int(edge_count),
                    "mean_attention": float(case_attention_values.mean()),
                    "std_attention": float(case_attention_values.std()),
                    "min_attention": float(case_attention_values.min()),
                    "max_attention": float(case_attention_values.max()),
                }

                print(
                    f"{case_name}: saved {edge_count} attention edges | "
                    f"mean={per_case_stats[case_name]['mean_attention']:.6f} | "
                    f"max={per_case_stats[case_name]['max_attention']:.6f}"
                )

    top_rows = [
        item[2]
        for item in sorted(top_heap, key=lambda x: x[0], reverse=True)
    ]

    with open(OUTPUT_TOP_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top_rows)

    print(f"Saved: {OUTPUT_ATTENTION_CSV}")
    print(f"Saved: {OUTPUT_TOP_CSV}")

    return all_mean_attention, per_case_stats, top_rows


# -----------------------
# Plotting
# -----------------------
def plot_attention_distribution(mean_attention_values):
    os.makedirs(FIGURE_DIR, exist_ok=True)

    values = np.array(mean_attention_values)

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=50)
    plt.title("GAT Attention Weight Distribution")
    plt.xlabel("Mean attention weight across heads")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {OUTPUT_FIGURE}")


# -----------------------
# Markdown summary
# -----------------------
def save_attention_summary(
    best_epoch,
    best_threshold,
    best_val_f1,
    test_metrics,
    per_case_stats,
    top_rows,
):
    with open(OUTPUT_SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("# GAT Attention Visualization Summary\n\n")

        f.write("This analysis extracts first-layer GAT attention weights from the current best strict protocol model.\n\n")

        f.write("```text\n")
        f.write("Model: GAT\n")
        f.write("Features: basic 3 features\n")
        f.write("Input dimension: 3\n")
        f.write("Split: train/validation/test\n")
        f.write("Attention layer: first GATConv layer\n")
        f.write("Attention aggregation: mean across heads\n")
        f.write("```\n\n")

        f.write("## Training Selection\n\n")
        f.write(f"- Best epoch: `{best_epoch}`\n")
        f.write(f"- Best validation threshold: `{best_threshold:.2f}`\n")
        f.write(f"- Best validation F1 for class 1: `{best_val_f1:.4f}`\n\n")

        f.write("## Test Metrics\n\n")
        f.write("| Precision 1 | Recall 1 | F1 1 | Accuracy |\n")
        f.write("|-------------|----------|------|----------|\n")
        f.write(
            f"| {test_metrics['precision_1']:.4f} "
            f"| {test_metrics['recall_1']:.4f} "
            f"| {test_metrics['f1_1']:.4f} "
            f"| {test_metrics['accuracy']:.4f} |\n\n"
        )

        f.write("## Per-Test-Graph Attention Statistics\n\n")
        f.write("| Case | Attention Edges | Mean Attention | Std Attention | Min Attention | Max Attention |\n")
        f.write("|------|-----------------|----------------|---------------|---------------|---------------|\n")

        for case_name, stats in per_case_stats.items():
            f.write(
                f"| {case_name} "
                f"| {stats['attention_edges']} "
                f"| {stats['mean_attention']:.6f} "
                f"| {stats['std_attention']:.6f} "
                f"| {stats['min_attention']:.6f} "
                f"| {stats['max_attention']:.6f} |\n"
            )

        f.write("\n## Top Attention Edges\n\n")
        f.write(
            "| Rank | Case | Src Node | Dst Node | Self Loop | "
            "Mean Attention | Max Attention | "
            "Src True | Dst True | Src P1 | Dst P1 | "
            "Src Residue Pair | Dst Residue Pair |\n"
        )
        f.write(
            "|------|------|----------|----------|-----------|"
            "----------------|---------------|"
            "----------|----------|--------|--------|"
            "------------------|------------------|\n"
        )

        for rank, row in enumerate(top_rows[:20], start=1):
            src_pair = f"({row['src_residue_a_idx']}, {row['src_residue_b_idx']})"
            dst_pair = f"({row['dst_residue_a_idx']}, {row['dst_residue_b_idx']})"

            f.write(
                f"| {rank} "
                f"| {row['case']} "
                f"| {row['src_node']} "
                f"| {row['dst_node']} "
                f"| {row['is_self_loop']} "
                f"| {row['mean_attention']:.6f} "
                f"| {row['max_attention']:.6f} "
                f"| {row['src_true_label']} "
                f"| {row['dst_true_label']} "
                f"| {row['src_prob_class_1']:.4f} "
                f"| {row['dst_prob_class_1']:.4f} "
                f"| {src_pair} "
                f"| {dst_pair} |\n"
            )

        f.write("\n## Output Files\n\n")
        f.write(f"- Full attention table: `{OUTPUT_ATTENTION_CSV}`\n")
        f.write(f"- Top attention edges: `{OUTPUT_TOP_CSV}`\n")
        f.write(f"- Attention distribution figure: `{OUTPUT_FIGURE}`\n")
        f.write(f"- Summary: `{OUTPUT_SUMMARY_MD}`\n\n")

        f.write("## Notes\n\n")
        f.write(
            "GAT attention weights are normalized over incoming neighborhoods. "
            "Therefore, attention values should be interpreted locally rather than as global importance scores.\n"
        )

    print(f"Saved: {OUTPUT_SUMMARY_MD}")


# -----------------------
# Main
# -----------------------
def main():
    os.makedirs("experiments", exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)

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

    print("\nAttention visualization model:")
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

    test_metrics = evaluate_test_dataset(
        model=model,
        test_dataset=test_dataset,
        device=device,
        threshold=best_threshold,
    )

    print("\nFinal Test Metrics:")
    print(
        f"P1={test_metrics['precision_1']:.4f} | "
        f"R1={test_metrics['recall_1']:.4f} | "
        f"F1={test_metrics['f1_1']:.4f} | "
        f"Acc={test_metrics['accuracy']:.4f}"
    )

    mean_attention_values, per_case_stats, top_rows = extract_and_save_attention(
        model=model,
        test_dataset=test_dataset,
        device=device,
        threshold=best_threshold,
        top_k=TOP_K,
    )

    plot_attention_distribution(mean_attention_values)

    save_attention_summary(
        best_epoch=best_epoch,
        best_threshold=best_threshold,
        best_val_f1=best_val_f1,
        test_metrics=test_metrics,
        per_case_stats=per_case_stats,
        top_rows=top_rows,
    )

    print("\nGAT attention visualization completed successfully.")


if __name__ == "__main__":
    main()