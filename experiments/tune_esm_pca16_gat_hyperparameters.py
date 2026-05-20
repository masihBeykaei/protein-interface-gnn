import os
import csv
import json
import argparse
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


# ----------------------------
# Reproducibility
# ----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Data loading
# ----------------------------
def load_split_cases(processed_dir):
    split_path = os.path.join(processed_dir, "split_cases.json")

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Missing split_cases.json: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_graph(processed_dir, case_name):
    x = np.load(os.path.join(processed_dir, f"{case_name}_corr_features.npy"))
    y = np.load(os.path.join(processed_dir, f"{case_name}_corr_labels.npy"))
    edge_index = np.load(os.path.join(processed_dir, f"{case_name}_corr_edge_index.npy"))

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )

    return data


def load_graphs(processed_dir, case_names, device):
    graphs = {}

    for case_name in case_names:
        data = load_graph(processed_dir, case_name).to(device)
        graphs[case_name] = data

    return graphs


def get_input_dim(graphs):
    first_graph = next(iter(graphs.values()))
    return int(first_graph.x.shape[1])


def summarize_graphs(graphs):
    rows = []

    for case_name, graph in graphs.items():
        labels = graph.y.detach().cpu().numpy()
        pos = int(labels.sum())
        neg = int(len(labels) - pos)

        rows.append({
            "case": case_name,
            "nodes": int(graph.num_nodes),
            "edges": int(graph.edge_index.shape[1]),
            "positive": pos,
            "negative": neg,
            "feature_dim": int(graph.x.shape[1]),
        })

    return rows


# ----------------------------
# Model
# ----------------------------
class TunableGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, dropout):
        super().__init__()

        self.dropout = dropout

        self.gat1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
        )

        self.gat2 = GATConv(
            in_channels=hidden_channels * heads,
            out_channels=2,
            heads=1,
            dropout=dropout,
            concat=False,
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x


# ----------------------------
# Metrics
# ----------------------------
@torch.no_grad()
def collect_predictions(model, graphs):
    model.eval()

    y_true_all = []
    prob_all = []

    for graph in graphs.values():
        out = model(graph.x, graph.edge_index)
        probs = F.softmax(out, dim=1)[:, 1]

        y_true_all.append(graph.y.detach().cpu().numpy())
        prob_all.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(y_true_all)
    probs = np.concatenate(prob_all)

    return y_true, probs


def metrics_from_probs(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(np.int64)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "p0": float(precision[0]),
        "r0": float(recall[0]),
        "f0": float(f1[0]),
        "p1": float(precision[1]),
        "r1": float(recall[1]),
        "f1": float(f1[1]),
        "acc": float(acc),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def find_best_threshold(y_true, probs, thresholds):
    best = None

    for threshold in thresholds:
        result = metrics_from_probs(y_true, probs, threshold)
        result["threshold"] = float(threshold)

        if best is None:
            best = result
            continue

        # Main objective: maximize positive-class F1.
        # Tie-breakers: higher precision, then higher accuracy.
        if (
            result["f1"] > best["f1"]
            or (
                result["f1"] == best["f1"]
                and result["p1"] > best["p1"]
            )
            or (
                result["f1"] == best["f1"]
                and result["p1"] == best["p1"]
                and result["acc"] > best["acc"]
            )
        ):
            best = result

    return best


# ----------------------------
# Training
# ----------------------------
def train_one_config(
    config,
    input_dim,
    train_graphs,
    val_graphs,
    test_graphs,
    device,
    args,
):
    set_seed(args.seed)

    model = TunableGAT(
        in_channels=input_dim,
        hidden_channels=config["hidden_channels"],
        heads=config["heads"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + 1e-9,
        args.threshold_step,
    )

    best_state = None
    best_val = None
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(args.max_epochs + 1):
        model.train()
        total_loss = 0.0

        for graph in train_graphs.values():
            optimizer.zero_grad()

            out = model(graph.x, graph.edge_index)
            loss = F.cross_entropy(out, graph.y)

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_graphs))

        if epoch % args.eval_every != 0:
            continue

        val_true, val_probs = collect_predictions(model, val_graphs)
        val_best = find_best_threshold(val_true, val_probs, thresholds)

        improved = best_val is None or val_best["f1"] > best_val["f1"]

        if improved:
            best_val = val_best
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += args.eval_every

        print(
            f"hidden={config['hidden_channels']} heads={config['heads']} "
            f"dropout={config['dropout']} lr={config['lr']} wd={config['weight_decay']} "
            f"| Epoch {epoch} | Loss={avg_loss:.4f} "
            f"| Val_F1_1={val_best['f1']:.4f} "
            f"| Best_Val_F1_1={best_val['f1']:.4f} "
            f"| Best_Threshold={best_val['threshold']:.2f}"
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping at epoch {epoch} for "
                f"hidden={config['hidden_channels']}, heads={config['heads']}, "
                f"dropout={config['dropout']}, lr={config['lr']}, "
                f"weight_decay={config['weight_decay']}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_true, test_probs = collect_predictions(model, test_graphs)
    test_result = metrics_from_probs(
        test_true,
        test_probs,
        threshold=best_val["threshold"],
    )

    row = {
        "hidden_channels": config["hidden_channels"],
        "heads": config["heads"],
        "dropout": config["dropout"],
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
        "best_epoch": best_epoch,
        "threshold": best_val["threshold"],
        "val_p1": best_val["p1"],
        "val_r1": best_val["r1"],
        "val_f1": best_val["f1"],
        "val_acc": best_val["acc"],
        "test_p1": test_result["p1"],
        "test_r1": test_result["r1"],
        "test_f1": test_result["f1"],
        "test_acc": test_result["acc"],
        "tn": test_result["tn"],
        "fp": test_result["fp"],
        "fn": test_result["fn"],
        "tp": test_result["tp"],
    }

    print("\nResult:")
    print(
        f"hidden={row['hidden_channels']} | heads={row['heads']} | "
        f"dropout={row['dropout']} | lr={row['lr']} | wd={row['weight_decay']} | "
        f"Val_F1_1={row['val_f1']:.4f} | "
        f"Test_P1={row['test_p1']:.4f} | Test_R1={row['test_r1']:.4f} | "
        f"Test_F1_1={row['test_f1']:.4f} | Test_Acc={row['test_acc']:.4f} | "
        f"FP={row['fp']} | TP={row['tp']}"
    )

    return row


# ----------------------------
# Output
# ----------------------------
def save_csv(path, rows):
    fieldnames = [
        "hidden_channels",
        "heads",
        "dropout",
        "lr",
        "weight_decay",
        "best_epoch",
        "threshold",
        "val_p1",
        "val_r1",
        "val_f1",
        "val_acc",
        "test_p1",
        "test_r1",
        "test_f1",
        "test_acc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def save_md(path, rows, args, input_dim, train_cases, val_cases, test_cases):
    best_by_val = max(rows, key=lambda r: r["val_f1"])
    best_by_test = max(rows, key=lambda r: r["test_f1"])

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# ESM-2 PCA16 GAT Hyperparameter Tuning Results\n\n")

        f.write("This experiment tunes GAT hyperparameters for the combined current + BM5 dataset with ESM-2 PCA16 features.\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Processed directory: `{args.processed_dir}`\n")
        f.write(f"- Input dimension: `{input_dim}`\n")
        f.write(f"- Max epochs: `{args.max_epochs}`\n")
        f.write(f"- Patience: `{args.patience}`\n")
        f.write(f"- Eval every: `{args.eval_every}` epochs\n")
        f.write(f"- Threshold range: `{args.threshold_min}` to `{args.threshold_max}` step `{args.threshold_step}`\n")
        f.write(f"- Seed: `{args.seed}`\n\n")

        f.write("## Split\n\n")
        f.write(f"- Train: {', '.join(train_cases)}\n")
        f.write(f"- Validation: {', '.join(val_cases)}\n")
        f.write(f"- Test: {', '.join(test_cases)}\n\n")

        f.write("## Results\n\n")
        f.write(
            "| Hidden | Heads | Dropout | LR | Weight Decay | Best Epoch | Threshold | "
            "Val F1 | Test P1 | Test R1 | Test F1 | Test Acc | TN | FP | FN | TP |\n"
        )
        f.write(
            "|-------:|------:|--------:|---:|-------------:|-----------:|----------:|"
            "-------:|--------:|--------:|--------:|---------:|---:|---:|---:|---:|\n"
        )

        for row in rows:
            f.write(
                f"| {row['hidden_channels']} "
                f"| {row['heads']} "
                f"| {row['dropout']} "
                f"| {row['lr']} "
                f"| {row['weight_decay']} "
                f"| {row['best_epoch']} "
                f"| {row['threshold']:.2f} "
                f"| {row['val_f1']:.4f} "
                f"| {row['test_p1']:.4f} "
                f"| {row['test_r1']:.4f} "
                f"| {row['test_f1']:.4f} "
                f"| {row['test_acc']:.4f} "
                f"| {row['tn']} "
                f"| {row['fp']} "
                f"| {row['fn']} "
                f"| {row['tp']} |\n"
            )

        f.write("\n## Best by Validation F1\n\n")
        f.write(
            f"- hidden={best_by_val['hidden_channels']}, heads={best_by_val['heads']}, "
            f"dropout={best_by_val['dropout']}, lr={best_by_val['lr']}, "
            f"weight_decay={best_by_val['weight_decay']}\n"
        )
        f.write(
            f"- Val F1 1 = `{best_by_val['val_f1']:.4f}`\n"
            f"- Test F1 1 = `{best_by_val['test_f1']:.4f}`\n"
            f"- Threshold = `{best_by_val['threshold']:.2f}`\n"
        )

        f.write("\n## Best by Test F1\n\n")
        f.write("Reported for analysis only. Model selection should be based on validation F1.\n\n")
        f.write(
            f"- hidden={best_by_test['hidden_channels']}, heads={best_by_test['heads']}, "
            f"dropout={best_by_test['dropout']}, lr={best_by_test['lr']}, "
            f"weight_decay={best_by_test['weight_decay']}\n"
        )
        f.write(
            f"- Val F1 1 = `{best_by_test['val_f1']:.4f}`\n"
            f"- Test F1 1 = `{best_by_test['test_f1']:.4f}`\n"
            f"- Threshold = `{best_by_test['threshold']:.2f}`\n"
        )

    print(f"Saved: {path}")


# ----------------------------
# Config grid
# ----------------------------
def get_search_space():
    return [
        # Current best-style baseline, but with fine threshold sweep.
        {"hidden_channels": 16, "heads": 4, "dropout": 0.2, "lr": 0.005, "weight_decay": 5e-4},

        # More regularization: likely useful for ESM features.
        {"hidden_channels": 16, "heads": 4, "dropout": 0.3, "lr": 0.005, "weight_decay": 5e-4},
        {"hidden_channels": 16, "heads": 4, "dropout": 0.4, "lr": 0.005, "weight_decay": 1e-3},

        # Smaller model: may reduce false positives.
        {"hidden_channels": 8, "heads": 4, "dropout": 0.3, "lr": 0.005, "weight_decay": 1e-3},
        {"hidden_channels": 8, "heads": 2, "dropout": 0.3, "lr": 0.005, "weight_decay": 1e-3},

        # Fewer heads: less aggressive attention capacity.
        {"hidden_channels": 16, "heads": 2, "dropout": 0.3, "lr": 0.005, "weight_decay": 1e-3},

        # Larger but more regularized model.
        {"hidden_channels": 32, "heads": 2, "dropout": 0.4, "lr": 0.003, "weight_decay": 1e-3},
        {"hidden_channels": 32, "heads": 4, "dropout": 0.4, "lr": 0.003, "weight_decay": 2e-3},
    ]


# ----------------------------
# Main
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune GAT hyperparameters for ESM-2 PCA16 combined current + BM5 dataset."
    )

    parser.add_argument(
        "--processed_dir",
        default=os.path.join("data", "processed_combined_current_bm5_esm2_pca16"),
    )

    parser.add_argument(
        "--output_csv",
        default=os.path.join("experiments", "esm2_pca16_gat_hyperparameter_tuning_results.csv"),
    )

    parser.add_argument(
        "--output_md",
        default=os.path.join("experiments", "esm2_pca16_gat_hyperparameter_tuning_results.md"),
    )

    parser.add_argument("--max_epochs", type=int, default=220)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--eval_every", type=int, default=5)

    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.60)
    parser.add_argument("--threshold_step", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_cases = load_split_cases(args.processed_dir)
    train_cases = split_cases["train"]
    val_cases = split_cases["val"]
    test_cases = split_cases["test"]

    print("\nESM-2 PCA16 GAT hyperparameter tuning")
    print(f"Processed dir: {args.processed_dir}")
    print("Train:", train_cases)
    print("Validation:", val_cases)
    print("Test:", test_cases)
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_graphs = load_graphs(args.processed_dir, train_cases, device)
    val_graphs = load_graphs(args.processed_dir, val_cases, device)
    test_graphs = load_graphs(args.processed_dir, test_cases, device)

    input_dim = get_input_dim(train_graphs)
    print(f"Input feature dimension: {input_dim}")

    print("\nLoaded train graphs:")
    for item in summarize_graphs(train_graphs):
        print(
            f"{item['case']}: nodes={item['nodes']}, edges={item['edges']}, "
            f"positive={item['positive']}, negative={item['negative']}, "
            f"feature_dim={item['feature_dim']}"
        )

    search_space = get_search_space()

    rows = []

    for i, config in enumerate(search_space, start=1):
        print("\n" + "=" * 72)
        print(
            f"Running config {i}/{len(search_space)}: "
            f"hidden={config['hidden_channels']}, heads={config['heads']}, "
            f"dropout={config['dropout']}, lr={config['lr']}, "
            f"weight_decay={config['weight_decay']}"
        )
        print("=" * 72)

        row = train_one_config(
            config=config,
            input_dim=input_dim,
            train_graphs=train_graphs,
            val_graphs=val_graphs,
            test_graphs=test_graphs,
            device=device,
            args=args,
        )

        rows.append(row)

        save_csv(args.output_csv, rows)
        save_md(args.output_md, rows, args, input_dim, train_cases, val_cases, test_cases)

    best_by_val = max(rows, key=lambda r: r["val_f1"])

    print("\nBest configuration by validation F1:")
    print(
        f"hidden={best_by_val['hidden_channels']} | heads={best_by_val['heads']} | "
        f"dropout={best_by_val['dropout']} | lr={best_by_val['lr']} | "
        f"weight_decay={best_by_val['weight_decay']} | "
        f"threshold={best_by_val['threshold']:.2f} | "
        f"Val_F1_1={best_by_val['val_f1']:.4f} | "
        f"Test_F1_1={best_by_val['test_f1']:.4f}"
    )

    print("\nESM-2 PCA16 GAT hyperparameter tuning completed successfully.")


if __name__ == "__main__":
    main()
