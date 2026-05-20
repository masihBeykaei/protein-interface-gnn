import os
import csv
import json
import random
import argparse
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_cases(processed_dir):
    split_file = os.path.join(processed_dir, "split_cases.json")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Missing split_cases.json: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_graph(processed_dir, case_name):
    x = np.load(os.path.join(processed_dir, f"{case_name}_corr_features.npy"))
    y = np.load(os.path.join(processed_dir, f"{case_name}_corr_labels.npy"))
    edge_index = np.load(os.path.join(processed_dir, f"{case_name}_corr_edge_index.npy"))

    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        y=torch.tensor(y, dtype=torch.long),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )


def load_graphs(processed_dir, case_names, device):
    return {case_name: load_graph(processed_dir, case_name).to(device) for case_name in case_names}


def print_graph_summary(title, graphs):
    print(f"\n{title}")

    for case_name, graph in graphs.items():
        y = graph.y.detach().cpu().numpy()
        pos = int(y.sum())
        neg = int(len(y) - pos)

        print(
            f"{case_name}: nodes={graph.num_nodes}, edges={graph.edge_index.shape[1]}, "
            f"positive={pos}, negative={neg}, feature_dim={graph.x.shape[1]}"
        )


class GraphSAGENet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, dropout=0.3, aggr="mean"):
        super().__init__()

        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr=aggr)
        self.conv2 = SAGEConv(hidden_channels, 2, aggr=aggr)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


@torch.no_grad()
def collect_probs(model, graphs):
    model.eval()

    y_true_list = []
    prob_list = []

    for graph in graphs.values():
        out = model(graph.x, graph.edge_index)
        prob = F.softmax(out, dim=1)[:, 1]
        y_true_list.append(graph.y.detach().cpu().numpy())
        prob_list.append(prob.detach().cpu().numpy())

    return np.concatenate(y_true_list), np.concatenate(prob_list)


def metrics_at_threshold(y_true, probs, threshold):
    pred = (probs >= threshold).astype(np.int64)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        pred,
        labels=[0, 1],
        zero_division=0,
    )

    acc = accuracy_score(y_true, pred)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    return {
        "threshold": float(threshold),
        "p1": float(precision[1]),
        "r1": float(recall[1]),
        "f1": float(f1[1]),
        "acc": float(acc),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def choose_best_threshold(y_true, probs, thresholds):
    best = None

    for threshold in thresholds:
        result = metrics_at_threshold(y_true, probs, threshold)

        if best is None:
            best = result
            continue

        if (
            result["f1"] > best["f1"]
            or (result["f1"] == best["f1"] and result["p1"] > best["p1"])
            or (
                result["f1"] == best["f1"]
                and result["p1"] == best["p1"]
                and result["acc"] > best["acc"]
            )
        ):
            best = result

    return best


def save_csv(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "model",
        "input_dim",
        "hidden_channels",
        "dropout",
        "aggr",
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

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    print(f"Saved: {path}")


def save_md(path, row, train_cases, val_cases, test_cases):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write("# GraphSAGE Experiment Result\n\n")

        f.write("## Dataset\n\n")
        f.write(f"- Train: {', '.join(train_cases)}\n")
        f.write(f"- Validation: {', '.join(val_cases)}\n")
        f.write(f"- Test: {', '.join(test_cases)}\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Model: `{row['model']}`\n")
        f.write(f"- Input dimension: `{row['input_dim']}`\n")
        f.write(f"- Hidden channels: `{row['hidden_channels']}`\n")
        f.write(f"- Aggregation: `{row['aggr']}`\n")
        f.write(f"- Dropout: `{row['dropout']}`\n")
        f.write(f"- Learning rate: `{row['lr']}`\n")
        f.write(f"- Weight decay: `{row['weight_decay']}`\n")
        f.write(f"- Best epoch: `{row['best_epoch']}`\n")
        f.write(f"- Selected threshold: `{row['threshold']:.2f}`\n\n")

        f.write("## Validation Metrics\n\n")
        f.write("| Precision 1 | Recall 1 | F1 1 | Accuracy |\n")
        f.write("|------------:|---------:|-----:|---------:|\n")
        f.write(
            f"| {row['val_p1']:.4f} | {row['val_r1']:.4f} "
            f"| {row['val_f1']:.4f} | {row['val_acc']:.4f} |\n\n"
        )

        f.write("## Test Metrics\n\n")
        f.write("| Precision 1 | Recall 1 | F1 1 | Accuracy |\n")
        f.write("|------------:|---------:|-----:|---------:|\n")
        f.write(
            f"| {row['test_p1']:.4f} | {row['test_r1']:.4f} "
            f"| {row['test_f1']:.4f} | {row['test_acc']:.4f} |\n\n"
        )

        f.write("## Confusion Matrix\n\n")
        f.write("| True / Pred | Pred 0 | Pred 1 |\n")
        f.write("|-------------|-------:|-------:|\n")
        f.write(f"| True 0 | {row['tn']} | {row['fp']} |\n")
        f.write(f"| True 1 | {row['fn']} | {row['tp']} |\n\n")

        f.write("## Comparison Target\n\n")
        f.write("```text\n")
        f.write("Current best: Combined Current + BM5 + Full Pair ESM-2 PCA16 + GAT\n")
        f.write("P1 = 0.2015\n")
        f.write("R1 = 0.5323\n")
        f.write("F1 = 0.2924\n")
        f.write("Acc = 0.9199\n")
        f.write("```\n")

    print(f"Saved: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GraphSAGE on the combined current + BM5 ESM-2 PCA16 dataset."
    )

    parser.add_argument(
        "--processed_dir",
        default=os.path.join("data", "processed_combined_current_bm5_esm2_pca16"),
    )

    parser.add_argument(
        "--output_csv",
        default=os.path.join("experiments", "combined_current_bm5_esm2_pca16_graphsage_results.csv"),
    )

    parser.add_argument(
        "--output_md",
        default=os.path.join("experiments", "combined_current_bm5_esm2_pca16_graphsage_results.md"),
    )

    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--aggr", type=str, default="mean", choices=["mean", "max", "sum"])
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--max_epochs", type=int, default=220)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=40)

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

    print("\nGraphSAGE experiment")
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

    print_graph_summary("Loaded train graphs:", train_graphs)

    input_dim = int(next(iter(train_graphs.values())).x.shape[1])
    print(f"\nInput feature dimension: {input_dim}")

    model = GraphSAGENet(
        in_channels=input_dim,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        aggr=args.aggr,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
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

        val_true, val_probs = collect_probs(model, val_graphs)
        val_best = choose_best_threshold(val_true, val_probs, thresholds)

        improved = best_val is None or val_best["f1"] > best_val["f1"]

        if improved:
            best_val = val_best
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += args.eval_every

        print(
            f"GraphSAGE | Epoch {epoch} | Loss={avg_loss:.4f} "
            f"| Val_F1_1={val_best['f1']:.4f} "
            f"| Best_Val_F1_1={best_val['f1']:.4f} "
            f"| Best_Threshold={best_val['threshold']:.2f}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"GraphSAGE early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_true, test_probs = collect_probs(model, test_graphs)
    test_metrics = metrics_at_threshold(test_true, test_probs, threshold=best_val["threshold"])

    row = {
        "model": "GraphSAGE",
        "input_dim": input_dim,
        "hidden_channels": args.hidden_channels,
        "dropout": args.dropout,
        "aggr": args.aggr,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "best_epoch": best_epoch,
        "threshold": best_val["threshold"],
        "val_p1": best_val["p1"],
        "val_r1": best_val["r1"],
        "val_f1": best_val["f1"],
        "val_acc": best_val["acc"],
        "test_p1": test_metrics["p1"],
        "test_r1": test_metrics["r1"],
        "test_f1": test_metrics["f1"],
        "test_acc": test_metrics["acc"],
        "tn": test_metrics["tn"],
        "fp": test_metrics["fp"],
        "fn": test_metrics["fn"],
        "tp": test_metrics["tp"],
    }

    print("\nFinal Test Metrics:")
    print(
        f"P1={row['test_p1']:.4f} | "
        f"R1={row['test_r1']:.4f} | "
        f"F1={row['test_f1']:.4f} | "
        f"Acc={row['test_acc']:.4f}"
    )

    print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
    print(np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]]))

    save_csv(args.output_csv, row)
    save_md(args.output_md, row, train_cases, val_cases, test_cases)

    print("\nGraphSAGE experiment completed successfully.")


if __name__ == "__main__":
    main()
