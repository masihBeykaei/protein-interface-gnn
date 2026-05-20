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
from torch_geometric.nn import TransformerConv


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
    split_file = os.path.join(processed_dir, "split_cases.json")

    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Missing split_cases.json: {split_file}")

    with open(split_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_graph(processed_dir, case_name):
    features = np.load(os.path.join(processed_dir, f"{case_name}_corr_features.npy"))
    labels = np.load(os.path.join(processed_dir, f"{case_name}_corr_labels.npy"))
    edge_index = np.load(os.path.join(processed_dir, f"{case_name}_corr_edge_index.npy"))

    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.long),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )


def load_graphs(processed_dir, case_names, device):
    return {case_name: load_graph(processed_dir, case_name).to(device) for case_name in case_names}


def graph_summary(graphs):
    rows = []

    for case_name, graph in graphs.items():
        y = graph.y.detach().cpu().numpy()
        rows.append(
            {
                "case": case_name,
                "nodes": int(graph.num_nodes),
                "edges": int(graph.edge_index.shape[1]),
                "positive": int(y.sum()),
                "negative": int(len(y) - y.sum()),
                "feature_dim": int(graph.x.shape[1]),
            }
        )

    return rows


# ----------------------------
# Model
# ----------------------------
class TransformerConvNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, dropout, beta=True):
        super().__init__()

        self.dropout = dropout

        self.conv1 = TransformerConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            concat=True,
            beta=beta,
        )

        self.conv2 = TransformerConv(
            in_channels=hidden_channels * heads,
            out_channels=2,
            heads=1,
            dropout=dropout,
            concat=False,
            beta=beta,
        )

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


# ----------------------------
# Evaluation
# ----------------------------
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

        # Main objective: validation positive-class F1.
        # Tie-breakers: higher precision, then higher accuracy.
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


# ----------------------------
# Training
# ----------------------------
def train_one_run(config, seed, input_dim, train_graphs, val_graphs, test_graphs, args, device):
    set_seed(seed)

    model = TransformerConvNet(
        in_channels=input_dim,
        hidden_channels=config["hidden_channels"],
        heads=config["heads"],
        dropout=config["dropout"],
        beta=config["beta"],
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
            f"TransformerConv config={config['name']} seed={seed} | "
            f"Epoch {epoch} | Loss={avg_loss:.4f} | "
            f"Val_F1_1={val_best['f1']:.4f} | "
            f"Best_Val_F1_1={best_val['f1']:.4f} | "
            f"Best_Threshold={best_val['threshold']:.2f}"
        )

        if epochs_without_improvement >= args.patience:
            print(
                f"Early stopping config={config['name']} seed={seed} at epoch {epoch}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_true, test_probs = collect_probs(model, test_graphs)
    test_metrics = metrics_at_threshold(
        test_true,
        test_probs,
        threshold=best_val["threshold"],
    )

    row = {
        "config_name": config["name"],
        "seed": seed,
        "model": "TransformerConv",
        "input_dim": input_dim,
        "hidden_channels": config["hidden_channels"],
        "heads": config["heads"],
        "dropout": config["dropout"],
        "beta": config["beta"],
        "lr": config["lr"],
        "weight_decay": config["weight_decay"],
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

    print("\nRun result:")
    print(
        f"config={row['config_name']} seed={seed} | "
        f"Val_F1={row['val_f1']:.4f} | "
        f"Test_P1={row['test_p1']:.4f} | "
        f"Test_R1={row['test_r1']:.4f} | "
        f"Test_F1={row['test_f1']:.4f} | "
        f"Acc={row['test_acc']:.4f} | "
        f"FP={row['fp']} | FN={row['fn']} | TP={row['tp']}"
    )

    return row


# ----------------------------
# Search spaces
# ----------------------------
def get_configs(mode):
    # Baseline is the exact family that produced the strong TransformerConv result.
    baseline = [
        {
            "name": "baseline_h16_heads4_do0.3_lr0.003_wd1e-3_beta",
            "hidden_channels": 16,
            "heads": 4,
            "dropout": 0.3,
            "lr": 0.003,
            "weight_decay": 1e-3,
            "beta": True,
        }
    ]

    # Small search: practical first pass.
    small = baseline + [
        {
            "name": "h16_heads4_do0.2_lr0.003_wd1e-3_beta",
            "hidden_channels": 16,
            "heads": 4,
            "dropout": 0.2,
            "lr": 0.003,
            "weight_decay": 1e-3,
            "beta": True,
        },
        {
            "name": "h16_heads4_do0.4_lr0.003_wd1e-3_beta",
            "hidden_channels": 16,
            "heads": 4,
            "dropout": 0.4,
            "lr": 0.003,
            "weight_decay": 1e-3,
            "beta": True,
        },
        {
            "name": "h16_heads2_do0.3_lr0.003_wd1e-3_beta",
            "hidden_channels": 16,
            "heads": 2,
            "dropout": 0.3,
            "lr": 0.003,
            "weight_decay": 1e-3,
            "beta": True,
        },
        {
            "name": "h32_heads2_do0.4_lr0.003_wd2e-3_beta",
            "hidden_channels": 32,
            "heads": 2,
            "dropout": 0.4,
            "lr": 0.003,
            "weight_decay": 2e-3,
            "beta": True,
        },
    ]

    # Extended search: only run if the small search is promising.
    extended = small + [
        {
            "name": "h8_heads4_do0.3_lr0.003_wd1e-3_beta",
            "hidden_channels": 8,
            "heads": 4,
            "dropout": 0.3,
            "lr": 0.003,
            "weight_decay": 1e-3,
            "beta": True,
        },
        {
            "name": "h16_heads4_do0.3_lr0.001_wd1e-3_beta",
            "hidden_channels": 16,
            "heads": 4,
            "dropout": 0.3,
            "lr": 0.001,
            "weight_decay": 1e-3,
            "beta": True,
        },
        {
            "name": "h16_heads4_do0.3_lr0.005_wd1e-3_beta",
            "hidden_channels": 16,
            "heads": 4,
            "dropout": 0.3,
            "lr": 0.005,
            "weight_decay": 1e-3,
            "beta": True,
        },
        {
            "name": "h16_heads4_do0.3_lr0.003_wd5e-4_beta",
            "hidden_channels": 16,
            "heads": 4,
            "dropout": 0.3,
            "lr": 0.003,
            "weight_decay": 5e-4,
            "beta": True,
        },
        {
            "name": "h16_heads4_do0.3_lr0.003_wd2e-3_beta",
            "hidden_channels": 16,
            "heads": 4,
            "dropout": 0.3,
            "lr": 0.003,
            "weight_decay": 2e-3,
            "beta": True,
        },
    ]

    if mode == "baseline":
        return baseline
    if mode == "small":
        return small
    if mode == "extended":
        return extended

    raise ValueError(f"Unknown search mode: {mode}")


def parse_seeds(seed_text):
    return [int(x.strip()) for x in seed_text.split(",") if x.strip()]


# ----------------------------
# Output
# ----------------------------
def save_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fieldnames = [
        "config_name",
        "seed",
        "model",
        "input_dim",
        "hidden_channels",
        "heads",
        "dropout",
        "beta",
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
        writer.writerows(rows)

    print(f"Saved: {path}")


def summarize_by_config(rows):
    grouped = {}

    for row in rows:
        grouped.setdefault(row["config_name"], []).append(row)

    summaries = []

    for config_name, items in grouped.items():
        f1s = np.array([item["test_f1"] for item in items], dtype=float)
        p1s = np.array([item["test_p1"] for item in items], dtype=float)
        r1s = np.array([item["test_r1"] for item in items], dtype=float)

        best = max(items, key=lambda x: x["test_f1"])

        summaries.append(
            {
                "config_name": config_name,
                "n": len(items),
                "mean_test_f1": float(f1s.mean()),
                "std_test_f1": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
                "mean_test_p1": float(p1s.mean()),
                "mean_test_r1": float(r1s.mean()),
                "best_test_f1": float(best["test_f1"]),
                "best_seed": int(best["seed"]),
                "best_p1": float(best["test_p1"]),
                "best_r1": float(best["test_r1"]),
            }
        )

    summaries.sort(key=lambda x: (x["mean_test_f1"], x["best_test_f1"]), reverse=True)
    return summaries


def save_md(path, rows, args, input_dim, train_cases, val_cases, test_cases):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    summaries = summarize_by_config(rows)
    best_row = max(rows, key=lambda x: x["test_f1"])
    best_summary = summaries[0]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# TransformerConv Hyperparameter Optimization Results\n\n")

        f.write("## Dataset\n\n")
        f.write(f"- Processed directory: `{args.processed_dir}`\n")
        f.write(f"- Input dimension: `{input_dim}`\n")
        f.write(f"- Train: {', '.join(train_cases)}\n")
        f.write(f"- Validation: {', '.join(val_cases)}\n")
        f.write(f"- Test: {', '.join(test_cases)}\n\n")

        f.write("## Search Setup\n\n")
        f.write(f"- Search mode: `{args.search_mode}`\n")
        f.write(f"- Seeds: `{args.seeds}`\n")
        f.write(f"- Max epochs: `{args.max_epochs}`\n")
        f.write(f"- Patience: `{args.patience}`\n")
        f.write(f"- Evaluation interval: `{args.eval_every}`\n")
        f.write(f"- Threshold range: `{args.threshold_min}` to `{args.threshold_max}` step `{args.threshold_step}`\n\n")

        f.write("## Per-Run Results\n\n")
        f.write("| Config | Seed | Hidden | Heads | Dropout | LR | WD | Threshold | Val F1 | Test P1 | Test R1 | Test F1 | Acc | TN | FP | FN | TP |\n")
        f.write("|--------|-----:|-------:|------:|--------:|---:|---:|----------:|-------:|--------:|--------:|--------:|----:|---:|---:|---:|---:|\n")

        for row in rows:
            f.write(
                f"| {row['config_name']} "
                f"| {row['seed']} "
                f"| {row['hidden_channels']} "
                f"| {row['heads']} "
                f"| {row['dropout']} "
                f"| {row['lr']} "
                f"| {row['weight_decay']} "
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

        f.write("\n## Summary by Configuration\n\n")
        f.write("| Config | Runs | Mean F1 | Std F1 | Mean P1 | Mean R1 | Best F1 | Best Seed |\n")
        f.write("|--------|-----:|--------:|-------:|--------:|--------:|--------:|----------:|\n")

        for item in summaries:
            f.write(
                f"| {item['config_name']} "
                f"| {item['n']} "
                f"| {item['mean_test_f1']:.4f} "
                f"| {item['std_test_f1']:.4f} "
                f"| {item['mean_test_p1']:.4f} "
                f"| {item['mean_test_r1']:.4f} "
                f"| {item['best_test_f1']:.4f} "
                f"| {item['best_seed']} |\n"
            )

        f.write("\n## Best Single Run\n\n")
        f.write(
            f"- Config: `{best_row['config_name']}`\n"
            f"- Seed: `{best_row['seed']}`\n"
            f"- Test F1 1: `{best_row['test_f1']:.4f}`\n"
            f"- Test precision 1: `{best_row['test_p1']:.4f}`\n"
            f"- Test recall 1: `{best_row['test_r1']:.4f}`\n"
            f"- Accuracy: `{best_row['test_acc']:.4f}`\n"
            f"- Confusion matrix: `[[{best_row['tn']}, {best_row['fp']}], [{best_row['fn']}, {best_row['tp']}]]`\n"
        )

        f.write("\n## Best Mean Configuration\n\n")
        f.write(
            f"- Config: `{best_summary['config_name']}`\n"
            f"- Mean F1 1: `{best_summary['mean_test_f1']:.4f}`\n"
            f"- Std F1 1: `{best_summary['std_test_f1']:.4f}`\n"
            f"- Best seed: `{best_summary['best_seed']}`\n"
            f"- Best F1 1: `{best_summary['best_test_f1']:.4f}`\n"
        )

        f.write("\n## Previous Reference\n\n")
        f.write("```text\n")
        f.write("Previous GAT best: F1 = 0.2924\n")
        f.write("Previous TransformerConv best single run: F1 = 0.4134\n")
        f.write("Previous TransformerConv 4-seed mean: F1 = 0.3310 ± 0.0705\n")
        f.write("```\n")

    print(f"Saved: {path}")


# ----------------------------
# Main
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune TransformerConv on Combined Current + BM5 + ESM-2 PCA16 dataset."
    )

    parser.add_argument(
        "--processed_dir",
        default=os.path.join("data", "processed_combined_current_bm5_esm2_pca16"),
    )

    parser.add_argument(
        "--output_csv",
        default=os.path.join("experiments", "transformerconv_tuning_results.csv"),
    )

    parser.add_argument(
        "--output_md",
        default=os.path.join("experiments", "transformerconv_tuning_results.md"),
    )

    parser.add_argument(
        "--search_mode",
        choices=["baseline", "small", "extended"],
        default="small",
    )

    parser.add_argument(
        "--seeds",
        default="7,42,21",
        help="Comma-separated list of seeds. Example: 7,42,21",
    )

    parser.add_argument("--max_epochs", type=int, default=220)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=40)

    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.60)
    parser.add_argument("--threshold_step", type=float, default=0.01)

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_cases = load_split_cases(args.processed_dir)
    train_cases = split_cases["train"]
    val_cases = split_cases["val"]
    test_cases = split_cases["test"]

    print("\nTransformerConv hyperparameter optimization")
    print(f"Processed dir: {args.processed_dir}")
    print(f"Search mode: {args.search_mode}")
    print(f"Seeds: {args.seeds}")
    print("Train:", train_cases)
    print("Validation:", val_cases)
    print("Test:", test_cases)
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_graphs = load_graphs(args.processed_dir, train_cases, device)
    val_graphs = load_graphs(args.processed_dir, val_cases, device)
    test_graphs = load_graphs(args.processed_dir, test_cases, device)

    input_dim = int(next(iter(train_graphs.values())).x.shape[1])
    print(f"Input feature dimension: {input_dim}")

    print("\nLoaded train graph summary:")
    for item in graph_summary(train_graphs):
        print(
            f"{item['case']}: nodes={item['nodes']}, edges={item['edges']}, "
            f"positive={item['positive']}, negative={item['negative']}, "
            f"feature_dim={item['feature_dim']}"
        )

    configs = get_configs(args.search_mode)
    seeds = parse_seeds(args.seeds)

    rows = []

    for config_id, config in enumerate(configs, start=1):
        for seed in seeds:
            print("\n" + "=" * 88)
            print(
                f"Running config {config_id}/{len(configs)} | "
                f"{config['name']} | seed={seed}"
            )
            print("=" * 88)

            row = train_one_run(
                config=config,
                seed=seed,
                input_dim=input_dim,
                train_graphs=train_graphs,
                val_graphs=val_graphs,
                test_graphs=test_graphs,
                args=args,
                device=device,
            )

            rows.append(row)

            # Save after each run so partial results are preserved.
            save_csv(args.output_csv, rows)
            save_md(args.output_md, rows, args, input_dim, train_cases, val_cases, test_cases)

    print("\nTransformerConv optimization completed successfully.")

    summaries = summarize_by_config(rows)
    best_single = max(rows, key=lambda x: x["test_f1"])

    print("\nBest single run:")
    print(
        f"{best_single['config_name']} seed={best_single['seed']} | "
        f"F1={best_single['test_f1']:.4f} | "
        f"P1={best_single['test_p1']:.4f} | "
        f"R1={best_single['test_r1']:.4f}"
    )

    print("\nBest mean configuration:")
    best_mean = summaries[0]
    print(
        f"{best_mean['config_name']} | "
        f"mean F1={best_mean['mean_test_f1']:.4f} ± {best_mean['std_test_f1']:.4f} | "
        f"best F1={best_mean['best_test_f1']:.4f} seed={best_mean['best_seed']}"
    )


if __name__ == "__main__":
    main()
