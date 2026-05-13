import os
import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# Output directory
# -----------------------
FIGURE_DIR = os.path.join("experiments", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)


# -----------------------
# Data
# -----------------------
dataset_summary = {
    "total_positive": 698,
    "total_negative": 20009,
}

feature_results = [
    {
        "feature_set": "Basic 3 features",
        "model": "GCN",
        "precision_1": 0.1940,
        "recall_1": 0.1722,
        "f1_1": 0.1825,
        "accuracy": 0.9488,
    },
    {
        "feature_set": "Basic 3 features",
        "model": "GAT",
        "precision_1": 0.1746,
        "recall_1": 0.3642,
        "f1_1": 0.2361,
        "accuracy": 0.9217,
    },
    {
        "feature_set": "AA one-hot",
        "model": "GCN",
        "precision_1": 0.1313,
        "recall_1": 0.2781,
        "f1_1": 0.1783,
        "accuracy": 0.9149,
    },
    {
        "feature_set": "AA one-hot",
        "model": "GAT",
        "precision_1": 0.1051,
        "recall_1": 0.7285,
        "f1_1": 0.1836,
        "accuracy": 0.7850,
    },
    {
        "feature_set": "Physicochemical",
        "model": "GCN",
        "precision_1": 0.2254,
        "recall_1": 0.1060,
        "f1_1": 0.1441,
        "accuracy": 0.9582,
    },
    {
        "feature_set": "Physicochemical",
        "model": "GAT",
        "precision_1": 0.1566,
        "recall_1": 0.2914,
        "f1_1": 0.2037,
        "accuracy": 0.9244,
    },
]

negative_ratio_results = {
    "GCN": {
        "ratios": [2, 3, 5, 10],
        "precision_1": [0.1442, 0.1643, 0.2059, 0.2653],
        "recall_1": [0.6225, 0.5298, 0.3245, 0.0861],
        "f1_1": [0.2341, 0.2508, 0.2519, 0.1300],
    },
    "GAT": {
        "ratios": [2, 3, 5, 10],
        "precision_1": [0.1107, 0.1165, 0.1274, 0.1985],
        "recall_1": [0.9007, 0.8874, 0.7483, 0.1788],
        "f1_1": [0.1972, 0.2060, 0.2177, 0.1882],
    },
}

threshold_results = {
    "GCN": {
        "thresholds": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        "precision_1": [0.0865, 0.1008, 0.1139, 0.1247, 0.1377, 0.1444, 0.1762, 0.2059, 0.2400, 0.2250, 0.0000, 0.0000],
        "recall_1": [0.9603, 0.9073, 0.8411, 0.7748, 0.7086, 0.6159, 0.4702, 0.3245, 0.1589, 0.0596, 0.0000, 0.0000],
        "f1_1": [0.1586, 0.1815, 0.2006, 0.2149, 0.2306, 0.2340, 0.2563, 0.2519, 0.1912, 0.0942, 0.0000, 0.0000],
    },
    "GAT": {
        "thresholds": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90],
        "precision_1": [0.0719, 0.0828, 0.0927, 0.1011, 0.1073, 0.1117, 0.1179, 0.1274, 0.1434, 0.1881, 0.0000, 0.0000],
        "recall_1": [0.9868, 0.9603, 0.9470, 0.9404, 0.9338, 0.9205, 0.8477, 0.7483, 0.4967, 0.1258, 0.0000, 0.0000],
        "f1_1": [0.1341, 0.1525, 0.1688, 0.1825, 0.1925, 0.1993, 0.2070, 0.2177, 0.2226, 0.1508, 0.0000, 0.0000],
    },
}


# -----------------------
# Plot helpers
# -----------------------
def save_current_figure(filename):
    path = os.path.join(FIGURE_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# -----------------------
# 1. Class imbalance chart
# -----------------------
def plot_class_imbalance():
    labels = ["Positive", "Negative"]
    values = [
        dataset_summary["total_positive"],
        dataset_summary["total_negative"],
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.title("Class Imbalance in Multi-Protein Dataset")
    plt.ylabel("Number of Correspondence Nodes")

    for i, value in enumerate(values):
        plt.text(i, value, f"{value:,}", ha="center", va="bottom")

    save_current_figure("class_imbalance.png")


# -----------------------
# 2. Feature-set F1 comparison
# -----------------------
def plot_feature_f1_comparison():
    labels = [f"{item['feature_set']}\n{item['model']}" for item in feature_results]
    f1_values = [item["f1_1"] for item in feature_results]

    x = np.arange(len(labels))

    plt.figure(figsize=(10, 5))
    plt.bar(x, f1_values)
    plt.title("Positive-Class F1 Comparison Across Feature Sets")
    plt.ylabel("F1-score for Class 1")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylim(0, max(f1_values) + 0.08)

    for i, value in enumerate(f1_values):
        plt.text(i, value, f"{value:.3f}", ha="center", va="bottom")

    save_current_figure("feature_set_f1_comparison.png")


# -----------------------
# 3. Feature-set precision/recall/F1 comparison
# -----------------------
def plot_feature_metrics_grouped():
    labels = [f"{item['feature_set']}\n{item['model']}" for item in feature_results]
    precision = [item["precision_1"] for item in feature_results]
    recall = [item["recall_1"] for item in feature_results]
    f1 = [item["f1_1"] for item in feature_results]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, precision, width, label="Precision 1")
    plt.bar(x, recall, width, label="Recall 1")
    plt.bar(x + width, f1, width, label="F1 1")

    plt.title("Precision / Recall / F1 Across Feature Sets")
    plt.ylabel("Score")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylim(0, 0.85)
    plt.legend()

    save_current_figure("feature_set_precision_recall_f1.png")


# -----------------------
# 4. Best strict model comparison
# -----------------------
def plot_best_strict_model_comparison():
    selected = [
        item for item in feature_results
        if item["feature_set"] == "Basic 3 features"
    ]

    labels = [item["model"] for item in selected]
    precision = [item["precision_1"] for item in selected]
    recall = [item["recall_1"] for item in selected]
    f1 = [item["f1_1"] for item in selected]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(7, 5))
    plt.bar(x - width, precision, width, label="Precision 1")
    plt.bar(x, recall, width, label="Recall 1")
    plt.bar(x + width, f1, width, label="F1 1")

    plt.title("Best Strict Protocol: GCN vs GAT")
    plt.ylabel("Score")
    plt.xticks(x, labels)
    plt.ylim(0, 0.45)
    plt.legend()

    save_current_figure("best_strict_gcn_vs_gat.png")


# -----------------------
# 5. Negative ratio tuning curves
# -----------------------
def plot_negative_ratio_tuning():
    for model_name, data in negative_ratio_results.items():
        ratios = data["ratios"]

        plt.figure(figsize=(8, 5))
        plt.plot(ratios, data["precision_1"], marker="o", label="Precision 1")
        plt.plot(ratios, data["recall_1"], marker="o", label="Recall 1")
        plt.plot(ratios, data["f1_1"], marker="o", label="F1 1")

        plt.title(f"Negative Ratio Tuning - {model_name}")
        plt.xlabel("Negative Ratio")
        plt.ylabel("Score")
        plt.xticks(ratios)
        plt.ylim(0, 1.0)
        plt.legend()

        save_current_figure(f"negative_ratio_tuning_{model_name.lower()}.png")


# -----------------------
# 6. Threshold tuning curves
# -----------------------
def plot_threshold_tuning():
    for model_name, data in threshold_results.items():
        thresholds = data["thresholds"]

        plt.figure(figsize=(9, 5))
        plt.plot(thresholds, data["precision_1"], marker="o", label="Precision 1")
        plt.plot(thresholds, data["recall_1"], marker="o", label="Recall 1")
        plt.plot(thresholds, data["f1_1"], marker="o", label="F1 1")

        plt.title(f"Probability Threshold Tuning - {model_name}")
        plt.xlabel("Probability Threshold")
        plt.ylabel("Score")
        plt.xticks(thresholds, rotation=45)
        plt.ylim(0, 1.0)
        plt.legend()

        save_current_figure(f"threshold_tuning_{model_name.lower()}.png")


# -----------------------
# 7. Write figure index
# -----------------------
def write_figure_index():
    path = os.path.join(FIGURE_DIR, "README.md")

    content = """# Experiment Figures

This directory contains generated figures for the protein-interface GNN experiments.

## Figures

| File | Description |
|------|-------------|
| `class_imbalance.png` | Positive vs negative correspondence node counts |
| `feature_set_f1_comparison.png` | Positive-class F1 comparison across feature sets |
| `feature_set_precision_recall_f1.png` | Precision, recall, and F1 comparison across feature sets |
| `best_strict_gcn_vs_gat.png` | GCN vs GAT under the strict train/validation/test protocol |
| `negative_ratio_tuning_gcn.png` | Negative sampling ratio tuning for GCN |
| `negative_ratio_tuning_gat.png` | Negative sampling ratio tuning for GAT |
| `threshold_tuning_gcn.png` | Probability threshold tuning for GCN |
| `threshold_tuning_gat.png` | Probability threshold tuning for GAT |
"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Saved: {path}")


# -----------------------
# Main
# -----------------------
def main():
    plot_class_imbalance()
    plot_feature_f1_comparison()
    plot_feature_metrics_grouped()
    plot_best_strict_model_comparison()
    plot_negative_ratio_tuning()
    plot_threshold_tuning()
    write_figure_index()

    print("\nAll figures generated successfully.")


if __name__ == "__main__":
    main()