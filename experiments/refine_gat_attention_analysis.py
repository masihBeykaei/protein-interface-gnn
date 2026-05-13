import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt


# -----------------------
# Config
# -----------------------
INPUT_ATTENTION_CSV = os.path.join("experiments", "gat_attention_weights.csv")

FIGURE_DIR = os.path.join("experiments", "figures")

OUTPUT_NON_SELF_CSV = os.path.join(
    "experiments",
    "gat_attention_top_non_self_edges.csv",
)

OUTPUT_PRED_POS_CSV = os.path.join(
    "experiments",
    "gat_attention_top_predicted_positive_edges.csv",
)

OUTPUT_TP_CONTEXT_CSV = os.path.join(
    "experiments",
    "gat_attention_top_tp_context_edges.csv",
)

OUTPUT_FP_CONTEXT_CSV = os.path.join(
    "experiments",
    "gat_attention_top_fp_context_edges.csv",
)

OUTPUT_FN_CONTEXT_CSV = os.path.join(
    "experiments",
    "gat_attention_top_fn_context_edges.csv",
)

OUTPUT_CONTEXT_CSV = os.path.join(
    "experiments",
    "gat_attention_error_context_edges.csv",
)

OUTPUT_LOG_FIGURE = os.path.join(
    FIGURE_DIR,
    "gat_attention_distribution_log.png",
)

OUTPUT_REFINED_SUMMARY = os.path.join(
    "experiments",
    "gat_attention_refined_summary.md",
)

TOP_K = 100


# -----------------------
# Helpers
# -----------------------
def ensure_dirs():
    os.makedirs("experiments", exist_ok=True)
    os.makedirs(FIGURE_DIR, exist_ok=True)


def to_int(row, key):
    return int(float(row[key]))


def to_float(row, key):
    return float(row[key])


def node_status(true_label, pred_label):
    if true_label == 1 and pred_label == 1:
        return "TP"

    if true_label == 0 and pred_label == 0:
        return "TN"

    if true_label == 0 and pred_label == 1:
        return "FP"

    if true_label == 1 and pred_label == 0:
        return "FN"

    return "UNKNOWN"


def annotate_row(row):
    """
    Add node-level prediction status for source and destination nodes.
    """
    row = dict(row)

    src_true = to_int(row, "src_true_label")
    src_pred = to_int(row, "src_pred_label")
    dst_true = to_int(row, "dst_true_label")
    dst_pred = to_int(row, "dst_pred_label")

    src_status = node_status(src_true, src_pred)
    dst_status = node_status(dst_true, dst_pred)

    context_labels = []

    if src_status != "TN":
        context_labels.append(f"src:{src_status}")

    if dst_status != "TN":
        context_labels.append(f"dst:{dst_status}")

    row["src_status"] = src_status
    row["dst_status"] = dst_status
    row["context_labels"] = ";".join(context_labels) if context_labels else "none"

    return row


def attention_key(row):
    """
    Unique edge key.
    Duplicate edges can appear in attention output, so we keep the highest-attention
    version for top-edge summaries.
    """
    return (
        row["case"],
        row["src_node"],
        row["dst_node"],
    )


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def top_by_attention(rows, top_k=100):
    return sorted(
        rows,
        key=lambda row: to_float(row, "mean_attention"),
        reverse=True,
    )[:top_k]


def row_has_predicted_positive(row):
    return (
        to_int(row, "src_pred_label") == 1
        or to_int(row, "dst_pred_label") == 1
    )


def row_has_status(row, status):
    return row["src_status"] == status or row["dst_status"] == status


def row_has_error_context(row):
    return (
        row["src_status"] in {"FP", "FN"}
        or row["dst_status"] in {"FP", "FN"}
    )


# -----------------------
# Load and refine attention rows
# -----------------------
def load_attention_rows():
    if not os.path.exists(INPUT_ATTENTION_CSV):
        raise FileNotFoundError(
            f"Missing input file: {INPUT_ATTENTION_CSV}. "
            "Run experiments/visualize_gat_attention.py first."
        )

    all_attention_values = []
    original_fieldnames = None

    unique_edges = {}
    total_rows = 0
    total_self_loops = 0
    total_non_self = 0

    with open(INPUT_ATTENTION_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_fieldnames = reader.fieldnames

        if original_fieldnames is None:
            raise RuntimeError("Input attention CSV has no header.")

        for row in reader:
            total_rows += 1

            annotated = annotate_row(row)

            mean_attention = to_float(annotated, "mean_attention")
            all_attention_values.append(mean_attention)

            if to_int(annotated, "is_self_loop") == 1:
                total_self_loops += 1
            else:
                total_non_self += 1

            key = attention_key(annotated)

            if key not in unique_edges:
                unique_edges[key] = annotated
            else:
                old_attention = to_float(unique_edges[key], "mean_attention")
                if mean_attention > old_attention:
                    unique_edges[key] = annotated

    extra_fields = [
        "src_status",
        "dst_status",
        "context_labels",
    ]

    fieldnames = original_fieldnames + [
        field for field in extra_fields
        if field not in original_fieldnames
    ]

    stats = {
        "total_attention_rows": total_rows,
        "total_self_loops": total_self_loops,
        "total_non_self": total_non_self,
        "unique_edges": len(unique_edges),
        "mean_attention_global": float(np.mean(all_attention_values)),
        "std_attention_global": float(np.std(all_attention_values)),
        "min_attention_global": float(np.min(all_attention_values)),
        "max_attention_global": float(np.max(all_attention_values)),
    }

    return list(unique_edges.values()), all_attention_values, fieldnames, stats


# -----------------------
# Plot log distribution
# -----------------------
def plot_log_attention_distribution(attention_values):
    values = np.array(attention_values, dtype=np.float64)

    # Attention values should be positive, but epsilon avoids log issues.
    epsilon = 1e-12
    log_values = np.log10(values + epsilon)

    plt.figure(figsize=(8, 5))
    plt.hist(log_values, bins=60)

    plt.title("GAT Attention Weight Distribution - Log Scale")
    plt.xlabel("log10(mean attention weight across heads)")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(OUTPUT_LOG_FIGURE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {OUTPUT_LOG_FIGURE}")


# -----------------------
# Markdown summary
# -----------------------
def save_refined_summary(
    stats,
    non_self_rows,
    pred_pos_rows,
    tp_rows,
    fp_rows,
    fn_rows,
    error_context_rows,
):
    with open(OUTPUT_REFINED_SUMMARY, "w", encoding="utf-8") as f:
        f.write("# Refined GAT Attention Analysis\n\n")

        f.write("This refinement post-processes the full GAT attention table and creates more interpretable attention subsets.\n\n")

        f.write("The original attention extraction used:\n\n")
        f.write("```text\n")
        f.write("Model: GAT\n")
        f.write("Feature set: basic 3 features\n")
        f.write("Attention layer: first GATConv layer\n")
        f.write("Attention aggregation: mean across heads\n")
        f.write("```\n\n")

        f.write("## Global Attention Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Total attention rows | {stats['total_attention_rows']} |\n")
        f.write(f"| Unique edges after deduplication | {stats['unique_edges']} |\n")
        f.write(f"| Self-loop rows | {stats['total_self_loops']} |\n")
        f.write(f"| Non-self rows | {stats['total_non_self']} |\n")
        f.write(f"| Global mean attention | {stats['mean_attention_global']:.6f} |\n")
        f.write(f"| Global std attention | {stats['std_attention_global']:.6f} |\n")
        f.write(f"| Global min attention | {stats['min_attention_global']:.6f} |\n")
        f.write(f"| Global max attention | {stats['max_attention_global']:.6f} |\n\n")

        f.write("## Refined Output Files\n\n")
        f.write("| File | Description |\n")
        f.write("|------|-------------|\n")
        f.write(f"| `{OUTPUT_NON_SELF_CSV}` | Top attention edges excluding self-loops |\n")
        f.write(f"| `{OUTPUT_PRED_POS_CSV}` | Top attention edges connected to predicted-positive nodes |\n")
        f.write(f"| `{OUTPUT_TP_CONTEXT_CSV}` | Top attention edges connected to true-positive nodes |\n")
        f.write(f"| `{OUTPUT_FP_CONTEXT_CSV}` | Top attention edges connected to false-positive nodes |\n")
        f.write(f"| `{OUTPUT_FN_CONTEXT_CSV}` | Top attention edges connected to false-negative nodes |\n")
        f.write(f"| `{OUTPUT_CONTEXT_CSV}` | Top attention edges connected to FP/FN error nodes |\n")
        f.write(f"| `{OUTPUT_LOG_FIGURE}` | Log-scale distribution of mean attention weights |\n\n")

        f.write("## Top Refined Subset Sizes\n\n")
        f.write("| Subset | Rows Saved |\n")
        f.write("|--------|------------|\n")
        f.write(f"| Non-self top edges | {len(non_self_rows)} |\n")
        f.write(f"| Predicted-positive top edges | {len(pred_pos_rows)} |\n")
        f.write(f"| TP-context top edges | {len(tp_rows)} |\n")
        f.write(f"| FP-context top edges | {len(fp_rows)} |\n")
        f.write(f"| FN-context top edges | {len(fn_rows)} |\n")
        f.write(f"| Error-context top edges | {len(error_context_rows)} |\n\n")

        f.write("## Interpretation Notes\n\n")
        f.write("- The non-self subset removes self-loops, which dominated the previous top-attention table.\n")
        f.write("- The predicted-positive subset focuses on edges connected to nodes predicted as interface/contact candidates.\n")
        f.write("- The TP/FP/FN context files make attention analysis more useful for understanding correct detections and errors.\n")
        f.write("- GAT attention is normalized locally over incoming neighborhoods, so it should not be interpreted as a global biological importance score.\n")
        f.write("- High attention can indicate local message-passing importance, but not necessarily true interface relevance.\n")

    print(f"Saved: {OUTPUT_REFINED_SUMMARY}")


# -----------------------
# Main
# -----------------------
def main():
    ensure_dirs()

    unique_rows, attention_values, fieldnames, stats = load_attention_rows()

    print("\nLoaded attention rows:")
    print(f"Total rows: {stats['total_attention_rows']}")
    print(f"Unique edges: {stats['unique_edges']}")
    print(f"Self-loop rows: {stats['total_self_loops']}")
    print(f"Non-self rows: {stats['total_non_self']}")
    print(f"Global max attention: {stats['max_attention_global']:.6f}")

    non_self_candidates = [
        row for row in unique_rows
        if to_int(row, "is_self_loop") == 0
    ]

    pred_pos_candidates = [
        row for row in unique_rows
        if row_has_predicted_positive(row)
    ]

    tp_candidates = [
        row for row in unique_rows
        if row_has_status(row, "TP")
    ]

    fp_candidates = [
        row for row in unique_rows
        if row_has_status(row, "FP")
    ]

    fn_candidates = [
        row for row in unique_rows
        if row_has_status(row, "FN")
    ]

    error_context_candidates = [
        row for row in unique_rows
        if row_has_error_context(row)
    ]

    non_self_rows = top_by_attention(non_self_candidates, TOP_K)
    pred_pos_rows = top_by_attention(pred_pos_candidates, TOP_K)
    tp_rows = top_by_attention(tp_candidates, TOP_K)
    fp_rows = top_by_attention(fp_candidates, TOP_K)
    fn_rows = top_by_attention(fn_candidates, TOP_K)
    error_context_rows = top_by_attention(error_context_candidates, TOP_K)

    write_csv(OUTPUT_NON_SELF_CSV, non_self_rows, fieldnames)
    write_csv(OUTPUT_PRED_POS_CSV, pred_pos_rows, fieldnames)
    write_csv(OUTPUT_TP_CONTEXT_CSV, tp_rows, fieldnames)
    write_csv(OUTPUT_FP_CONTEXT_CSV, fp_rows, fieldnames)
    write_csv(OUTPUT_FN_CONTEXT_CSV, fn_rows, fieldnames)
    write_csv(OUTPUT_CONTEXT_CSV, error_context_rows, fieldnames)

    plot_log_attention_distribution(attention_values)

    save_refined_summary(
        stats=stats,
        non_self_rows=non_self_rows,
        pred_pos_rows=pred_pos_rows,
        tp_rows=tp_rows,
        fp_rows=fp_rows,
        fn_rows=fn_rows,
        error_context_rows=error_context_rows,
    )

    print("\nRefined GAT attention analysis completed successfully.")


if __name__ == "__main__":
    main()