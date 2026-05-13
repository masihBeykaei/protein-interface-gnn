import os
import csv
import copy
import random
import numpy as np
import torch
import torch.nn.functional as F

from Bio.PDB import PDBParser
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix


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

TOP_K_PER_TYPE = 10

PROCESSED_DIR = os.path.join("data", "processed")
RAW_PDB_DIR = os.path.join("data", "raw_pdb")

OUTPUT_DIR = os.path.join("experiments", "structural_error_visualization")
OUTPUT_SUMMARY_MD = os.path.join(OUTPUT_DIR, "structural_error_visualization_summary.md")
OUTPUT_PAIRS_CSV = os.path.join(OUTPUT_DIR, "structural_error_visualization_pairs.csv")

THRESHOLDS = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
    0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
]


# -----------------------
# Fixed split
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
# Case metadata
# -----------------------
CASES = {
    "1BRS_A_B": {"pdb_id": "1BRS", "partner1": "A", "partner2": "B"},
    "1FSS_A_B": {"pdb_id": "1FSS", "partner1": "A", "partner2": "B"},
    "1AHW_AB_C": {"pdb_id": "1AHW", "partner1": "AB", "partner2": "C"},
    "1DQJ_AB_C": {"pdb_id": "1DQJ", "partner1": "AB", "partner2": "C"},
    "1E6J_HL_P": {"pdb_id": "1E6J", "partner1": "HL", "partner2": "P"},
    "1JPS_HL_T": {"pdb_id": "1JPS", "partner1": "HL", "partner2": "T"},
    "1MLC_AB_E": {"pdb_id": "1MLC", "partner1": "AB", "partner2": "E"},
    "1WEJ_HL_F": {"pdb_id": "1WEJ", "partner1": "HL", "partner2": "F"},
    "2FD6_HL_U": {"pdb_id": "2FD6", "partner1": "HL", "partner2": "U"},
    "2VIS_AB_C": {"pdb_id": "2VIS", "partner1": "AB", "partner2": "C"},
    "3HMX_LH_AB": {"pdb_id": "3HMX", "partner1": "LH", "partner2": "AB"},
    "3MJ9_HL_A": {"pdb_id": "3MJ9", "partner1": "HL", "partner2": "A"},
}


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
    pairs_path = os.path.join(PROCESSED_DIR, f"{case_name}_corr_pairs.npy")

    for path in [features_path, labels_path, edge_path, pairs_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    x = np.load(features_path)
    y = np.load(labels_path)
    edge_index = np.load(edge_path)
    pairs = np.load(pairs_path)

    # Best strict model uses only:
    # [CA_distance, degree_A, degree_B]
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
    return [load_graph(case_name) for case_name in case_names]


def normalize_features(train_dataset, val_dataset, test_dataset):
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
def train_gat_with_early_stopping(train_loader, val_loader, device, input_dim):
    set_seed(SEED)

    model = MultiGraphGAT(
        in_channels=input_dim,
        hidden_channels=16,
        heads=4,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR_GAT,
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
# PDB residue mapping
# -----------------------
def load_structure_model(pdb_id):
    pdb_path = os.path.join(RAW_PDB_DIR, f"{pdb_id}.pdb")

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Missing PDB file: {pdb_path}")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)

    return next(structure.get_models()), pdb_path


def extract_residue_metadata(model, chain_ids):
    """
    Extract residues in exactly the same order as preprocessing:
    chain order follows the chain_ids string, and only standard residues with CA are kept.
    """
    metadata = []

    for chain_id in chain_ids:
        if chain_id not in model:
            available = list(model.child_dict.keys())
            raise ValueError(
                f"Chain {chain_id} not found. Available chains: {available}"
            )

        chain = model[chain_id]

        for res in chain:
            if res.get_id()[0] != " ":
                continue

            if not res.has_id("CA"):
                continue

            hetflag, resseq, icode = res.get_id()
            resname = res.get_resname()

            metadata.append({
                "chain_id": chain_id,
                "resseq": int(resseq),
                "icode": icode.strip(),
                "resname": resname,
            })

    return metadata


def get_case_residue_metadata(case_name):
    case_info = CASES[case_name]

    model, pdb_path = load_structure_model(case_info["pdb_id"])

    partner1_meta = extract_residue_metadata(model, case_info["partner1"])
    partner2_meta = extract_residue_metadata(model, case_info["partner2"])

    return case_info, pdb_path, partner1_meta, partner2_meta


def pymol_resi(meta):
    if meta["icode"]:
        return f"{meta['resseq']}{meta['icode']}"
    return str(meta["resseq"])


def pymol_residue_selection(meta):
    return f"(chain {meta['chain_id']} and resi {pymol_resi(meta)})"


def pymol_ca_selection(meta):
    return f"(chain {meta['chain_id']} and resi {pymol_resi(meta)} and name CA)"


def residue_label(meta):
    return f"{meta['chain_id']}:{meta['resname']}{pymol_resi(meta)}"


# -----------------------
# Prediction extraction
# -----------------------
def classify_node(true_label, pred_label):
    if true_label == 1 and pred_label == 1:
        return "TP"

    if true_label == 0 and pred_label == 1:
        return "FP"

    if true_label == 1 and pred_label == 0:
        return "FN"

    return "TN"


def collect_case_predictions(model, test_dataset, device, threshold):
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
            pairs = graph.pairs.cpu().numpy()

            all_true.append(true)
            all_pred.append(pred)

            case_rows = []

            for node_id in range(len(true)):
                status = classify_node(
                    true_label=int(true[node_id]),
                    pred_label=int(pred[node_id]),
                )

                if status == "TN":
                    continue

                a_idx = int(pairs[node_id][0])
                b_idx = int(pairs[node_id][1])

                row = {
                    "case": case_name,
                    "node_id": int(node_id),
                    "residue_a_idx": a_idx,
                    "residue_b_idx": b_idx,
                    "true_label": int(true[node_id]),
                    "pred_label": int(pred[node_id]),
                    "prob_class_1": float(prob_1[node_id]),
                    "status": status,
                }

                case_rows.append(row)
                all_rows.append(row)

            tp = sum(row["status"] == "TP" for row in case_rows)
            fp = sum(row["status"] == "FP" for row in case_rows)
            fn = sum(row["status"] == "FN" for row in case_rows)

            per_case_summary[case_name] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "selected_rows": len(case_rows),
            }

    true_all = np.concatenate(all_true)
    pred_all = np.concatenate(all_pred)

    return all_rows, per_case_summary, true_all, pred_all


def select_top_rows_for_case(rows, case_name, status, top_k):
    selected = [
        row for row in rows
        if row["case"] == case_name and row["status"] == status
    ]

    if status in {"TP", "FP"}:
        selected = sorted(
            selected,
            key=lambda row: row["prob_class_1"],
            reverse=True,
        )
    elif status == "FN":
        selected = sorted(
            selected,
            key=lambda row: row["prob_class_1"],
        )

    return selected[:top_k]


# -----------------------
# PyMOL script generation
# -----------------------
def selection_or_none(selection_items):
    unique_items = sorted(set(selection_items))

    if not unique_items:
        return "none"

    return " or ".join(unique_items)


def add_distance_commands(f, object_prefix, rows, color_name, partner1_meta, partner2_meta):
    for rank, row in enumerate(rows, start=1):
        a_meta = partner1_meta[row["residue_a_idx"]]
        b_meta = partner2_meta[row["residue_b_idx"]]

        name = f"{object_prefix}_{rank}"

        f.write(
            f"distance {name}, "
            f"{pymol_ca_selection(a_meta)}, "
            f"{pymol_ca_selection(b_meta)}\n"
        )
        f.write(f"color {color_name}, {name}\n")
        f.write(f"hide labels, {name}\n")


def write_pymol_script(case_name, rows):
    case_info, pdb_path, partner1_meta, partner2_meta = get_case_residue_metadata(case_name)

    pdb_path_for_pymol = pdb_path.replace("\\", "/")

    tp_rows = select_top_rows_for_case(rows, case_name, "TP", TOP_K_PER_TYPE)
    fp_rows = select_top_rows_for_case(rows, case_name, "FP", TOP_K_PER_TYPE)
    fn_rows = select_top_rows_for_case(rows, case_name, "FN", TOP_K_PER_TYPE)

    tp_selections = []
    fp_selections = []
    fn_selections = []

    output_rows = []

    for status, selected_rows, selection_bucket in [
        ("TP", tp_rows, tp_selections),
        ("FP", fp_rows, fp_selections),
        ("FN", fn_rows, fn_selections),
    ]:
        for rank, row in enumerate(selected_rows, start=1):
            a_meta = partner1_meta[row["residue_a_idx"]]
            b_meta = partner2_meta[row["residue_b_idx"]]

            selection_bucket.append(pymol_residue_selection(a_meta))
            selection_bucket.append(pymol_residue_selection(b_meta))

            output_rows.append({
                "case": case_name,
                "status": status,
                "rank": rank,
                "node_id": row["node_id"],
                "prob_class_1": row["prob_class_1"],
                "true_label": row["true_label"],
                "pred_label": row["pred_label"],
                "partner1_chain": a_meta["chain_id"],
                "partner1_resseq": pymol_resi(a_meta),
                "partner1_resname": a_meta["resname"],
                "partner1_label": residue_label(a_meta),
                "partner2_chain": b_meta["chain_id"],
                "partner2_resseq": pymol_resi(b_meta),
                "partner2_resname": b_meta["resname"],
                "partner2_label": residue_label(b_meta),
            })

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{case_name}_structural_errors.pml",
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# PyMOL structural error visualization for {case_name}\n")
        f.write("# Green = TP, Red = FP, Orange = FN\n\n")

        f.write("reinitialize\n")
        f.write(f"load {pdb_path_for_pymol}, {case_name}\n\n")

        f.write("hide everything\n")
        f.write("show cartoon\n")
        f.write("color gray80\n")
        f.write("set cartoon_transparency, 0.25\n")
        f.write("set stick_radius, 0.22\n")
        f.write("set dash_width, 2.5\n")
        f.write("set dash_gap, 0.35\n\n")

        f.write(f"select TP_residues, {selection_or_none(tp_selections)}\n")
        f.write(f"select FP_residues, {selection_or_none(fp_selections)}\n")
        f.write(f"select FN_residues, {selection_or_none(fn_selections)}\n\n")

        f.write("color green, TP_residues\n")
        f.write("color red, FP_residues\n")
        f.write("color orange, FN_residues\n\n")

        f.write("show sticks, TP_residues\n")
        f.write("show sticks, FP_residues\n")
        f.write("show sticks, FN_residues\n\n")

        add_distance_commands(
            f,
            object_prefix="TP_pair",
            rows=tp_rows,
            color_name="green",
            partner1_meta=partner1_meta,
            partner2_meta=partner2_meta,
        )

        add_distance_commands(
            f,
            object_prefix="FP_pair",
            rows=fp_rows,
            color_name="red",
            partner1_meta=partner1_meta,
            partner2_meta=partner2_meta,
        )

        add_distance_commands(
            f,
            object_prefix="FN_pair",
            rows=fn_rows,
            color_name="orange",
            partner1_meta=partner1_meta,
            partner2_meta=partner2_meta,
        )

        f.write("\nzoom TP_residues or FP_residues or FN_residues\n")
        f.write("orient\n")
        f.write("bg_color white\n")

    return output_path, output_rows, {
        "tp_selected": len(tp_rows),
        "fp_selected": len(fp_rows),
        "fn_selected": len(fn_rows),
    }


# -----------------------
# Save outputs
# -----------------------
def save_selected_pairs_csv(rows):
    fieldnames = [
        "case",
        "status",
        "rank",
        "node_id",
        "prob_class_1",
        "true_label",
        "pred_label",
        "partner1_chain",
        "partner1_resseq",
        "partner1_resname",
        "partner1_label",
        "partner2_chain",
        "partner2_resseq",
        "partner2_resname",
        "partner2_label",
    ]

    with open(OUTPUT_PAIRS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {OUTPUT_PAIRS_CSV}")


def save_summary_md(
    best_epoch,
    best_threshold,
    best_val_f1,
    metrics,
    cm,
    pml_outputs,
    per_case_summary,
    selected_counts,
):
    tn, fp, fn, tp = cm.ravel()

    with open(OUTPUT_SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("# Structural 3D Error Visualization Summary\n\n")

        f.write("This experiment generates PyMOL scripts to visualize important TP, FP, and FN residue pairs in 3D structure.\n\n")

        f.write("```text\n")
        f.write("Model: GAT\n")
        f.write("Feature set: basic 3 features\n")
        f.write("Input dimension: 3\n")
        f.write("Visualization target: top TP / FP / FN residue pairs per test complex\n")
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

        f.write("## Confusion Matrix\n\n")
        f.write("| True / Pred | Pred 0 | Pred 1 |\n")
        f.write("|-------------|--------|--------|\n")
        f.write(f"| True 0 | {tn} | {fp} |\n")
        f.write(f"| True 1 | {fn} | {tp} |\n\n")

        f.write("## Per-Test-Graph Available Error Counts\n\n")
        f.write("| Case | TP | FP | FN |\n")
        f.write("|------|----|----|----|\n")

        for case_name, item in per_case_summary.items():
            f.write(
                f"| {case_name} "
                f"| {item['tp']} "
                f"| {item['fp']} "
                f"| {item['fn']} |\n"
            )

        f.write("\n## Selected Residue Pairs for Visualization\n\n")
        f.write(f"Top `{TOP_K_PER_TYPE}` pairs per class were selected when available.\n\n")
        f.write("| Case | TP Selected | FP Selected | FN Selected |\n")
        f.write("|------|-------------|-------------|-------------|\n")

        for case_name, item in selected_counts.items():
            f.write(
                f"| {case_name} "
                f"| {item['tp_selected']} "
                f"| {item['fp_selected']} "
                f"| {item['fn_selected']} |\n"
            )

        f.write("\n## Output PyMOL Scripts\n\n")
        f.write("| Case | PyMOL Script |\n")
        f.write("|------|--------------|\n")

        for case_name, path in pml_outputs.items():
            f.write(f"| {case_name} | `{path}` |\n")

        f.write("\n## Color Legend\n\n")
        f.write("| Color | Meaning |\n")
        f.write("|-------|---------|\n")
        f.write("| Green | True Positive residue pairs |\n")
        f.write("| Red | False Positive residue pairs |\n")
        f.write("| Orange | False Negative residue pairs |\n\n")

        f.write("## How to Open in PyMOL\n\n")
        f.write("Example:\n\n")
        f.write("```bash\n")
        f.write("pymol experiments/structural_error_visualization/1BRS_A_B_structural_errors.pml\n")
        f.write("```\n\n")

        f.write("## Notes\n\n")
        f.write("- TP and FP pairs are ranked by highest predicted probability for class 1.\n")
        f.write("- FN pairs are ranked by lowest predicted probability for class 1, because these are the most strongly missed contacts.\n")
        f.write("- The visualization highlights residues and draws CA-to-CA distance objects for selected pairs.\n")
        f.write("- This is intended for qualitative structural interpretation, not as an additional quantitative metric.\n\n")

        f.write("## Output Tables\n\n")
        f.write(f"- Selected residue pairs: `{OUTPUT_PAIRS_CSV}`\n")
        f.write(f"- Summary: `{OUTPUT_SUMMARY_MD}`\n")

    print(f"Saved: {OUTPUT_SUMMARY_MD}")


# -----------------------
# Main
# -----------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    print("\nStructural 3D error visualization")
    print("Model: GAT")
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

    model, best_epoch, best_threshold, best_val_f1 = train_gat_with_early_stopping(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        input_dim=input_dim,
    )

    rows, per_case_summary, true_all, pred_all = collect_case_predictions(
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

    pml_outputs = {}
    selected_counts = {}
    all_selected_pair_rows = []

    for case_name in TEST_CASES:
        pml_path, selected_pair_rows, counts = write_pymol_script(
            case_name=case_name,
            rows=rows,
        )

        pml_outputs[case_name] = pml_path
        selected_counts[case_name] = counts
        all_selected_pair_rows.extend(selected_pair_rows)

        print(f"Saved: {pml_path}")

    save_selected_pairs_csv(all_selected_pair_rows)

    save_summary_md(
        best_epoch=best_epoch,
        best_threshold=best_threshold,
        best_val_f1=best_val_f1,
        metrics=metrics,
        cm=cm,
        pml_outputs=pml_outputs,
        per_case_summary=per_case_summary,
        selected_counts=selected_counts,
    )

    print("\nStructural 3D error visualization completed successfully.")


if __name__ == "__main__":
    main()