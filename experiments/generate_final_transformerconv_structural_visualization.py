"""
Generate structural 3D error visualization for the FINAL tuned TransformerConv model.

Final model setting:
- Dataset: Combined Current + BM5
- Features: Basic 3 + Full Pair ESM-2 PCA16
- Model: TransformerConv
- hidden_channels = 16
- heads = 4
- dropout = 0.2
- lr = 0.003
- weight_decay = 0.001
- threshold_max = 0.90
- seed = 1

This v2 version also supports corr_pairs.npy rows stored as integer residue indices, e.g. array([4, 6]).

Outputs:
- PyMOL .pml files for each final test complex
- Clean PyMOL .pml files for presentation
- CSV of selected TP/FP/FN residue-pair examples
- Markdown summary

Usage:
python experiments/generate_final_transformerconv_structural_visualization.py ^
  --processed_dir data/processed_combined_current_bm5_esm2_pca16 ^
  --out_dir experiments/structural_error_visualization_final_transformerconv ^
  --hidden_channels 16 --heads 4 --dropout 0.2 ^
  --lr 0.003 --weight_decay 0.001 ^
  --threshold_max 0.90 --seed 1 --top_k 10
"""

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
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa


# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Keep this script practical on Windows/GPU. Full deterministic mode can
    # slow training down and is not necessary for visualization generation.
    torch.backends.cudnn.benchmark = False


# ============================================================
# Data loading
# ============================================================
def load_split_cases(processed_dir: str):
    path = os.path.join(processed_dir, "split_cases.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing split_cases.json: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_graph(processed_dir: str, case_name: str):
    features = np.load(os.path.join(processed_dir, f"{case_name}_corr_features.npy"))
    labels = np.load(os.path.join(processed_dir, f"{case_name}_corr_labels.npy"))
    edge_index = np.load(os.path.join(processed_dir, f"{case_name}_corr_edge_index.npy"))

    return Data(
        x=torch.tensor(features, dtype=torch.float32),
        y=torch.tensor(labels, dtype=torch.long),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )


def load_graphs(processed_dir: str, case_names, device):
    graphs = {}

    for case_name in case_names:
        graphs[case_name] = load_graph(processed_dir, case_name).to(device)

    return graphs


def load_pairs(processed_dir: str, case_name: str):
    path = os.path.join(processed_dir, f"{case_name}_corr_pairs.npy")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing correspondence pairs file: {path}")

    return np.load(path, allow_pickle=True)


# ============================================================
# Final TransformerConv model
# ============================================================
class TransformerConvNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=16, heads=4, dropout=0.2, beta=True):
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


# ============================================================
# Metrics and threshold selection
# ============================================================
@torch.no_grad()
def collect_probs(model, graphs):
    model.eval()

    y_true_list = []
    prob_list = []

    for graph in graphs.values():
        out = model(graph.x, graph.edge_index)
        probs = F.softmax(out, dim=1)[:, 1]

        y_true_list.append(graph.y.detach().cpu().numpy())
        prob_list.append(probs.detach().cpu().numpy())

    return np.concatenate(y_true_list), np.concatenate(prob_list)


@torch.no_grad()
def predict_graph_probs(model, graph):
    model.eval()

    out = model(graph.x, graph.edge_index)
    probs = F.softmax(out, dim=1)[:, 1]

    return probs.detach().cpu().numpy()


def metrics_at_threshold(y_true, probs, threshold):
    pred = (probs >= threshold).astype(np.int64)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        pred,
        labels=[0, 1],
        zero_division=0,
    )

    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    return {
        "threshold": float(threshold),
        "p1": float(precision[1]),
        "r1": float(recall[1]),
        "f1": float(f1[1]),
        "acc": float(accuracy_score(y_true, pred)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def choose_best_threshold(y_true, probs, thresholds):
    best = None

    for threshold in thresholds:
        row = metrics_at_threshold(y_true, probs, threshold)

        if best is None:
            best = row
            continue

        # Main objective: validation positive-class F1.
        # Tie-breakers: higher precision, then higher accuracy.
        if (
            row["f1"] > best["f1"]
            or (row["f1"] == best["f1"] and row["p1"] > best["p1"])
            or (
                row["f1"] == best["f1"]
                and row["p1"] == best["p1"]
                and row["acc"] > best["acc"]
            )
        ):
            best = row

    return best


# ============================================================
# Training
# ============================================================
def train_final_model(args, input_dim, train_graphs, val_graphs, device):
    model = TransformerConvNet(
        in_channels=input_dim,
        hidden_channels=args.hidden_channels,
        heads=args.heads,
        dropout=args.dropout,
        beta=True,
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
            f"TransformerConv final | Epoch {epoch} | Loss={avg_loss:.4f} "
            f"| Val_F1_1={val_best['f1']:.4f} "
            f"| Best_Val_F1_1={best_val['f1']:.4f} "
            f"| Best_Threshold={best_val['threshold']:.2f}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val, best_epoch


# ============================================================
# Pair parsing
# ============================================================
def as_text(x):
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def clean_chain(x):
    return as_text(x).strip()


def clean_resi(x):
    value = as_text(x).strip()

    # Convert "108.0" -> "108" if needed.
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass

    return value


def extract_pair_record(record, residue_maps=None):
    """
    Parse one row from <case>_corr_pairs.npy into:
    (chain1, resi1, chain2, resi2)

    Supports common formats used in NumPy object arrays:
    - (chain1, resi1, chain2, resi2)
    - ((chain1, resi1), (chain2, resi2))
    - dict-like records
    - structured numpy rows
    """

    if isinstance(record, np.ndarray) and record.shape == ():
        record = record.item()

    if isinstance(record, np.void) and record.dtype.names:
        return extract_pair_record({name: record[name] for name in record.dtype.names}, residue_maps=residue_maps)

    if isinstance(record, dict):
        key_sets = [
            ("chain1", "resi1", "chain2", "resi2"),
            ("chain_a", "resi_a", "chain_b", "resi_b"),
            ("chain_A", "resi_A", "chain_B", "resi_B"),
            ("chain_i", "resi_i", "chain_j", "resi_j"),
            ("p1_chain", "p1_resi", "p2_chain", "p2_resi"),
            ("partner1_chain", "partner1_resi", "partner2_chain", "partner2_resi"),
            ("chain1", "residue1", "chain2", "residue2"),
            ("chain_a", "residue_a", "chain_b", "residue_b"),
        ]

        for keys in key_sets:
            if all(k in record for k in keys):
                return (
                    clean_chain(record[keys[0]]),
                    clean_resi(record[keys[1]]),
                    clean_chain(record[keys[2]]),
                    clean_resi(record[keys[3]]),
                )

        nested_key_sets = [
            ("residue1", "residue2"),
            ("res1", "res2"),
            ("pair1", "pair2"),
            ("a", "b"),
        ]

        for left_key, right_key in nested_key_sets:
            if left_key in record and right_key in record:
                left = record[left_key]
                right = record[right_key]
                return (
                    clean_chain(left[0]),
                    clean_resi(left[1]),
                    clean_chain(right[0]),
                    clean_resi(right[1]),
                )

        raise ValueError(f"Unsupported dict pair record keys: {list(record.keys())}")

    if isinstance(record, (tuple, list, np.ndarray)):
        values = list(record)

        if len(values) == 1:
            return extract_pair_record(values[0], residue_maps=residue_maps)

        # Final ESM-PCA16 processed datasets commonly store:
        #   [partner1_residue_index, partner2_residue_index]
        if (
            len(values) == 2
            and residue_maps is not None
            and is_integer_like(values[0])
            and is_integer_like(values[1])
        ):
            idx1 = integer_value(values[0])
            idx2 = integer_value(values[1])

            try:
                chain1, resi1 = residue_maps["partner1"][idx1]
                chain2, resi2 = residue_maps["partner2"][idx2]
            except IndexError as exc:
                raise IndexError(
                    f"Residue index out of range: pair=({idx1}, {idx2}), "
                    f"partner1_len={len(residue_maps['partner1'])}, "
                    f"partner2_len={len(residue_maps['partner2'])}"
                ) from exc

            return chain1, resi1, chain2, resi2

        # Format: (chain1, resi1, chain2, resi2)
        if len(values) >= 4 and not isinstance(values[0], (tuple, list, np.ndarray, dict)):
            return (
                clean_chain(values[0]),
                clean_resi(values[1]),
                clean_chain(values[2]),
                clean_resi(values[3]),
            )

        # Format: ((chain1, resi1), (chain2, resi2))
        if len(values) >= 2:
            left = values[0]
            right = values[1]

            if isinstance(left, np.ndarray):
                left = list(left)
            if isinstance(right, np.ndarray):
                right = list(right)

            if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
                return (
                    clean_chain(left[0]),
                    clean_resi(left[1]),
                    clean_chain(right[0]),
                    clean_resi(right[1]),
                )

    raise ValueError(f"Unsupported pair record format: type={type(record)} value={record!r}")


# ============================================================
# PyMOL path handling
# ============================================================
def parse_case_name(case_name):
    """
    Returns:
      pdb_id, partner1_chains, partner2_chains

    Examples:
      1BRS_A_B      -> 1BRS, A, B
      3HMX_LH_AB    -> 3HMX, LH, AB
      BM5_1A2K_A_B  -> 1A2K, A, B
    """

    parts = case_name.split("_")

    if case_name.startswith("BM5_"):
        if len(parts) < 4:
            raise ValueError(f"Cannot parse BM5 case name: {case_name}")
        return parts[1], parts[2], parts[3]

    if len(parts) < 3:
        raise ValueError(f"Cannot parse case name: {case_name}")

    return parts[0], parts[1], parts[2]


def case_to_pdb_path(case_name):
    """
    Path is written relative to the default output directory:
    experiments/structural_error_visualization_final_transformerconv/

    Current test cases:
      data/raw_pdb/<PDB>.pdb

    BM5 cases:
      data/raw_pdb_expanded_bm5/<CASE>.pdb
    """

    if case_name.startswith("BM5_"):
        return f"../../data/raw_pdb_expanded_bm5/{case_name}.pdb"

    pdb_id = case_name.split("_")[0]
    return f"../../data/raw_pdb/{pdb_id}.pdb"


def case_to_pdb_read_path(case_name):
    """
    Filesystem path used by Python while running from the project root.
    This is different from the path written inside the .pml file.
    """

    if case_name.startswith("BM5_"):
        return os.path.join("data", "raw_pdb_expanded_bm5", f"{case_name}.pdb")

    pdb_id = case_name.split("_")[0]
    return os.path.join("data", "raw_pdb", f"{pdb_id}.pdb")


def residue_to_pymol_resi(residue):
    """
    Converts a Biopython residue id to a PyMOL-compatible residue identifier.
    Handles insertion codes if present.
    """

    _, resseq, icode = residue.get_id()
    icode = icode.strip()

    if icode:
        return f"{resseq}{icode}"

    return str(resseq)


def collect_partner_residues_from_pdb(case_name):
    """
    Builds index -> (chain, resi) maps for partner 1 and partner 2.

    This fixes the common corr_pairs.npy format where each row is simply:
      [partner1_residue_index, partner2_residue_index]

    The old script expected corr_pairs.npy to already contain chain/residue IDs,
    but the final processed ESM-PCA16 dataset stores integer residue indices.
    """

    pdb_id, partner1_chains, partner2_chains = parse_case_name(case_name)
    pdb_path = case_to_pdb_read_path(case_name)

    if not os.path.exists(pdb_path):
        raise FileNotFoundError(
            f"Could not find PDB file for {case_name}: {pdb_path}\n"
            "If this is a BM5 case, check data/raw_pdb_expanded_bm5/. "
            "If this is a current case, check data/raw_pdb/."
        )

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    model = next(structure.get_models())

    def collect(chains_text):
        residues = []

        for chain_id in list(chains_text):
            if chain_id not in model:
                raise KeyError(
                    f"Chain {chain_id} not found in {pdb_path} for case {case_name}. "
                    f"Available chains: {[c.id for c in model]}"
                )

            chain = model[chain_id]

            for residue in chain:
                if residue.id[0] != " ":
                    continue
                if not is_aa(residue, standard=True):
                    continue
                if "CA" not in residue:
                    continue

                residues.append((chain_id, residue_to_pymol_resi(residue)))

        return residues

    partner1 = collect(partner1_chains)
    partner2 = collect(partner2_chains)

    return {
        "partner1": partner1,
        "partner2": partner2,
    }


def is_integer_like(x):
    try:
        f = float(as_text(x).strip())
        return f.is_integer()
    except Exception:
        return False


def integer_value(x):
    return int(float(as_text(x).strip()))


def pymol_residue_selection(residues):
    if not residues:
        return "none"

    parts = []

    for chain, resi in sorted(residues, key=lambda x: (x[0], x[1])):
        parts.append(f"(chain {chain} and resi {resi})")

    return " or ".join(parts)


def add_distance_lines(lines, prefix, pairs, color_name):
    for i, row in enumerate(pairs, start=1):
        chain1, resi1, chain2, resi2 = row["chain1"], row["resi1"], row["chain2"], row["resi2"]
        obj = f"{prefix}_pair_{i}"

        lines.append(
            f"distance {obj}, "
            f"(chain {chain1} and resi {resi1} and name CA), "
            f"(chain {chain2} and resi {resi2} and name CA)"
        )
        lines.append(f"color {color_name}, {obj}")
        lines.append(f"hide labels, {obj}")


def write_pml(case_name, selected, out_dir):
    pml_path = os.path.join(out_dir, f"{case_name}_final_transformerconv_structural_errors.pml")

    tp_residues = set()
    fp_residues = set()
    fn_residues = set()

    for group, rows in selected.items():
        for row in rows:
            residue_a = (row["chain1"], row["resi1"])
            residue_b = (row["chain2"], row["resi2"])

            if group == "TP":
                tp_residues.add(residue_a)
                tp_residues.add(residue_b)
            elif group == "FP":
                fp_residues.add(residue_a)
                fp_residues.add(residue_b)
            elif group == "FN":
                fn_residues.add(residue_a)
                fn_residues.add(residue_b)

    lines = [
        "reinitialize",
        f"load {case_to_pdb_path(case_name)}, {case_name}",
        "hide everything",
        f"show cartoon, {case_name}",
        f"color gray80, {case_name}",
        "set cartoon_transparency, 0.25",
        "set stick_radius, 0.22",
        "set dash_width, 2.5",
        "set dash_gap, 0.35",
        f"select TP_residues, {pymol_residue_selection(tp_residues)}",
        f"select FP_residues, {pymol_residue_selection(fp_residues)}",
        f"select FN_residues, {pymol_residue_selection(fn_residues)}",
        "color green, TP_residues",
        "color red, FP_residues",
        "color orange, FN_residues",
        "show sticks, TP_residues",
        "show sticks, FP_residues",
        "show sticks, FN_residues",
    ]

    add_distance_lines(lines, "TP", selected["TP"], "green")
    add_distance_lines(lines, "FP", selected["FP"], "red")
    add_distance_lines(lines, "FN", selected["FN"], "orange")

    lines += [
        "zoom TP_residues or FP_residues or FN_residues",
        "orient",
        "bg_color white",
        "# Optional high-resolution export:",
        "# ray 1800, 1400",
        f"# png {case_name}_final_transformerconv_structural_errors.png, dpi=300",
    ]

    with open(pml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return pml_path


def write_clean_pml(case_name, selected, out_dir):
    clean_path = os.path.join(out_dir, f"{case_name}_final_transformerconv_structural_errors_clean.pml")

    tp_residues = set()
    fp_residues = set()
    fn_residues = set()

    for group, rows in selected.items():
        for row in rows:
            residue_a = (row["chain1"], row["resi1"])
            residue_b = (row["chain2"], row["resi2"])

            if group == "TP":
                tp_residues.add(residue_a)
                tp_residues.add(residue_b)
            elif group == "FP":
                fp_residues.add(residue_a)
                fp_residues.add(residue_b)
            elif group == "FN":
                fn_residues.add(residue_a)
                fn_residues.add(residue_b)

    lines = [
        "reinitialize",
        f"load {case_to_pdb_path(case_name)}, {case_name}",
        "hide everything",
        f"show cartoon, {case_name}",
        f"color gray80, {case_name}",
        "set cartoon_transparency, 0.30",
        "set stick_radius, 0.28",
        f"select TP_residues, {pymol_residue_selection(tp_residues)}",
        f"select FP_residues, {pymol_residue_selection(fp_residues)}",
        f"select FN_residues, {pymol_residue_selection(fn_residues)}",
        "show sticks, TP_residues",
        "show sticks, FP_residues",
        "show sticks, FN_residues",
        "color green, TP_residues",
        "color red, FP_residues",
        "color orange, FN_residues",
        "zoom TP_residues or FP_residues or FN_residues",
        "orient",
        "bg_color white",
        "# Optional high-resolution export:",
        "# ray 1800, 1400",
        f"# png {case_name}_final_transformerconv_structural_errors_clean.png, dpi=300",
    ]

    with open(clean_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return clean_path


# ============================================================
# Selecting TP/FP/FN examples
# ============================================================
def build_prediction_rows(case_name, pairs_array, labels, probs, threshold, residue_maps):
    preds = (probs >= threshold).astype(np.int64)
    rows = []

    for idx in range(len(labels)):
        try:
            chain1, resi1, chain2, resi2 = extract_pair_record(pairs_array[idx], residue_maps=residue_maps)
        except Exception as exc:
            raise RuntimeError(
                f"Could not parse pair record for case={case_name}, index={idx}, "
                f"record={pairs_array[idx]!r}"
            ) from exc

        label = int(labels[idx])
        pred = int(preds[idx])
        prob = float(probs[idx])

        if label == 1 and pred == 1:
            error_type = "TP"
        elif label == 0 and pred == 1:
            error_type = "FP"
        elif label == 1 and pred == 0:
            error_type = "FN"
        else:
            error_type = "TN"

        rows.append(
            {
                "case": case_name,
                "index": idx,
                "chain1": chain1,
                "resi1": resi1,
                "chain2": chain2,
                "resi2": resi2,
                "prob": prob,
                "label": label,
                "pred": pred,
                "error_type": error_type,
            }
        )

    return rows


def select_top_examples(rows, top_k):
    tp = [r for r in rows if r["error_type"] == "TP"]
    fp = [r for r in rows if r["error_type"] == "FP"]
    fn = [r for r in rows if r["error_type"] == "FN"]

    # TP and FP: most confident positive examples.
    tp = sorted(tp, key=lambda r: r["prob"], reverse=True)[:top_k]
    fp = sorted(fp, key=lambda r: r["prob"], reverse=True)[:top_k]

    # FN: strongest missed positives, lowest positive probability first.
    fn = sorted(fn, key=lambda r: r["prob"])[:top_k]

    return {"TP": tp, "FP": fp, "FN": fn}


# ============================================================
# Output files
# ============================================================
def write_examples_csv(path, all_selected_rows):
    fieldnames = [
        "case",
        "error_type",
        "index",
        "chain1",
        "resi1",
        "chain2",
        "resi2",
        "prob",
        "label",
        "pred",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in all_selected_rows:
            writer.writerow(row)


def write_summary_md(path, args, global_metrics, best_epoch, case_summaries):
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Final Tuned TransformerConv Structural Error Visualization\n\n")

        f.write("## Model Setup\n\n")
        f.write("```text\n")
        f.write("Model: Tuned TransformerConv\n")
        f.write(f"Processed dir: {args.processed_dir}\n")
        f.write(f"hidden_channels: {args.hidden_channels}\n")
        f.write(f"heads: {args.heads}\n")
        f.write(f"dropout: {args.dropout}\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"weight_decay: {args.weight_decay}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"best_epoch: {best_epoch}\n")
        f.write(f"threshold_max: {args.threshold_max}\n")
        f.write("```\n\n")

        f.write("## Global Test Metrics\n\n")
        f.write("| Threshold | P1 | R1 | F1 | Acc | TN | FP | FN | TP |\n")
        f.write("|----------:|---:|---:|---:|----:|---:|---:|---:|---:|\n")
        f.write(
            f"| {global_metrics['threshold']:.2f} "
            f"| {global_metrics['p1']:.4f} "
            f"| {global_metrics['r1']:.4f} "
            f"| {global_metrics['f1']:.4f} "
            f"| {global_metrics['acc']:.4f} "
            f"| {global_metrics['tn']} "
            f"| {global_metrics['fp']} "
            f"| {global_metrics['fn']} "
            f"| {global_metrics['tp']} |\n\n"
        )

        f.write("## Per-Case Test Summary\n\n")
        f.write("| Case | Nodes | Positives | Negatives | P1 | R1 | F1 | Acc | TN | FP | FN | TP | PML |\n")
        f.write("|------|------:|----------:|----------:|---:|---:|---:|----:|---:|---:|---:|---:|-----|\n")

        for row in case_summaries:
            pml_name = os.path.basename(row["pml_path"])
            f.write(
                f"| {row['case']} "
                f"| {row['nodes']} "
                f"| {row['positive']} "
                f"| {row['negative']} "
                f"| {row['p1']:.4f} "
                f"| {row['r1']:.4f} "
                f"| {row['f1']:.4f} "
                f"| {row['acc']:.4f} "
                f"| {row['tn']} "
                f"| {row['fp']} "
                f"| {row['fn']} "
                f"| {row['tp']} "
                f"| `{pml_name}` |\n"
            )

        f.write("\n## Color Legend\n\n")
        f.write("| Color | Meaning |\n")
        f.write("|-------|---------|\n")
        f.write("| Green | True Positive residue-pair examples |\n")
        f.write("| Red | False Positive residue-pair examples |\n")
        f.write("| Orange | False Negative residue-pair examples |\n")
        f.write("| Gray | Full protein complex cartoon |\n\n")

        f.write("## Interpretation\n\n")
        f.write(
            "These PyMOL files visualize selected TP, FP, and FN residue-pair examples from the final tuned TransformerConv model. "
            "They are intended for qualitative structural analysis and presentation. "
            "The selected examples are not necessarily all model predictions; by default, the script exports the top-k examples for each class of prediction outcome.\n"
        )


# ============================================================
# Main
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate final tuned TransformerConv structural error visualization PML files."
    )

    parser.add_argument(
        "--processed_dir",
        default=os.path.join("data", "processed_combined_current_bm5_esm2_pca16"),
    )

    parser.add_argument(
        "--out_dir",
        default=os.path.join("experiments", "structural_error_visualization_final_transformerconv"),
    )

    parser.add_argument("--hidden_channels", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--max_epochs", type=int, default=220)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=40)

    parser.add_argument("--threshold_min", type=float, default=0.05)
    parser.add_argument("--threshold_max", type=float, default=0.90)
    parser.add_argument("--threshold_step", type=float, default=0.01)

    parser.add_argument("--top_k", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split_cases = load_split_cases(args.processed_dir)
    train_cases = split_cases["train"]
    val_cases = split_cases["val"]
    test_cases = split_cases["test"]

    print("\nFinal tuned TransformerConv structural visualization")
    print(f"Processed dir: {args.processed_dir}")
    print(f"Output dir: {args.out_dir}")
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

    model, best_val, best_epoch = train_final_model(args, input_dim, train_graphs, val_graphs, device)

    test_true, test_probs = collect_probs(model, test_graphs)
    global_metrics = metrics_at_threshold(test_true, test_probs, best_val["threshold"])

    print("\nGlobal final test metrics:")
    print(
        f"Threshold={global_metrics['threshold']:.2f} | "
        f"P1={global_metrics['p1']:.4f} | "
        f"R1={global_metrics['r1']:.4f} | "
        f"F1={global_metrics['f1']:.4f} | "
        f"Acc={global_metrics['acc']:.4f}"
    )
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(np.array([[global_metrics["tn"], global_metrics["fp"]], [global_metrics["fn"], global_metrics["tp"]]]))

    all_selected_rows = []
    case_summaries = []

    for case_name, graph in test_graphs.items():
        labels = graph.y.detach().cpu().numpy()
        probs = predict_graph_probs(model, graph)
        pairs = load_pairs(args.processed_dir, case_name)

        residue_maps = collect_partner_residues_from_pdb(case_name)

        rows = build_prediction_rows(
            case_name=case_name,
            pairs_array=pairs,
            labels=labels,
            probs=probs,
            threshold=global_metrics["threshold"],
            residue_maps=residue_maps,
        )

        selected = select_top_examples(rows, args.top_k)

        for group, group_rows in selected.items():
            for row in group_rows:
                out_row = dict(row)
                out_row["error_type"] = group
                all_selected_rows.append(out_row)

        case_metrics = metrics_at_threshold(labels, probs, global_metrics["threshold"])
        positive = int(labels.sum())
        negative = int(len(labels) - positive)

        pml_path = write_pml(case_name, selected, args.out_dir)
        clean_pml_path = write_clean_pml(case_name, selected, args.out_dir)

        case_summaries.append(
            {
                "case": case_name,
                "nodes": int(len(labels)),
                "positive": positive,
                "negative": negative,
                **case_metrics,
                "pml_path": pml_path,
                "clean_pml_path": clean_pml_path,
            }
        )

        print(
            f"{case_name}: nodes={len(labels)} | pos={positive} | neg={negative} "
            f"| P1={case_metrics['p1']:.4f} | R1={case_metrics['r1']:.4f} "
            f"| F1={case_metrics['f1']:.4f} | PML={pml_path}"
        )

    examples_csv = os.path.join(args.out_dir, "final_transformerconv_structural_error_examples.csv")
    write_examples_csv(examples_csv, all_selected_rows)

    summary_md = os.path.join(args.out_dir, "final_transformerconv_structural_error_visualization_summary.md")
    write_summary_md(summary_md, args, global_metrics, best_epoch, case_summaries)

    print(f"\nSaved examples CSV: {examples_csv}")
    print(f"Saved summary: {summary_md}")
    print("Final tuned TransformerConv structural visualization completed successfully.")


if __name__ == "__main__":
    main()
