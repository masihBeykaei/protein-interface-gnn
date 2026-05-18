import os
import csv
import json
import argparse
import numpy as np
from Bio.PDB import PDBParser


# ----------------------------
# Config defaults
# ----------------------------
CONTACT_THRESHOLD = 5.0
INTRA_CA_EDGE_THRESHOLD = 8.0
CANDIDATE_RADIUS = 12.0
MAX_CORR_EDGES = 3_000_000
PROGRESS_EVERY = 2000

RAW_PDB_DIRS = [
    os.path.join("data", "raw_pdb_expanded"),
    os.path.join("data", "raw_pdb"),
]


# ----------------------------
# PDB utilities
# ----------------------------
def load_structure(pdb_path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure("protein", pdb_path)


def find_pdb_path(row):
    explicit = row.get("pdb_file", "").strip()

    if explicit:
        if os.path.exists(explicit):
            return explicit
        raise FileNotFoundError(f"Explicit pdb_file not found: {explicit}")

    pdb_id = row.get("pdb_id", "").strip()
    case_name = row.get("case_name", "").strip()

    candidates = []

    for raw_dir in RAW_PDB_DIRS:
        if pdb_id:
            candidates.extend([
                os.path.join(raw_dir, f"{pdb_id}.pdb"),
                os.path.join(raw_dir, f"{pdb_id.lower()}.pdb"),
                os.path.join(raw_dir, f"{pdb_id.upper()}.pdb"),
            ])

        if case_name:
            candidates.extend([
                os.path.join(raw_dir, f"{case_name}.pdb"),
                os.path.join(raw_dir, f"{case_name.lower()}.pdb"),
                os.path.join(raw_dir, f"{case_name.upper()}.pdb"),
            ])

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"No PDB file found for case={case_name}, pdb_id={pdb_id}. "
        f"Searched raw dirs: {RAW_PDB_DIRS}"
    )


def extract_partner_residues(model, chain_ids):
    residues_atoms = []
    ca_coords = []

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

            atoms = [atom.get_coord() for atom in res]

            if not atoms:
                continue

            residues_atoms.append(np.array(atoms, dtype=np.float32))
            ca_coords.append(res["CA"].get_coord())

    return residues_atoms, np.array(ca_coords, dtype=np.float32)


# ----------------------------
# Graph utilities
# ----------------------------
def build_edge_index_from_ca(ca_coords, threshold):
    n = len(ca_coords)
    edges = []

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(ca_coords[i] - ca_coords[j]) < threshold:
                edges.append((i, j))
                edges.append((j, i))

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)

    return np.array(edges, dtype=np.int64).T


def build_adjacency_list(edge_index, num_nodes):
    adj = [set() for _ in range(num_nodes)]

    if edge_index.size == 0:
        return adj

    for src, dst in edge_index.T:
        adj[int(src)].add(int(dst))

    return adj


def candidate_filter(ca_A, ca_B, radius):
    keep_A = np.array([
        np.any(np.linalg.norm(ca_B - ca_A[i], axis=1) < radius)
        for i in range(len(ca_A))
    ])

    keep_B = np.array([
        np.any(np.linalg.norm(ca_A - ca_B[j], axis=1) < radius)
        for j in range(len(ca_B))
    ])

    return np.where(keep_A)[0], np.where(keep_B)[0]


def residues_in_contact(resA_atoms, resB_atoms, threshold):
    for a in resA_atoms:
        for b in resB_atoms:
            if np.linalg.norm(a - b) < threshold:
                return 1

    return 0


# ----------------------------
# Correspondence node generation
# ----------------------------
def build_all_candidate_pairs_labels_and_distances(
    resA_atoms,
    resB_atoms,
    caA,
    caB,
    idx_A,
    idx_B,
    contact_threshold,
):
    pairs = []
    labels = []
    ca_distances = []

    total = len(idx_A) * len(idx_B)
    done = 0

    for a_idx in idx_A:
        resA_atoms_i = resA_atoms[a_idx]

        for b_idx in idx_B:
            ca_dist = float(np.linalg.norm(caA[a_idx] - caB[b_idx]))

            label = residues_in_contact(
                resA_atoms_i,
                resB_atoms[b_idx],
                contact_threshold,
            )

            pairs.append((int(a_idx), int(b_idx)))
            labels.append(int(label))
            ca_distances.append(ca_dist)

            done += 1

            if done % PROGRESS_EVERY == 0:
                print(f"Processed {done}/{total} residue pairs...")

    return (
        np.array(pairs, dtype=np.int64),
        np.array(labels, dtype=np.int64),
        np.array(ca_distances, dtype=np.float32),
    )


def select_hard_negative_train_pairs(
    pairs,
    labels,
    ca_distances,
    hard_negative_ratio,
):
    """
    Keep all positive pairs and choose the nearest non-contact residue pairs
    as hard negatives.

    Hard negatives are sorted by smallest CA distance because these pairs are
    geometrically close to the interface region but are not true atom-level contacts.
    """
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    if len(pos_idx) == 0:
        return pairs, labels, ca_distances

    max_neg = min(len(neg_idx), len(pos_idx) * hard_negative_ratio)

    # Sort negative examples by CA distance ascending.
    sorted_neg_idx = neg_idx[np.argsort(ca_distances[neg_idx])]
    selected_neg_idx = sorted_neg_idx[:max_neg]

    selected_idx = np.concatenate([pos_idx, selected_neg_idx])

    # Stable ordering by CA distance makes the dataset deterministic.
    # Shuffle is intentionally not used here.
    return (
        pairs[selected_idx],
        labels[selected_idx],
        ca_distances[selected_idx],
    )


def build_features(pairs, caA, caB, degree_A, degree_B):
    features = []

    for a_idx, b_idx in pairs:
        ca_dist = np.linalg.norm(caA[a_idx] - caB[b_idx])
        degA = degree_A[a_idx]
        degB = degree_B[b_idx]

        features.append([ca_dist, degA, degB])

    return np.array(features, dtype=np.float32)


def build_correspondence_edges_arbitrary_pairs(
    adjA,
    adjB,
    pairs,
    max_edges,
):
    pair_to_node = {
        (int(a), int(b)): idx
        for idx, (a, b) in enumerate(pairs)
    }

    edges = []

    for node_id, (a_g, b_g) in enumerate(pairs):
        a_g = int(a_g)
        b_g = int(b_g)

        for a2_g in adjA[a_g]:
            for b2_g in adjB[b_g]:
                target = pair_to_node.get((int(a2_g), int(b2_g)))

                if target is None:
                    continue

                edges.append((node_id, target))

                if len(edges) >= max_edges:
                    print(
                        f"Reached MAX_CORR_EDGES={max_edges}. "
                        "Stopping edge generation."
                    )
                    return np.array(edges, dtype=np.int64).T

    if not edges:
        return np.zeros((2, 0), dtype=np.int64)

    return np.array(edges, dtype=np.int64).T


# ----------------------------
# Cases / splits
# ----------------------------
def read_cases(cases_csv):
    rows = []

    if not os.path.exists(cases_csv):
        raise FileNotFoundError(f"Cases CSV not found: {cases_csv}")

    with open(cases_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            enabled = row.get("enabled", "1").strip()

            if enabled in {"0", "false", "False", "no", "No"}:
                continue

            rows.append(row)

    return rows


def normalize_split_name(split):
    split = split.strip().lower()

    if split in {"train", "training"}:
        return "train"

    if split in {"val", "valid", "validation"}:
        return "val"

    if split == "test":
        return "test"

    return "train"


def build_split_map(rows):
    split_map = {
        "train": [],
        "val": [],
        "test": [],
    }

    for row in rows:
        case_name = row["case_name"].strip()
        split = normalize_split_name(row.get("split", "train"))
        split_map[split].append(case_name)

    return split_map


# ----------------------------
# Save summaries
# ----------------------------
def save_summary_csv(path, summary):
    fieldnames = [
        "case",
        "split",
        "original_nodes",
        "original_positive",
        "original_negative",
        "original_positive_ratio",
        "saved_nodes",
        "saved_positive",
        "saved_negative",
        "saved_positive_ratio",
        "edges",
        "feature_dim",
        "hard_negative_applied",
        "hard_negative_ratio",
        "max_selected_negative_ca_distance",
        "mean_selected_negative_ca_distance",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print(f"Saved: {path}")


def split_totals(summary, split_name):
    items = [item for item in summary if item["split"] == split_name]

    nodes = sum(item["saved_nodes"] for item in items)
    pos = sum(item["saved_positive"] for item in items)
    neg = sum(item["saved_negative"] for item in items)
    ratio = pos / nodes if nodes > 0 else 0.0

    return nodes, pos, neg, ratio


def save_summary_md(path, summary, split_map, args):
    train_nodes, train_pos, train_neg, train_ratio = split_totals(summary, "train")
    val_nodes, val_pos, val_neg, val_ratio = split_totals(summary, "val")
    test_nodes, test_pos, test_neg, test_ratio = split_totals(summary, "test")

    total_nodes = train_nodes + val_nodes + test_nodes
    total_pos = train_pos + val_pos + test_pos
    total_neg = train_neg + val_neg + test_neg
    total_ratio = total_pos / total_nodes if total_nodes > 0 else 0.0

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Hard-Negative Train / Natural-Evaluation Dataset Summary\n\n")

        f.write("This dataset keeps validation and test splits natural, while training graphs are built from all positive pairs plus hard negative pairs.\n\n")

        f.write("Hard negatives are non-contact residue pairs with the smallest C-alpha distances.\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Cases CSV: `{args.cases}`\n")
        f.write(f"- Output directory: `{args.out_dir}`\n")
        f.write(f"- Hard negative ratio: `{args.hard_negative_ratio}`\n")
        f.write(f"- Candidate radius: `{args.candidate_radius}` Å\n")
        f.write(f"- Contact threshold: `{args.contact_threshold}` Å\n")
        f.write("- Feature mode: `basic 3 features`\n\n")

        f.write("## Split Cases\n\n")
        f.write("| Split | Cases |\n")
        f.write("|-------|-------|\n")
        f.write(f"| Train | {', '.join(split_map['train'])} |\n")
        f.write(f"| Validation | {', '.join(split_map['val'])} |\n")
        f.write(f"| Test | {', '.join(split_map['test'])} |\n\n")

        f.write("## Split Totals After Saving\n\n")
        f.write("| Split | Nodes | Positive | Negative | Positive Ratio |\n")
        f.write("|-------|-------|----------|----------|----------------|\n")
        f.write(f"| Train | {train_nodes} | {train_pos} | {train_neg} | {train_ratio:.4f} |\n")
        f.write(f"| Validation | {val_nodes} | {val_pos} | {val_neg} | {val_ratio:.4f} |\n")
        f.write(f"| Test | {test_nodes} | {test_pos} | {test_neg} | {test_ratio:.4f} |\n")
        f.write(f"| Total | {total_nodes} | {total_pos} | {total_neg} | {total_ratio:.4f} |\n\n")

        f.write("## Per-Case Summary\n\n")
        f.write(
            "| Case | Split | Original Nodes | Original Pos | Original Neg | "
            "Saved Nodes | Saved Pos | Saved Neg | Saved Pos Ratio | Edges | Hard Neg? |\n"
        )
        f.write(
            "|------|-------|----------------|--------------|--------------|"
            "-------------|-----------|-----------|-----------------|-------|-----------|\n"
        )

        for item in summary:
            f.write(
                f"| {item['case']} "
                f"| {item['split']} "
                f"| {item['original_nodes']} "
                f"| {item['original_positive']} "
                f"| {item['original_negative']} "
                f"| {item['saved_nodes']} "
                f"| {item['saved_positive']} "
                f"| {item['saved_negative']} "
                f"| {item['saved_positive_ratio']:.4f} "
                f"| {item['edges']} "
                f"| {item['hard_negative_applied']} |\n"
            )

        f.write("\n## Interpretation\n\n")
        f.write("- Training graphs use all positive pairs and nearest non-contact hard negatives.\n")
        f.write("- Validation and test graphs keep the natural class imbalance.\n")
        f.write("- This setup tests whether targeted hard negatives can reduce false positives under realistic evaluation.\n")

    print(f"Saved: {path}")


# ----------------------------
# Build one case
# ----------------------------
def build_case(row, split, args):
    case_name = row["case_name"].strip()
    partner1 = row["partner1_chains"].strip()
    partner2 = row["partner2_chains"].strip()

    print(f"\nProcessing {case_name} ({partner1} vs {partner2}) | split={split}")

    pdb_path = find_pdb_path(row)

    structure = load_structure(pdb_path)
    model = next(structure.get_models())

    resA_atoms, caA = extract_partner_residues(model, partner1)
    resB_atoms, caB = extract_partner_residues(model, partner2)

    if len(resA_atoms) == 0 or len(resB_atoms) == 0:
        raise RuntimeError(f"Empty partner in {case_name}")

    edge_A = build_edge_index_from_ca(caA, args.intra_ca_edge_threshold)
    edge_B = build_edge_index_from_ca(caB, args.intra_ca_edge_threshold)

    adjA = build_adjacency_list(edge_A, len(resA_atoms))
    adjB = build_adjacency_list(edge_B, len(resB_atoms))

    degree_A = np.array([len(neighbors) for neighbors in adjA])
    degree_B = np.array([len(neighbors) for neighbors in adjB])

    idx_A, idx_B = candidate_filter(caA, caB, args.candidate_radius)

    print(
        f"Residues A={len(resA_atoms)}, B={len(resB_atoms)} | "
        f"Keep A={len(idx_A)}, Keep B={len(idx_B)}"
    )

    if len(idx_A) == 0 or len(idx_B) == 0:
        raise RuntimeError(f"Candidate filter removed all residues in {case_name}")

    all_pairs, all_labels, all_ca_distances = build_all_candidate_pairs_labels_and_distances(
        resA_atoms=resA_atoms,
        resB_atoms=resB_atoms,
        caA=caA,
        caB=caB,
        idx_A=idx_A,
        idx_B=idx_B,
        contact_threshold=args.contact_threshold,
    )

    original_nodes = len(all_labels)
    original_positive = int(all_labels.sum())
    original_negative = int(original_nodes - original_positive)
    original_ratio = original_positive / original_nodes if original_nodes > 0 else 0.0

    hard_negative_applied = split == "train"

    if hard_negative_applied:
        pairs, labels, ca_distances = select_hard_negative_train_pairs(
            pairs=all_pairs,
            labels=all_labels,
            ca_distances=all_ca_distances,
            hard_negative_ratio=args.hard_negative_ratio,
        )
    else:
        pairs, labels, ca_distances = all_pairs, all_labels, all_ca_distances

    features = build_features(
        pairs=pairs,
        caA=caA,
        caB=caB,
        degree_A=degree_A,
        degree_B=degree_B,
    )

    corr_edge_index = build_correspondence_edges_arbitrary_pairs(
        adjA=adjA,
        adjB=adjB,
        pairs=pairs,
        max_edges=args.max_corr_edges,
    )

    saved_nodes = len(labels)
    saved_positive = int(labels.sum())
    saved_negative = int(saved_nodes - saved_positive)
    saved_ratio = saved_positive / saved_nodes if saved_nodes > 0 else 0.0

    selected_neg_distances = ca_distances[labels == 0]

    if len(selected_neg_distances) > 0:
        max_selected_negative_ca_distance = float(selected_neg_distances.max())
        mean_selected_negative_ca_distance = float(selected_neg_distances.mean())
    else:
        max_selected_negative_ca_distance = 0.0
        mean_selected_negative_ca_distance = 0.0

    print(
        f"Original: nodes={original_nodes}, pos={original_positive}, "
        f"neg={original_negative}, ratio={original_ratio:.4f}"
    )

    print(
        f"Saved: nodes={saved_nodes}, pos={saved_positive}, "
        f"neg={saved_negative}, ratio={saved_ratio:.4f}, "
        f"edges={corr_edge_index.shape[1]}, feature_dim={features.shape[1]}"
    )

    if hard_negative_applied:
        print(
            f"Hard negative CA distance: "
            f"mean={mean_selected_negative_ca_distance:.3f}, "
            f"max={max_selected_negative_ca_distance:.3f}"
        )

    os.makedirs(args.out_dir, exist_ok=True)

    np.save(os.path.join(args.out_dir, f"{case_name}_corr_labels.npy"), labels)
    np.save(os.path.join(args.out_dir, f"{case_name}_corr_pairs.npy"), pairs)
    np.save(os.path.join(args.out_dir, f"{case_name}_corr_edge_index.npy"), corr_edge_index)
    np.save(os.path.join(args.out_dir, f"{case_name}_corr_features.npy"), features)

    return {
        "case": case_name,
        "split": split,
        "original_nodes": original_nodes,
        "original_positive": original_positive,
        "original_negative": original_negative,
        "original_positive_ratio": original_ratio,
        "saved_nodes": saved_nodes,
        "saved_positive": saved_positive,
        "saved_negative": saved_negative,
        "saved_positive_ratio": saved_ratio,
        "edges": int(corr_edge_index.shape[1]),
        "feature_dim": int(features.shape[1]),
        "hard_negative_applied": str(hard_negative_applied),
        "hard_negative_ratio": int(args.hard_negative_ratio),
        "max_selected_negative_ca_distance": max_selected_negative_ca_distance,
        "mean_selected_negative_ca_distance": mean_selected_negative_ca_distance,
    }


# ----------------------------
# Main
# ----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Build hard-negative train / natural-val-test dataset."
    )

    parser.add_argument(
        "--cases",
        default=os.path.join("data", "cases", "expanded_cases.csv"),
    )

    parser.add_argument(
        "--out_dir",
        default=os.path.join("data", "processed_hard_negative_train_natural_test"),
    )

    parser.add_argument("--hard_negative_ratio", type=int, default=5)
    parser.add_argument("--contact_threshold", type=float, default=CONTACT_THRESHOLD)
    parser.add_argument("--intra_ca_edge_threshold", type=float, default=INTRA_CA_EDGE_THRESHOLD)
    parser.add_argument("--candidate_radius", type=float, default=CANDIDATE_RADIUS)
    parser.add_argument("--max_corr_edges", type=int, default=MAX_CORR_EDGES)

    return parser.parse_args()


def main():
    args = parse_args()

    rows = read_cases(args.cases)
    split_map = build_split_map(rows)

    case_to_split = {}

    for split, cases in split_map.items():
        for case_name in cases:
            case_to_split[case_name] = split

    print("\nHard-negative train / natural-evaluation dataset build")
    print(f"Cases CSV: {args.cases}")
    print(f"Output dir: {args.out_dir}")
    print(f"Hard negative ratio: {args.hard_negative_ratio}")
    print("\nTrain:", split_map["train"])
    print("Validation:", split_map["val"])
    print("Test:", split_map["test"])

    summary = []

    for row in rows:
        case_name = row["case_name"].strip()
        split = case_to_split.get(case_name, "train")

        item = build_case(row, split, args)
        summary.append(item)

    os.makedirs(args.out_dir, exist_ok=True)

    split_path = os.path.join(args.out_dir, "split_cases.json")

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_map, f, indent=2)

    print(f"\nSaved: {split_path}")

    summary_csv = os.path.join(args.out_dir, "hard_negative_train_natural_eval_summary.csv")
    summary_md = os.path.join(args.out_dir, "hard_negative_train_natural_eval_summary.md")

    save_summary_csv(summary_csv, summary)
    save_summary_md(summary_md, summary, split_map, args)

    total_saved_nodes = sum(item["saved_nodes"] for item in summary)
    total_saved_pos = sum(item["saved_positive"] for item in summary)
    total_saved_neg = sum(item["saved_negative"] for item in summary)
    total_ratio = total_saved_pos / total_saved_nodes if total_saved_nodes > 0 else 0.0

    print("\n================ DATASET SUMMARY ================")
    print(f"Cases built: {len(summary)}")
    print(f"Total saved nodes: {total_saved_nodes}")
    print(f"Total saved positive: {total_saved_pos}")
    print(f"Total saved negative: {total_saved_neg}")
    print(f"Total saved positive ratio: {total_ratio:.4f}")
    print("=================================================")


if __name__ == "__main__":
    main()
