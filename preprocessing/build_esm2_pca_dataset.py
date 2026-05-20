import os
import json
import csv
import shutil
import pickle
import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_split_map(source_processed_dir):
    split_file = os.path.join(source_processed_dir, "split_cases.json")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file, "r", encoding="utf-8") as f:
        return json.load(f)


def all_cases_from_split(split_map):
    cases = []
    for split in ["train", "val", "test"]:
        cases.extend(split_map.get(split, []))
    return cases


def case_to_split_map(split_map):
    result = {}
    for split, cases in split_map.items():
        for case_name in cases:
            result[case_name] = split
    return result


def load_case_arrays(source_processed_dir, case_name):
    features = np.load(os.path.join(source_processed_dir, f"{case_name}_corr_features.npy"))
    labels = np.load(os.path.join(source_processed_dir, f"{case_name}_corr_labels.npy"))
    pairs = np.load(os.path.join(source_processed_dir, f"{case_name}_corr_pairs.npy"))
    edge_index = np.load(os.path.join(source_processed_dir, f"{case_name}_corr_edge_index.npy"))
    return features, labels, pairs, edge_index


def load_embeddings(embedding_dir, case_name):
    emb1_path = os.path.join(embedding_dir, f"{case_name}_partner1_esm2.npy")
    emb2_path = os.path.join(embedding_dir, f"{case_name}_partner2_esm2.npy")
    if not os.path.exists(emb1_path):
        raise FileNotFoundError(f"Missing embedding file: {emb1_path}")
    if not os.path.exists(emb2_path):
        raise FileNotFoundError(f"Missing embedding file: {emb2_path}")
    return np.load(emb1_path), np.load(emb2_path)


def build_raw_pair_esm_features(pairs, emb1, emb2):
    a_idx = pairs[:, 0].astype(np.int64)
    b_idx = pairs[:, 1].astype(np.int64)

    if len(a_idx) and a_idx.max() >= emb1.shape[0]:
        raise IndexError(f"Partner 1 pair index out of range: max={a_idx.max()}, embeddings={emb1.shape[0]}")
    if len(b_idx) and b_idx.max() >= emb2.shape[0]:
        raise IndexError(f"Partner 2 pair index out of range: max={b_idx.max()}, embeddings={emb2.shape[0]}")

    emb_a = emb1[a_idx]
    emb_b = emb2[b_idx]
    emb_diff = np.abs(emb_a - emb_b)
    return np.concatenate([emb_a, emb_b, emb_diff], axis=1).astype(np.float32)


def build_train_matrix(source_processed_dir, embedding_dir, train_cases):
    matrices = []
    for case_name in train_cases:
        _, _, pairs, _ = load_case_arrays(source_processed_dir, case_name)
        emb1, emb2 = load_embeddings(embedding_dir, case_name)
        raw_pair = build_raw_pair_esm_features(pairs, emb1, emb2)
        matrices.append(raw_pair)
        print(f"Loaded train ESM pair matrix for {case_name}: {raw_pair.shape}")
    return np.concatenate(matrices, axis=0)


def save_summary_csv(path, rows):
    fieldnames = ["case", "split", "nodes", "positive", "negative", "feature_dim", "esm_raw_dim", "pca_components"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_summary_md(path, rows, args, final_dim, raw_dim, actual_components, explained_var):
    total_nodes = sum(row["nodes"] for row in rows)
    total_pos = sum(row["positive"] for row in rows)
    total_neg = sum(row["negative"] for row in rows)
    ratio = total_pos / total_nodes if total_nodes else 0.0

    with open(path, "w", encoding="utf-8") as f:
        f.write("# ESM-2 PCA Pair-Feature Dataset Summary\n\n")
        f.write("## Configuration\n\n")
        f.write(f"- Source processed directory: `{args.source_processed_dir}`\n")
        f.write(f"- Embedding directory: `{args.embedding_dir}`\n")
        f.write(f"- Output directory: `{args.out_dir}`\n")
        f.write(f"- Basic feature count: `{args.basic_feature_count}`\n")
        f.write(f"- Raw ESM pair dimension: `{raw_dim}`\n")
        f.write(f"- PCA components: `{actual_components}`\n")
        f.write(f"- Explained variance ratio sum: `{explained_var:.4f}`\n")
        f.write(f"- Final feature dimension: `{final_dim}`\n\n")
        f.write("Feature vector:\n\n")
        f.write("```text\n[basic_3_features, PCA(ESM_A, ESM_B, abs(ESM_A - ESM_B))]\n```\n\n")
        f.write("PCA and standardization are fitted only on training pairs to avoid validation/test leakage.\n\n")
        f.write("## Per-Case Summary\n\n")
        f.write("| Case | Split | Nodes | Positive | Negative | Feature Dim |\n")
        f.write("|------|-------|-------|----------|----------|-------------|\n")
        for row in rows:
            f.write(f"| {row['case']} | {row['split']} | {row['nodes']} | {row['positive']} | {row['negative']} | {row['feature_dim']} |\n")
        f.write("\n## Total\n\n")
        f.write("| Total Nodes | Total Positive | Total Negative | Positive Ratio |\n")
        f.write("|-------------|----------------|----------------|----------------|\n")
        f.write(f"| {total_nodes} | {total_pos} | {total_neg} | {ratio:.4f} |\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Build pair-level ESM-2 PCA features for processed correspondence graphs.")
    parser.add_argument("--source_processed_dir", default=os.path.join("data", "processed_combined_current_bm5_train_balanced_natural_test"))
    parser.add_argument("--embedding_dir", default=os.path.join("data", "esm2_embeddings"))
    parser.add_argument("--out_dir", default=os.path.join("data", "processed_combined_current_bm5_esm2_pca64"))
    parser.add_argument("--pca_components", type=int, default=64)
    parser.add_argument("--basic_feature_count", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("\nBuilding ESM-2 PCA correspondence features")
    print(f"Source processed dir: {args.source_processed_dir}")
    print(f"Embedding dir: {args.embedding_dir}")
    print(f"Output dir: {args.out_dir}")
    print(f"Requested PCA components: {args.pca_components}")

    split_map = load_split_map(args.source_processed_dir)
    train_cases = split_map["train"]
    all_cases = all_cases_from_split(split_map)
    split_lookup = case_to_split_map(split_map)

    train_matrix = build_train_matrix(args.source_processed_dir, args.embedding_dir, train_cases)
    raw_dim = train_matrix.shape[1]
    actual_components = min(args.pca_components, raw_dim, train_matrix.shape[0])

    print(f"Train pair ESM matrix shape: {train_matrix.shape}")
    print(f"Raw ESM pair feature dimension: {raw_dim}")
    print(f"Actual PCA components: {actual_components}")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_matrix)

    pca = PCA(n_components=actual_components, random_state=42)
    pca.fit(train_scaled)
    explained_var = float(np.sum(pca.explained_variance_ratio_))

    with open(os.path.join(args.out_dir, "esm2_pair_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(args.out_dir, "esm2_pair_pca.pkl"), "wb") as f:
        pickle.dump(pca, f)

    shutil.copyfile(os.path.join(args.source_processed_dir, "split_cases.json"), os.path.join(args.out_dir, "split_cases.json"))

    rows = []
    final_dim = None

    for case_name in all_cases:
        split = split_lookup[case_name]
        basic_features, labels, pairs, edge_index = load_case_arrays(args.source_processed_dir, case_name)
        emb1, emb2 = load_embeddings(args.embedding_dir, case_name)
        raw_pair = build_raw_pair_esm_features(pairs, emb1, emb2)
        esm_pca = pca.transform(scaler.transform(raw_pair)).astype(np.float32)

        basic = basic_features[:, :args.basic_feature_count].astype(np.float32)
        final_features = np.concatenate([basic, esm_pca], axis=1).astype(np.float32)
        final_dim = int(final_features.shape[1])

        np.save(os.path.join(args.out_dir, f"{case_name}_corr_features.npy"), final_features)
        np.save(os.path.join(args.out_dir, f"{case_name}_corr_labels.npy"), labels)
        np.save(os.path.join(args.out_dir, f"{case_name}_corr_pairs.npy"), pairs)
        np.save(os.path.join(args.out_dir, f"{case_name}_corr_edge_index.npy"), edge_index)

        pos = int(labels.sum())
        neg = int(len(labels) - pos)
        rows.append({
            "case": case_name,
            "split": split,
            "nodes": int(len(labels)),
            "positive": pos,
            "negative": neg,
            "feature_dim": final_dim,
            "esm_raw_dim": raw_dim,
            "pca_components": actual_components,
        })

        print(f"{case_name}: split={split}, nodes={len(labels)}, positive={pos}, feature_dim={final_dim}")

    info = {
        "source_processed_dir": args.source_processed_dir,
        "embedding_dir": args.embedding_dir,
        "out_dir": args.out_dir,
        "raw_esm_pair_dim": int(raw_dim),
        "pca_components": int(actual_components),
        "basic_feature_count": int(args.basic_feature_count),
        "final_feature_dim": int(final_dim),
        "explained_variance_ratio_sum": explained_var,
    }

    with open(os.path.join(args.out_dir, "esm2_pca_feature_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    summary_csv = os.path.join(args.out_dir, "esm2_pca_dataset_summary.csv")
    summary_md = os.path.join(args.out_dir, "esm2_pca_dataset_summary.md")
    save_summary_csv(summary_csv, rows)
    save_summary_md(summary_md, rows, args, final_dim, raw_dim, actual_components, explained_var)

    print(f"\nSaved: {summary_csv}")
    print(f"Saved: {summary_md}")
    print("ESM-2 PCA feature dataset completed successfully.")


if __name__ == "__main__":
    main()
