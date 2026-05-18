import os
import csv
import json
import argparse
import random

import numpy as np
from Bio.PDB import PDBParser

CONTACT_THRESHOLD = 5.0
INTRA_CA_EDGE_THRESHOLD = 8.0
CANDIDATE_RADIUS = 12.0
MAX_CORR_EDGES = 3_000_000
RAW_DIRS = [os.path.join("data", "raw_pdb_expanded"), os.path.join("data", "raw_pdb")]

HYDRO = {"ILE":4.5,"VAL":4.2,"LEU":3.8,"PHE":2.8,"CYS":2.5,"MET":1.9,"ALA":1.8,"GLY":-0.4,"THR":-0.7,"SER":-0.8,"TRP":-0.9,"TYR":-1.3,"PRO":-1.6,"HIS":-3.2,"GLU":-3.5,"GLN":-3.5,"ASP":-3.5,"ASN":-3.5,"LYS":-3.9,"ARG":-4.5}
CHARGE = {"ARG":1.0,"LYS":1.0,"HIS":1.0,"ASP":-1.0,"GLU":-1.0}
POLAR = {"ARG","ASN","ASP","GLN","GLU","HIS","LYS","SER","THR","TYR","CYS"}
AROM = {"PHE","TRP","TYR","HIS"}

try:
    from Bio.PDB.SASA import ShrakeRupley
except ImportError:
    ShrakeRupley = None


def find_pdb(row):
    explicit = row.get("pdb_file", "").strip()
    if explicit and os.path.exists(explicit):
        return explicit
    names = []
    for key in ["pdb_id", "case_name"]:
        value = row.get(key, "").strip()
        if value:
            names += [value, value.lower(), value.upper()]
    for raw in RAW_DIRS:
        for name in names:
            p = os.path.join(raw, f"{name}.pdb")
            if os.path.exists(p):
                return p
    raise FileNotFoundError(f"No PDB found for {row.get('case_name')}")


def load_model(path):
    return next(PDBParser(QUIET=True).get_structure("protein", path).get_models())


def compute_asa(model):
    if ShrakeRupley is not None:
        ShrakeRupley().compute(model, level="R")


def extract(model, chain_ids):
    atoms, ca, names, asa = [], [], [], []
    for cid in chain_ids:
        if cid not in model:
            raise ValueError(f"Chain {cid} not found. Available: {list(model.child_dict.keys())}")
        for res in model[cid]:
            if res.get_id()[0] != " " or not res.has_id("CA"):
                continue
            aa = [a.get_coord() for a in res]
            if not aa:
                continue
            atoms.append(np.asarray(aa, dtype=np.float32))
            ca.append(res["CA"].get_coord())
            names.append(res.get_resname())
            asa.append(float(getattr(res, "sasa", 0.0)))
    return atoms, np.asarray(ca, dtype=np.float32), names, np.asarray(asa, dtype=np.float32)


def edge_index(ca, thr):
    edges = []
    for i in range(len(ca)):
        for j in range(i + 1, len(ca)):
            if np.linalg.norm(ca[i] - ca[j]) < thr:
                edges.append((i, j)); edges.append((j, i))
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


def adjacency(ei, n):
    adj = [set() for _ in range(n)]
    if ei.size:
        for s, d in ei.T:
            adj[int(s)].add(int(d))
    return adj


def candidate_filter(ca_a, ca_b, radius):
    idx_a = [i for i in range(len(ca_a)) if np.any(np.linalg.norm(ca_b - ca_a[i], axis=1) < radius)]
    idx_b = [j for j in range(len(ca_b)) if np.any(np.linalg.norm(ca_a - ca_b[j], axis=1) < radius)]
    return np.asarray(idx_a, dtype=np.int64), np.asarray(idx_b, dtype=np.int64)


def contact(a_atoms, b_atoms, thr):
    for a in a_atoms:
        for b in b_atoms:
            if np.linalg.norm(a - b) < thr:
                return 1
    return 0


def build_pairs(atoms_a, atoms_b, idx_a, idx_b, thr):
    pairs, labels = [], []
    for a in idx_a:
        for b in idx_b:
            pairs.append((int(a), int(b)))
            labels.append(contact(atoms_a[a], atoms_b[b], thr))
    return np.asarray(pairs, dtype=np.int64), np.asarray(labels, dtype=np.int64)


def balance(pairs, labels, ratio, seed):
    rng = np.random.default_rng(seed)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    if len(pos) == 0:
        return pairs, labels
    n_neg = min(len(neg), len(pos) * ratio)
    keep = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)])
    rng.shuffle(keep)
    return pairs[keep], labels[keep]


def phys(name):
    return np.asarray([HYDRO.get(name, 0.0), CHARGE.get(name, 0.0), 1.0 if name in POLAR else 0.0, 1.0 if name in AROM else 0.0], dtype=np.float32)


def feature(a, b, ca_a, ca_b, deg_a, deg_b, names_a, names_b, asa_a, asa_b, mode):
    basic = np.asarray([np.linalg.norm(ca_a[a] - ca_b[b]), deg_a[a], deg_b[b]], dtype=np.float32)
    if mode == "basic":
        return basic
    if mode == "physicochemical":
        return np.concatenate([basic, phys(names_a[a]), phys(names_b[b])])
    if mode == "basic_asa":
        return np.concatenate([basic, np.asarray([asa_a[a], asa_b[b]], dtype=np.float32)])
    if mode == "physicochemical_asa":
        return np.concatenate([basic, phys(names_a[a]), phys(names_b[b]), np.asarray([asa_a[a], asa_b[b]], dtype=np.float32)])
    raise ValueError(f"Unknown feature mode: {mode}")


def corr_edges(adj_a, adj_b, pairs, max_edges):
    pair_to_node = {(int(a), int(b)): i for i, (a, b) in enumerate(pairs)}
    edges = []
    for i, (a, b) in enumerate(pairs):
        for a2 in adj_a[int(a)]:
            for b2 in adj_b[int(b)]:
                j = pair_to_node.get((int(a2), int(b2)))
                if j is not None:
                    edges.append((i, j))
                    if len(edges) >= max_edges:
                        return np.asarray(edges, dtype=np.int64).T
    if not edges:
        return np.zeros((2, 0), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64).T


def read_cases(path):
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("enabled", "1").strip() not in {"0", "false", "False"}]


def assign_splits(rows, seed):
    splits = {"train": [], "val": [], "test": []}
    missing = []
    for r in rows:
        s = r.get("split", "").lower().strip()
        if s in {"train", "training"}: splits["train"].append(r["case_name"])
        elif s in {"val", "valid", "validation"}: splits["val"].append(r["case_name"])
        elif s == "test": splits["test"].append(r["case_name"])
        else: missing.append(r["case_name"])
    if missing:
        rng = random.Random(seed); rng.shuffle(missing)
        n = len(missing); n_test = max(1, round(n * 0.15)) if n >= 3 else 0; n_val = max(1, round(n * 0.15)) if n >= 3 else 0
        splits["test"] += missing[:n_test]
        splits["val"] += missing[n_test:n_test+n_val]
        splits["train"] += missing[n_test+n_val:]
    return splits


def build_case(row, split, args):
    name = row["case_name"]
    print(f"\nProcessing {name} ({row['partner1_chains']} vs {row['partner2_chains']})")
    model = load_model(find_pdb(row))
    if "asa" in args.feature_mode:
        compute_asa(model)
    atoms_a, ca_a, names_a, asa_a = extract(model, row["partner1_chains"])
    atoms_b, ca_b, names_b, asa_b = extract(model, row["partner2_chains"])
    ei_a = edge_index(ca_a, args.intra_ca_edge_threshold); ei_b = edge_index(ca_b, args.intra_ca_edge_threshold)
    adj_a = adjacency(ei_a, len(atoms_a)); adj_b = adjacency(ei_b, len(atoms_b))
    deg_a = np.asarray([len(x) for x in adj_a]); deg_b = np.asarray([len(x) for x in adj_b])
    idx_a, idx_b = candidate_filter(ca_a, ca_b, args.candidate_radius)
    pairs, labels = build_pairs(atoms_a, atoms_b, idx_a, idx_b, args.contact_threshold)
    print(f"Original: nodes={len(labels)}, positive={int(labels.sum())}, negative={int(len(labels)-labels.sum())}")
    if args.balanced:
        pairs, labels = balance(pairs, labels, args.dataset_negative_ratio, args.seed)
    x = np.asarray([feature(int(a), int(b), ca_a, ca_b, deg_a, deg_b, names_a, names_b, asa_a, asa_b, args.feature_mode) for a, b in pairs], dtype=np.float32)
    ei = corr_edges(adj_a, adj_b, pairs, args.max_corr_edges)
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, f"{name}_corr_features.npy"), x)
    np.save(os.path.join(args.out_dir, f"{name}_corr_labels.npy"), labels)
    np.save(os.path.join(args.out_dir, f"{name}_corr_pairs.npy"), pairs)
    np.save(os.path.join(args.out_dir, f"{name}_corr_edge_index.npy"), ei)
    pos = int(labels.sum()); neg = int(len(labels) - pos); ratio = pos / len(labels) if len(labels) else 0.0
    print(f"Saved: nodes={len(labels)}, positive={pos}, negative={neg}, pos_ratio={ratio:.4f}, edges={ei.shape[1]}, feature_dim={x.shape[1]}")
    return {"case": name, "split": split, "nodes": len(labels), "positive": pos, "negative": neg, "positive_ratio": ratio, "edges": int(ei.shape[1]), "feature_dim": int(x.shape[1])}


def write_summary(out_dir, summary, splits, args):
    with open(os.path.join(out_dir, "split_cases.json"), "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    with open(os.path.join(out_dir, "expanded_dataset_summary.csv"), "w", newline="", encoding="utf-8") as f:
        fields = ["case","split","nodes","positive","negative","positive_ratio","edges","feature_dim"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summary)
    total_nodes = sum(x["nodes"] for x in summary); total_pos = sum(x["positive"] for x in summary); total_neg = sum(x["negative"] for x in summary)
    with open(os.path.join(out_dir, "expanded_dataset_summary.md"), "w", encoding="utf-8") as f:
        f.write("# Expanded Balanced Dataset Summary\n\n")
        f.write(f"Feature mode: `{args.feature_mode}`  \nBalanced: `{args.balanced}`  \nNegative ratio: `{args.dataset_negative_ratio}`\n\n")
        f.write("| Case | Split | Nodes | Positive | Negative | Pos Ratio | Edges | Feature Dim |\n")
        f.write("|------|-------|-------|----------|----------|-----------|-------|-------------|\n")
        for r in summary:
            f.write(f"| {r['case']} | {r['split']} | {r['nodes']} | {r['positive']} | {r['negative']} | {r['positive_ratio']:.4f} | {r['edges']} | {r['feature_dim']} |\n")
        f.write("\n## Total\n\n")
        f.write("| Nodes | Positive | Negative | Pos Ratio |\n|-------|----------|----------|-----------|\n")
        f.write(f"| {total_nodes} | {total_pos} | {total_neg} | {(total_pos/total_nodes if total_nodes else 0):.4f} |\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join("data", "cases", "expanded_cases_accepted.csv"))
    ap.add_argument("--out_dir", default=os.path.join("data", "processed_expanded"))
    ap.add_argument("--feature_mode", default="basic", choices=["basic", "physicochemical", "basic_asa", "physicochemical_asa"])
    ap.add_argument("--balanced", action="store_true", default=True)
    ap.add_argument("--unbalanced", dest="balanced", action="store_false")
    ap.add_argument("--dataset_negative_ratio", type=int, default=3)
    ap.add_argument("--contact_threshold", type=float, default=CONTACT_THRESHOLD)
    ap.add_argument("--intra_ca_edge_threshold", type=float, default=INTRA_CA_EDGE_THRESHOLD)
    ap.add_argument("--candidate_radius", type=float, default=CANDIDATE_RADIUS)
    ap.add_argument("--max_corr_edges", type=int, default=MAX_CORR_EDGES)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rows = read_cases(args.cases); splits = assign_splits(rows, args.seed); c2s = {c: s for s, cs in splits.items() for c in cs}
    print(f"Expanded build | cases={len(rows)} | feature_mode={args.feature_mode} | balanced={args.balanced}")
    summary = []
    for row in rows:
        try:
            summary.append(build_case(row, c2s.get(row["case_name"], "train"), args))
        except Exception as e:
            print(f"Warning: failed {row.get('case_name')}: {type(e).__name__}: {e}")
    write_summary(args.out_dir, summary, splits, args)
    print(f"\nSaved expanded dataset to: {args.out_dir}")

if __name__ == "__main__":
    main()
