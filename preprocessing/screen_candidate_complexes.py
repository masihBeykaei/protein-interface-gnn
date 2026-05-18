import os
import csv
import argparse
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

CONTACT_THRESHOLD = 5.0
INTRA_CA_EDGE_THRESHOLD = 8.0
CANDIDATE_RADIUS = 12.0
RAW_DIRS = [os.path.join("data", "raw_pdb_expanded"), os.path.join("data", "raw_pdb")]


def load_structure(path):
    return PDBParser(QUIET=True).get_structure("protein", path)


def find_pdb(row):
    explicit = row.get("pdb_file", "").strip()
    if explicit:
        if os.path.exists(explicit):
            return explicit
        raise FileNotFoundError(explicit)

    names = []
    for key in ["pdb_id", "case_name"]:
        value = row.get(key, "").strip()
        if value:
            names += [value, value.lower(), value.upper()]

    for raw_dir in RAW_DIRS:
        for name in names:
            path = os.path.join(raw_dir, f"{name}.pdb")
            if os.path.exists(path):
                return path

    raise FileNotFoundError(f"No PDB found for {row.get('case_name')}")


def extract_residues(model, chain_ids):
    atoms_list, ca = [], []
    for chain_id in chain_ids:
        if chain_id not in model:
            raise ValueError(f"Chain {chain_id} not found. Available: {list(model.child_dict.keys())}")
        for res in model[chain_id]:
            if res.get_id()[0] != " " or not res.has_id("CA"):
                continue
            atoms = [a.get_coord() for a in res]
            if not atoms:
                continue
            atoms_list.append(np.asarray(atoms, dtype=np.float32))
            ca.append(res["CA"].get_coord())
    return atoms_list, np.asarray(ca, dtype=np.float32)


def ca_edges(ca, threshold):
    n = len(ca)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(ca[i] - ca[j]) < threshold:
                count += 2
    return count


def candidate_filter(ca_a, ca_b, radius):
    keep_a = [i for i in range(len(ca_a)) if np.any(np.linalg.norm(ca_b - ca_a[i], axis=1) < radius)]
    keep_b = [j for j in range(len(ca_b)) if np.any(np.linalg.norm(ca_a - ca_b[j], axis=1) < radius)]
    return np.asarray(keep_a, dtype=np.int64), np.asarray(keep_b, dtype=np.int64)


def contact(res_a, res_b, threshold):
    for a in res_a:
        for b in res_b:
            if np.linalg.norm(a - b) < threshold:
                return 1
    return 0


def count_labels(atoms_a, atoms_b, idx_a, idx_b, threshold):
    pos = 0
    for a in idx_a:
        for b in idx_b:
            pos += contact(atoms_a[a], atoms_b[b], threshold)
    total = len(idx_a) * len(idx_b)
    return pos, total - pos, total


def read_cases(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("enabled", "1").strip() not in {"0", "false", "False"}]


def discover_auto(auto_dir):
    rows = []
    p = Path(auto_dir)
    if not p.exists():
        return rows
    for pdb in sorted(p.glob("*.pdb")):
        rows.append({
            "case_name": pdb.stem,
            "pdb_id": pdb.stem,
            "pdb_file": str(pdb),
            "partner1_chains": "A",
            "partner2_chains": "B",
            "source": "auto_ab",
            "split": "",
            "enabled": "1",
        })
    return rows


def unique(rows):
    seen, out = set(), []
    for r in rows:
        key = (r.get("case_name", ""), r.get("pdb_file", ""), r.get("partner1_chains", ""), r.get("partner2_chains", ""))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def screen(row, args):
    result = {
        "case_name": row.get("case_name", ""),
        "pdb_id": row.get("pdb_id", ""),
        "pdb_file": row.get("pdb_file", ""),
        "partner1_chains": row.get("partner1_chains", ""),
        "partner2_chains": row.get("partner2_chains", ""),
        "source": row.get("source", ""),
        "split": row.get("split", ""),
        "status": "reject",
        "reason": "",
        "residues_A": 0,
        "residues_B": 0,
        "graph_A_edges": 0,
        "graph_B_edges": 0,
        "keep_A": 0,
        "keep_B": 0,
        "candidate_nodes": 0,
        "positive": 0,
        "negative": 0,
        "positive_ratio": 0.0,
    }
    try:
        pdb = find_pdb(row)
        result["pdb_file"] = pdb
        model = next(load_structure(pdb).get_models())
        atoms_a, ca_a = extract_residues(model, row["partner1_chains"])
        atoms_b, ca_b = extract_residues(model, row["partner2_chains"])
        result["residues_A"], result["residues_B"] = len(atoms_a), len(atoms_b)
        if not atoms_a or not atoms_b:
            result["reason"] = "empty_partner"
            return result
        result["graph_A_edges"] = ca_edges(ca_a, args.intra_ca_edge_threshold)
        result["graph_B_edges"] = ca_edges(ca_b, args.intra_ca_edge_threshold)
        idx_a, idx_b = candidate_filter(ca_a, ca_b, args.candidate_radius)
        result["keep_A"], result["keep_B"] = len(idx_a), len(idx_b)
        pos, neg, total = count_labels(atoms_a, atoms_b, idx_a, idx_b, args.contact_threshold)
        result["candidate_nodes"] = total
        result["positive"], result["negative"] = pos, neg
        result["positive_ratio"] = pos / total if total else 0.0
        reasons = []
        if pos < args.min_positive:
            reasons.append(f"positive<{args.min_positive}")
        if result["positive_ratio"] < args.min_positive_ratio:
            reasons.append(f"positive_ratio<{args.min_positive_ratio}")
        if total > args.max_candidate_nodes:
            reasons.append(f"candidate_nodes>{args.max_candidate_nodes}")
        if reasons:
            result["reason"] = ";".join(reasons)
        else:
            result["status"] = "accept"
            result["reason"] = "passed"
        return result
    except Exception as e:
        result["reason"] = f"error:{type(e).__name__}:{e}"
        return result


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=os.path.join("data", "cases", "expanded_cases.csv"))
    ap.add_argument("--auto_from_dir", default=os.path.join("data", "raw_pdb_expanded"))
    ap.add_argument("--output", default=os.path.join("data", "cases", "expanded_screening_results.csv"))
    ap.add_argument("--accepted_output", default=os.path.join("data", "cases", "expanded_cases_accepted.csv"))
    ap.add_argument("--contact_threshold", type=float, default=CONTACT_THRESHOLD)
    ap.add_argument("--intra_ca_edge_threshold", type=float, default=INTRA_CA_EDGE_THRESHOLD)
    ap.add_argument("--candidate_radius", type=float, default=CANDIDATE_RADIUS)
    ap.add_argument("--min_positive", type=int, default=30)
    ap.add_argument("--min_positive_ratio", type=float, default=0.02)
    ap.add_argument("--max_candidate_nodes", type=int, default=8000)
    args = ap.parse_args()

    rows = unique(read_cases(args.cases) + discover_auto(args.auto_from_dir))
    print(f"Total candidate cases: {len(rows)}")
    results = []
    for i, row in enumerate(rows, 1):
        print(f"\n[{i}/{len(rows)}] Screening {row['case_name']} ({row['partner1_chains']} vs {row['partner2_chains']})")
        r = screen(row, args)
        results.append(r)
        print(f"{r['status']} | {r['reason']} | nodes={r['candidate_nodes']} | pos={r['positive']} | ratio={r['positive_ratio']:.4f}")

    fields = ["case_name","pdb_id","pdb_file","partner1_chains","partner2_chains","source","split","status","reason","residues_A","residues_B","graph_A_edges","graph_B_edges","keep_A","keep_B","candidate_nodes","positive","negative","positive_ratio"]
    write_csv(args.output, results, fields)
    accepted = []
    for r in results:
        if r["status"] == "accept":
            accepted.append({k: r[k] for k in ["case_name","pdb_id","pdb_file","partner1_chains","partner2_chains","source","split"]} | {"enabled":"1"})
    write_csv(args.accepted_output, accepted, ["case_name","pdb_id","pdb_file","partner1_chains","partner2_chains","source","split","enabled"])
    print(f"\nAccepted cases: {len(accepted)}")


if __name__ == "__main__":
    main()
