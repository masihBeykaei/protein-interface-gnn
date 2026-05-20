import os
import csv
import json
import argparse

import numpy as np
import torch
from Bio.PDB import PDBParser
from transformers import AutoTokenizer, AutoModel

RAW_PDB_DIRS = [
    os.path.join("data", "raw_pdb_expanded_bm5"),
    os.path.join("data", "raw_pdb_expanded"),
    os.path.join("data", "raw_pdb"),
]

AA3_TO_AA1 = {
    "ALA":"A", "ARG":"R", "ASN":"N", "ASP":"D", "CYS":"C",
    "GLN":"Q", "GLU":"E", "GLY":"G", "HIS":"H", "ILE":"I",
    "LEU":"L", "LYS":"K", "MET":"M", "PHE":"F", "PRO":"P",
    "SER":"S", "THR":"T", "TRP":"W", "TYR":"Y", "VAL":"V",
    "MSE":"M",
}


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
        for key in [pdb_id, case_name]:
            if key:
                candidates.extend([
                    os.path.join(raw_dir, f"{key}.pdb"),
                    os.path.join(raw_dir, f"{key.lower()}.pdb"),
                    os.path.join(raw_dir, f"{key.upper()}.pdb"),
                ])

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"No PDB file found for case={case_name}, pdb_id={pdb_id}")


def read_cases(cases_csv):
    rows = []
    with open(cases_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            enabled = row.get("enabled", "1").strip()
            if enabled in {"0", "false", "False", "no", "No"}:
                continue
            rows.append(row)
    return rows


def extract_partner_sequence(model, chain_ids):
    sequence = []
    residues = []

    for chain_id in chain_ids:
        if chain_id not in model:
            available = list(model.child_dict.keys())
            raise ValueError(f"Chain {chain_id} not found. Available chains: {available}")

        chain = model[chain_id]

        for res in chain:
            if res.get_id()[0] != " " and res.get_resname() not in AA3_TO_AA1:
                continue
            if not res.has_id("CA"):
                continue

            resname = res.get_resname()
            aa = AA3_TO_AA1.get(resname, "X")
            res_id = res.get_id()

            sequence.append(aa)
            residues.append({
                "chain_id": chain_id,
                "hetflag": str(res_id[0]),
                "resseq": int(res_id[1]),
                "icode": str(res_id[2]).strip(),
                "resname": resname,
                "aa": aa,
            })

    return "".join(sequence), residues


def embed_sequence(sequence, tokenizer, model, device, max_seq_len):
    if not sequence:
        raise ValueError("Cannot embed empty sequence")

    chunks = []

    for start in range(0, len(sequence), max_seq_len):
        chunk = sequence[start:start + max_seq_len]
        encoded = tokenizer(chunk, return_tensors="pt", add_special_tokens=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            output = model(**encoded)

        hidden = output.last_hidden_state[0]
        residue_embeddings = hidden[1:1 + len(chunk)].detach().cpu().numpy()

        if residue_embeddings.shape[0] != len(chunk):
            raise RuntimeError(
                f"Embedding length mismatch: got {residue_embeddings.shape[0]}, expected {len(chunk)}"
            )

        chunks.append(residue_embeddings.astype(np.float32))

    return np.concatenate(chunks, axis=0)


def get_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract per-residue ESM-2 embeddings for project cases.")
    parser.add_argument("--cases", default=os.path.join("data", "cases", "combined_current_bm5_cases.csv"))
    parser.add_argument("--out_dir", default=os.path.join("data", "esm2_embeddings"))
    parser.add_argument("--model_name", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--max_seq_len", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device(args.device)

    print("\nESM-2 embedding extraction")
    print(f"Cases CSV: {args.cases}")
    print(f"Output dir: {args.out_dir}")
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    rows = read_cases(args.cases)
    summary = []

    for idx, row in enumerate(rows, start=1):
        case_name = row["case_name"].strip()
        p1 = row["partner1_chains"].strip()
        p2 = row["partner2_chains"].strip()

        out_a = os.path.join(args.out_dir, f"{case_name}_partner1_esm2.npy")
        out_b = os.path.join(args.out_dir, f"{case_name}_partner2_esm2.npy")
        out_meta = os.path.join(args.out_dir, f"{case_name}_esm2_metadata.json")

        if not args.force and all(os.path.exists(p) for p in [out_a, out_b, out_meta]):
            print(f"[{idx}/{len(rows)}] Skipping {case_name}: embeddings already exist.")
            continue

        print(f"\n[{idx}/{len(rows)}] Processing {case_name} ({p1} vs {p2})")
        pdb_path = find_pdb_path(row)
        structure = load_structure(pdb_path)
        model0 = next(structure.get_models())

        seq1, residues1 = extract_partner_sequence(model0, p1)
        seq2, residues2 = extract_partner_sequence(model0, p2)

        print(f"Partner 1 length: {len(seq1)}")
        print(f"Partner 2 length: {len(seq2)}")

        emb1 = embed_sequence(seq1, tokenizer, model, device, args.max_seq_len)
        emb2 = embed_sequence(seq2, tokenizer, model, device, args.max_seq_len)

        if emb1.shape[0] != len(residues1) or emb2.shape[0] != len(residues2):
            raise RuntimeError(f"Embedding/residue length mismatch in {case_name}")

        np.save(out_a, emb1)
        np.save(out_b, emb2)

        metadata = {
            "case_name": case_name,
            "pdb_file": pdb_path,
            "model_name": args.model_name,
            "embedding_dim": int(emb1.shape[1]),
            "partner1_chains": p1,
            "partner2_chains": p2,
            "partner1_length": len(seq1),
            "partner2_length": len(seq2),
            "partner1_residues": residues1,
            "partner2_residues": residues2,
        }

        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved: {out_a}")
        print(f"Saved: {out_b}")
        print(f"Saved: {out_meta}")

        summary.append({
            "case_name": case_name,
            "partner1_length": len(seq1),
            "partner2_length": len(seq2),
            "embedding_dim": int(emb1.shape[1]),
        })

    summary_path = os.path.join(args.out_dir, "esm2_embedding_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_name", "partner1_length", "partner2_length", "embedding_dim"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"\nSaved: {summary_path}")
    print("ESM-2 embedding extraction completed successfully.")


if __name__ == "__main__":
    main()
