import os
import csv
import shutil
import random
import argparse
from pathlib import Path


def safe_case_name(name):
    safe = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)
    return safe.strip("_")


def infer_entry_name_from_reference(ref_path, root):
    """
    Infer BM5 entry name from layouts such as:
      - .../<ENTRY>/<ENTRY>_reference.pdb
      - .../<ENTRY>/ana_scripts/target.pdb
      - .../<ENTRY>/ana_scripts_2/target.pdb
      - .../<ENTRY>/ana_scripts_3/target.pdb

    BM5-clean commonly stores reference targets under ana_scripts_2/target.pdb
    or ana_scripts_3/target.pdb, so we must use the grandparent folder as
    the actual complex entry name.
    """
    ref_path = Path(ref_path)
    root = Path(root)

    stem = ref_path.stem

    if stem.endswith("_reference"):
        return stem.replace("_reference", "")

    if ref_path.name == "target.pdb":
        parent_name = ref_path.parent.name.lower()

        if parent_name.startswith("ana_scripts"):
            return ref_path.parent.parent.name

        return ref_path.parent.name

    if stem.endswith("_ref"):
        return stem.replace("_ref", "")

    # Fallback: first directory below root.
    try:
        rel = ref_path.relative_to(root)
        if len(rel.parts) > 1:
            return rel.parts[0]
    except ValueError:
        pass

    return stem


def target_priority(path):
    """
    Prefer:
      1. *_reference.pdb
      2. ana_scripts/target.pdb
      3. ana_scripts_2/target.pdb
      4. ana_scripts_3/target.pdb
      5. *_ref.pdb
    """
    path = Path(path)
    name = path.name
    parent = path.parent.name.lower()

    if name.endswith("_reference.pdb"):
        return 0

    if name == "target.pdb":
        if parent == "ana_scripts":
            return 1

        if parent.startswith("ana_scripts_"):
            suffix = parent.replace("ana_scripts_", "")

            try:
                return 1 + int(suffix)
            except ValueError:
                return 10

        return 10

    if name.endswith("_ref.pdb"):
        return 20

    return 30


def find_reference_pdbs(haddock_ready_dir):
    """
    Search recursively because BM5-clean layouts can differ between versions
    or local checkouts.
    """
    root = Path(haddock_ready_dir)

    if not root.exists():
        raise FileNotFoundError(f"HADDOCK-ready directory not found: {root}")

    candidates = []

    for path in sorted(root.rglob("*_reference.pdb")):
        candidates.append(path)

    for path in sorted(root.rglob("target.pdb")):
        candidates.append(path)

    for path in sorted(root.rglob("*_ref.pdb")):
        candidates.append(path)

    by_entry = {}

    for path in candidates:
        entry = infer_entry_name_from_reference(path, root)
        entry = safe_case_name(entry)

        current = by_entry.get(entry)

        if current is None:
            by_entry[entry] = path
            continue

        if target_priority(path) < target_priority(current):
            by_entry[entry] = path

    return sorted(by_entry.items(), key=lambda x: x[0])


def assign_splits(entries, train_frac, val_frac, seed):
    rng = random.Random(seed)
    shuffled = list(entries)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))

    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))

    train_entries = set(shuffled[:n_train])
    val_entries = set(shuffled[n_train:n_train + n_val])

    split_map = {}

    for entry in entries:
        if entry in train_entries:
            split_map[entry] = "train"
        elif entry in val_entries:
            split_map[entry] = "val"
        else:
            split_map[entry] = "test"

    return split_map


def write_cases_csv(path, rows):
    fieldnames = [
        "case_name",
        "pdb_id",
        "pdb_file",
        "partner1_chains",
        "partner2_chains",
        "source",
        "split",
        "enabled",
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import BM5-clean reference complexes into the project case CSV format."
    )

    parser.add_argument(
        "--bm5_haddock_ready_dir",
        required=True,
        help="Path to BM5-clean/HADDOCK-ready directory.",
    )

    parser.add_argument(
        "--out_pdb_dir",
        default=os.path.join("data", "raw_pdb_expanded_bm5"),
        help="Directory where copied reference complex PDB files will be stored.",
    )

    parser.add_argument(
        "--out_cases",
        default=os.path.join("data", "cases", "bm5_reference_cases.csv"),
        help="Output cases CSV.",
    )

    parser.add_argument("--max_cases", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)

    parser.add_argument(
        "--prefix",
        default="BM5",
        help="Prefix added to generated case names.",
    )

    parser.add_argument(
        "--partner1_chains",
        default="A",
        help="Partner 1 chains in copied BM5 reference complex files.",
    )

    parser.add_argument(
        "--partner2_chains",
        default="B",
        help="Partner 2 chains in copied BM5 reference complex files.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    refs = find_reference_pdbs(args.bm5_haddock_ready_dir)

    print(f"Found candidate reference complexes: {len(refs)}")

    if args.max_cases and args.max_cases > 0:
        refs = refs[:args.max_cases]
        print(f"Using first max_cases={args.max_cases}")

    entries = [entry for entry, _ in refs]

    if not entries:
        print("\nNo reference PDB files found.")
        print("Debug commands to run in PowerShell:")
        print(
            f'Get-ChildItem "{args.bm5_haddock_ready_dir}" -Recurse -Filter "*reference*.pdb" '
            "| Select-Object -First 10 FullName"
        )
        print(
            f'Get-ChildItem "{args.bm5_haddock_ready_dir}" -Recurse -Filter "target.pdb" '
            "| Select-Object -First 10 FullName"
        )
        return

    split_map = assign_splits(
        entries=entries,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    os.makedirs(args.out_pdb_dir, exist_ok=True)

    rows = []

    for entry, ref_path in refs:
        case_name = safe_case_name(f"{args.prefix}_{entry}_A_B")
        out_pdb = Path(args.out_pdb_dir) / f"{case_name}.pdb"

        shutil.copyfile(ref_path, out_pdb)

        rows.append({
            "case_name": case_name,
            "pdb_id": entry,
            "pdb_file": str(out_pdb).replace("\\", "/"),
            "partner1_chains": args.partner1_chains,
            "partner2_chains": args.partner2_chains,
            "source": "BM5-clean",
            "split": split_map[entry],
            "enabled": "1",
        })

    write_cases_csv(args.out_cases, rows)

    counts = {"train": 0, "val": 0, "test": 0}
    for row in rows:
        counts[row["split"]] += 1

    print("\n================ BM5 IMPORT SUMMARY ================")
    print(f"Imported reference complexes: {len(rows)}")
    print(f"Output PDB directory: {args.out_pdb_dir}")
    print(f"Cases CSV: {args.out_cases}")
    print(f"Train cases: {counts['train']}")
    print(f"Validation cases: {counts['val']}")
    print(f"Test cases: {counts['test']}")
    print(f"Partner chains: {args.partner1_chains} vs {args.partner2_chains}")
    print("\nFirst imported examples:")

    for row in rows[:8]:
        print(
            f"- {row['case_name']} | split={row['split']} | "
            f"file={row['pdb_file']}"
        )

    print("====================================================")


if __name__ == "__main__":
    main()
