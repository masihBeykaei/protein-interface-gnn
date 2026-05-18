import csv
from pathlib import Path


INPUT_FILES = [
    "data/cases/expanded_cases.csv",
    "data/cases/bm5_cases_accepted.csv",
]

OUTPUT_FILE = "data/cases/combined_current_bm5_cases.csv"

FIELDNAMES = [
    "case_name",
    "pdb_id",
    "pdb_file",
    "partner1_chains",
    "partner2_chains",
    "source",
    "split",
    "enabled",
]


def normalize_row(row):
    normalized = {}

    for field in FIELDNAMES:
        normalized[field] = row.get(field, "").strip()

    if not normalized["enabled"]:
        normalized["enabled"] = "1"

    return normalized


def main():
    rows = []
    seen = set()

    for input_file in INPUT_FILES:
        input_path = Path(input_file)

        if not input_path.exists():
            raise FileNotFoundError(f"Missing input file: {input_file}")

        with input_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                normalized = normalize_row(row)
                case_name = normalized["case_name"]

                if not case_name:
                    continue

                if case_name in seen:
                    continue

                seen.add(case_name)
                rows.append(normalized)

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    split_counts = {
        "train": 0,
        "val": 0,
        "test": 0,
        "other": 0,
    }

    source_counts = {}

    for row in rows:
        split = row.get("split", "").strip().lower()

        if split in {"train", "val", "test"}:
            split_counts[split] += 1
        else:
            split_counts["other"] += 1

        source = row.get("source", "").strip() or "unknown"
        source_counts[source] = source_counts.get(source, 0) + 1

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Total cases: {len(rows)}")
    print("Split counts:", split_counts)
    print("Source counts:", source_counts)


if __name__ == "__main__":
    main()
