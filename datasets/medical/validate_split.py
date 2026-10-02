import json
from pathlib import Path


BASE_DIR = Path(r"D:\AbhiLLM")

TRAIN_FILE = (
    BASE_DIR
    / "datasets"
    / "medical"
    / "tokenized"
    / "train.jsonl"
)

VAL_FILE = (
    BASE_DIR
    / "datasets"
    / "medical"
    / "tokenized"
    / "validation.jsonl"
)


def load_jsonl(path):
    records = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    return records


def get_id(record):
    metadata = record.get("metadata", {})
    return (
        metadata.get("source"),
        metadata.get("original_id"),
        metadata.get("pubid"),
    )


def main():

    print("=" * 70)
    print("ABHILLM TRAIN / VALIDATION DATASET VALIDATION")
    print("=" * 70)

    train = load_jsonl(TRAIN_FILE)
    validation = load_jsonl(VAL_FILE)

    print(f"\nTrain records      : {len(train)}")
    print(f"Validation records : {len(validation)}")
    print(f"Total               : {len(train) + len(validation)}")

    # ---------------------------------------------------------
    # Basic structure
    # ---------------------------------------------------------

    invalid_train = 0
    invalid_val = 0

    for record in train:

        if (
            "input_ids" not in record
            or "attention_mask" not in record
            or "metadata" not in record
        ):
            invalid_train += 1

    for record in validation:

        if (
            "input_ids" not in record
            or "attention_mask" not in record
            or "metadata" not in record
        ):
            invalid_val += 1

    print("\nStructural validation:")
    print(f"  Invalid train      : {invalid_train}")
    print(f"  Invalid validation : {invalid_val}")

    # ---------------------------------------------------------
    # Source distribution
    # ---------------------------------------------------------

    def source_counts(records):

        counts = {}

        for record in records:

            source = record["metadata"].get("source")

            counts[source] = counts.get(source, 0) + 1

        return counts

    train_sources = source_counts(train)
    val_sources = source_counts(validation)

    print("\nTrain source distribution:")

    for source, count in train_sources.items():
        print(f"  {source:<12}: {count}")

    print("\nValidation source distribution:")

    for source, count in val_sources.items():
        print(f"  {source:<12}: {count}")

    # ---------------------------------------------------------
    # Duplicate / leakage check
    # ---------------------------------------------------------

    train_ids = {get_id(r) for r in train}
    val_ids = {get_id(r) for r in validation}

    overlap = train_ids.intersection(val_ids)

    print("\nData leakage check:")
    print(f"  Train unique IDs      : {len(train_ids)}")
    print(f"  Validation unique IDs : {len(val_ids)}")
    print(f"  Train/Val overlap     : {len(overlap)}")

    # ---------------------------------------------------------
    # Token statistics
    # ---------------------------------------------------------

    train_lengths = [
        len(r["input_ids"])
        for r in train
    ]

    val_lengths = [
        len(r["input_ids"])
        for r in validation
    ]

    print("\nToken statistics:")

    print(
        f"  Train min/max         : "
        f"{min(train_lengths)} / {max(train_lengths)}"
    )

    print(
        f"  Validation min/max    : "
        f"{min(val_lengths)} / {max(val_lengths)}"
    )

    print(
        f"  Train average         : "
        f"{sum(train_lengths) / len(train_lengths):.2f}"
    )

    print(
        f"  Validation average    : "
        f"{sum(val_lengths) / len(val_lengths):.2f}"
    )

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------

    expected_total = 51000
    expected_train = 48450
    expected_validation = 2550

    passed = True

    if len(train) != expected_train:
        passed = False

    if len(validation) != expected_validation:
        passed = False

    if len(train) + len(validation) != expected_total:
        passed = False

    if invalid_train != 0 or invalid_val != 0:
        passed = False

    if len(overlap) != 0:
        passed = False

    print("\n" + "=" * 70)

    if passed:
        print("DATASET SPLIT VALIDATION PASSED")
    else:
        print("DATASET SPLIT VALIDATION FAILED")

    print("=" * 70)


if __name__ == "__main__":
    main()