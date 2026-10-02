import json
import random
from pathlib import Path
from collections import Counter


BASE_DIR = Path(r"D:\AbhiLLM")

INPUT_FILE = (
    BASE_DIR
    / "datasets"
    / "medical"
    / "tokenized"
    / "train_tokenized.jsonl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "datasets"
    / "medical"
    / "tokenized"
)

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALIDATION_FILE = OUTPUT_DIR / "validation.jsonl"

SEED = 42
VALIDATION_RATIO = 0.05


def load_data():

    records = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                records.append(json.loads(line))

    return records


def main():

    print("=" * 70)
    print("ABHILLM MEDICAL TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    print("\nLoading dataset...")

    records = load_data()

    print(f"Total records: {len(records)}")

    # Separate by source so both datasets are represented
    medmcqa = [
        r for r in records
        if r["metadata"].get("source") == "MedMCQA"
    ]

    pubmedqa = [
        r for r in records
        if r["metadata"].get("source") == "PubMedQA"
    ]

    print(f"MedMCQA      : {len(medmcqa)}")
    print(f"PubMedQA     : {len(pubmedqa)}")

    rng = random.Random(SEED)

    rng.shuffle(medmcqa)
    rng.shuffle(pubmedqa)

    # 5% validation from each source
    med_val_count = round(len(medmcqa) * VALIDATION_RATIO)
    pub_val_count = round(len(pubmedqa) * VALIDATION_RATIO)

    med_validation = medmcqa[:med_val_count]
    med_train = medmcqa[med_val_count:]

    pub_validation = pubmedqa[:pub_val_count]
    pub_train = pubmedqa[pub_val_count:]

    train_records = med_train + pub_train
    validation_records = med_validation + pub_validation

    # Shuffle final datasets
    rng.shuffle(train_records)
    rng.shuffle(validation_records)

    print("\n" + "-" * 70)
    print("SPLIT RESULTS")
    print("-" * 70)

    print(f"\nTraining records   : {len(train_records)}")
    print(f"Validation records: {len(validation_records)}")

    print("\nTraining distribution:")

    train_sources = Counter(
        r["metadata"]["source"]
        for r in train_records
    )

    for source, count in train_sources.items():
        print(
            f"  {source:<12}: "
            f"{count:>6} "
            f"({count / len(train_records) * 100:.2f}%)"
        )

    print("\nValidation distribution:")

    val_sources = Counter(
        r["metadata"]["source"]
        for r in validation_records
    )

    for source, count in val_sources.items():
        print(
            f"  {source:<12}: "
            f"{count:>6} "
            f"({count / len(validation_records) * 100:.2f}%)"
        )

    # Save
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:

        for record in train_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    with open(VALIDATION_FILE, "w", encoding="utf-8") as f:

        for record in validation_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    print("\n" + "-" * 70)
    print("OUTPUT")
    print("-" * 70)

    print(f"\nTraining:")
    print(TRAIN_FILE)

    print(f"\nValidation:")
    print(VALIDATION_FILE)

    print("\n" + "=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()