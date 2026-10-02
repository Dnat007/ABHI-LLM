import json
import random
from pathlib import Path
from collections import Counter


BASE_DIR = Path(r"D:\AbhiLLM")

MEDMCQA_FILE = (
    BASE_DIR
    / "datasets"
    / "medical"
    / "processed"
    / "medmcqa_clean"
    / "train.jsonl"
)

PUBMEDQA_FILE = (
    BASE_DIR
    / "datasets"
    / "medical"
    / "processed"
    / "pubmedqa"
    / "train_chat.jsonl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "datasets"
    / "medical"
    / "processed"
    / "combined"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "train.jsonl"
STATS_FILE = OUTPUT_DIR / "dataset_stats.json"

MEDMCQA_SAMPLES = 50_000
RANDOM_SEED = 42


def load_jsonl(path):

    records = []

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            if line.strip():
                records.append(json.loads(line))

    return records


def convert_medmcqa(row):

    question = row["question"]
    options = row["options"]

    user_content = (
        f"Question:\n{question}\n\n"
        "Options:\n"
        f"A: {options['A']}\n"
        f"B: {options['B']}\n"
        f"C: {options['C']}\n"
        f"D: {options['D']}"
    )

    answer = row["correct_answer"]

    explanation = row.get("explanation", "")

    if explanation:
        assistant_content = (
            f"The correct answer is: {answer}\n\n"
            f"Explanation:\n{explanation}"
        )
    else:
        assistant_content = (
            f"The correct answer is: {answer}"
        )

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are AbhiLLM, a medical-focused AI assistant. "
                    "Provide accurate, evidence-grounded medical answers."
                )
            },
            {
                "role": "user",
                "content": user_content
            },
            {
                "role": "assistant",
                "content": assistant_content
            }
        ],
        "metadata": {
            "source": "MedMCQA",
            "original_id": row.get("id"),
            "subject": row.get("subject"),
            "topic": row.get("topic"),
            "choice_type": row.get("choice_type")
        }
    }


def main():

    print("=" * 70)
    print("COMBINING MEDICAL DATASETS")
    print("=" * 70)

    random.seed(RANDOM_SEED)

    print("\nLoading MedMCQA...")
    medmcqa = load_jsonl(MEDMCQA_FILE)

    print(f"MedMCQA available: {len(medmcqa)}")

    print("\nLoading PubMedQA...")
    pubmedqa = load_jsonl(PUBMEDQA_FILE)

    print(f"PubMedQA available: {len(pubmedqa)}")

    # --------------------------------------------------
    # Sample MedMCQA
    # --------------------------------------------------

    if len(medmcqa) < MEDMCQA_SAMPLES:

        raise ValueError(
            f"MedMCQA contains only {len(medmcqa)} records."
        )

    selected_medmcqa = random.sample(
        medmcqa,
        MEDMCQA_SAMPLES
    )

    # Convert MedMCQA to common chat format
    selected_medmcqa = [
        convert_medmcqa(row)
        for row in selected_medmcqa
    ]

    # --------------------------------------------------
    # PubMedQA already has chat format
    # --------------------------------------------------

    selected_pubmedqa = pubmedqa

    combined = (
        selected_medmcqa
        + selected_pubmedqa
    )

    # Shuffle so datasets are mixed during training
    random.shuffle(combined)

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        for record in combined:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    # --------------------------------------------------
    # Statistics
    # --------------------------------------------------

    source_counts = Counter()

    subject_counts = Counter()

    decision_counts = Counter()

    for record in combined:

        metadata = record.get("metadata", {})

        source = metadata.get("source", "Unknown")

        source_counts[source] += 1

        if source == "MedMCQA":

            subject = metadata.get(
                "subject",
                "Unknown"
            )

            subject_counts[subject] += 1

        elif source == "PubMedQA":

            decision = metadata.get(
                "final_decision",
                "Unknown"
            )

            decision_counts[decision] += 1

    stats = {
        "random_seed": RANDOM_SEED,
        "max_sequence_length": 1024,

        "total_records": len(combined),

        "source_distribution": dict(
            source_counts
        ),

        "medmcqa_subject_distribution": dict(
            subject_counts
        ),

        "pubmedqa_decision_distribution": dict(
            decision_counts
        ),

        "files": {
            "medmcqa_source": str(MEDMCQA_FILE),
            "pubmedqa_source": str(PUBMEDQA_FILE),
            "combined_output": str(OUTPUT_FILE)
        }
    }

    with open(STATS_FILE, "w", encoding="utf-8") as f:

        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("COMBINATION COMPLETE")
    print("=" * 70)

    print(f"\nMedMCQA selected : {len(selected_medmcqa)}")
    print(f"PubMedQA selected: {len(selected_pubmedqa)}")
    print(f"Total            : {len(combined)}")

    print("\nSource distribution:")

    for source, count in source_counts.items():

        print(
            f"  {source:<10}: "
            f"{count:>6} "
            f"({count / len(combined) * 100:.2f}%)"
        )

    print("\nPubMedQA decisions:")

    for decision, count in decision_counts.items():

        print(
            f"  {decision:<8}: "
            f"{count:>5}"
        )

    print(f"\nOutput:")
    print(OUTPUT_FILE)

    print("\nStatistics:")
    print(STATS_FILE)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()