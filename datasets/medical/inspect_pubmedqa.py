import json
from pathlib import Path
from collections import Counter

DATASET_DIR = Path(
    r"D:\AbhiLLM\datasets\medical\raw\pubmedqa"
)

FILE = DATASET_DIR / "train.jsonl"

print("=" * 70)
print("PUBMEDQA INSPECTION")
print("=" * 70)

records = []

with open(FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print(f"\nTotal records: {len(records)}")

print("\nFields:")
if records:
    for key, value in records[0].items():
        print(f"  {key}: {type(value).__name__}")

print("\nFinal decision distribution:")
decision_counter = Counter(
    str(row.get("final_decision"))
    for row in records
)

for decision, count in decision_counter.items():
    print(f"  {decision}: {count}")

print("\n" + "=" * 70)
print("FIRST 5 RECORDS")
print("=" * 70)

for i, row in enumerate(records[:5], start=1):

    print(f"\n### RECORD {i}")

    print(f"\nPubID:")
    print(row.get("pubid"))

    print(f"\nQuestion:")
    print(row.get("question"))

    print(f"\nContext:")
    context = row.get("context")
    print(context)

    print(f"\nLong Answer:")
    print(row.get("long_answer"))

    print(f"\nFinal Decision:")
    print(row.get("final_decision"))

print("\n" + "=" * 70)
print("PUBMEDQA INSPECTION COMPLETE")
print("=" * 70)