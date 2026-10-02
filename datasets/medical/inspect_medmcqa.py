from pathlib import Path
import json
from collections import Counter

DATA_DIR = Path(r"D:\AbhiLLM\datasets\medical\raw\medmcqa")

def inspect_file(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    print(f"Records: {len(rows)}")

    if not rows:
        print("EMPTY FILE")
        return

    print("\nColumns:")
    for key in rows[0].keys():
        print(f"  - {key}")

    print("\nFirst record:")
    print(json.dumps(rows[0], indent=2, ensure_ascii=False))
    print("\nMissing values:")

    for key in rows[0].keys():
        missing = sum(
            1 for row in rows
            if row.get(key) is None
            or str(row.get(key)).strip() == "")

        percentage = (missing / len(rows) * 100 if rows else 0 )

        print(
            f"  {key}: "
            f"{missing} "
            f"({percentage:.2f}%)"
        )

    if "subject_name" in rows[0]:

        subjects = Counter(
            row.get("subject_name")
            for row in rows
        )

        print("\nTop subjects:")

        for subject, count in subjects.most_common(20):
            print(f"  {subject}: {count}")

    if "topic_name" in rows[0]:

        topics = Counter(
            row.get("topic_name")
            for row in rows
        )

        print("\nTop topics:")

        for topic, count in topics.most_common(20):
            print(f"  {topic}: {count}")

if not DATA_DIR.exists():
    raise FileNotFoundError(f"Dataset directory not found:\n{DATA_DIR}")

files = sorted(DATA_DIR.glob("*.jsonl"))

if not files:
    raise FileNotFoundError(f"No JSONL files found in:\n{DATA_DIR}")

print(f"\nFound {len(files)} JSONL files.")

for file in files:
    inspect_file(file)
print("INSPECTION COMPLETE")
