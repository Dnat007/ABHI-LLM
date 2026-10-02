import json
from pathlib import Path
from collections import Counter

RAW_DIR = Path(r"D:\AbhiLLM\datasets\medical\raw\medmcqa")

FILES = [
    "train.jsonl",
    "validation.jsonl",
    "test.jsonl",
]

for filename in FILES:
    path = RAW_DIR / filename

    print("\n" + "=" * 80)
    print(f"FILE: {filename}")
    print("=" * 80)

    choice_types = Counter()
    cop_types = Counter()
    multi_examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            row = json.loads(line)

            choice_type = row.get("choice_type")
            cop = row.get("cop")

            choice_types[str(choice_type)] += 1
            cop_types[type(cop).__name__] += 1

            if choice_type == "multi" and len(multi_examples) < 20:
                multi_examples.append({
                    "line": line_no,
                    "question": row.get("question"),
                    "opa": row.get("opa"),
                    "opb": row.get("opb"),
                    "opc": row.get("opc"),
                    "opd": row.get("opd"),
                    "cop": cop,
                    "cop_type": type(cop).__name__,
                    "exp": row.get("exp"),
                    "subject": row.get("subject_name"),
                    "topic": row.get("topic_name"),
                })

    print("\nChoice types:")
    for key, value in choice_types.items():
        print(f"  {key}: {value}")

    print("\nCOP Python types:")
    for key, value in cop_types.items():
        print(f"  {key}: {value}")

    if multi_examples:
        print("\n" + "-" * 80)
        print("MULTI-CHOICE EXAMPLES")
        print("-" * 80)

        for i, example in enumerate(multi_examples, start=1):
            print(f"\n### Example {i}")
            print(f"Line: {example['line']}")
            print(f"Question: {example['question']}")

            print("\nOptions:")
            print(f"A: {example['opa']}")
            print(f"B: {example['opb']}")
            print(f"C: {example['opc']}")
            print(f"D: {example['opd']}")

            print(f"\nCOP: {example['cop']}")
            print(f"COP type: {example['cop_type']}")
            print(f"Explanation: {example['exp']}")
            print(f"Subject: {example['subject']}")
            print(f"Topic: {example['topic']}")

    else:
        print("\nNo multi-choice examples found.")