from pathlib import Path
import json
from collections import Counter

DATA_DIR = Path(r"D:\AbhiLLM\datasets\medical\processed\medmcqa")

def validate_file(path):
    print("\n" + "=" * 70)
    print(f"VALIDATING: {path.name}")
    print("=" * 70)

    total = 0
    valid = 0
    invalid = 0

    duplicate_questions = 0
    questions = set()

    subjects = Counter()

    answer_lengths = []
    prompt_lengths = []

    first_record = None

    with path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(f"Invalid JSON at line {line_number}")
                invalid += 1
                continue
            
            if "messages" not in record:
                invalid += 1
                continue

            messages = record["messages"]
            if not isinstance(messages, list):
                invalid += 1
                continue

            if len(messages) != 3:
                invalid += 1
                continue

            roles = [message.get("role") for message in messages]

            if roles != [
                "system",
                "user",
                "assistant"
            ]:
                invalid += 1
                continue

            if any(
                not isinstance(message.get("content"), str)
                or not message.get("content").strip()
                for message in messages
            ):
                invalid += 1
                continue

            user_content = messages[1]["content"]
            assistant_content = messages[2]["content"]

            # find duplicate
            question_key = user_content.strip().lower()

            if question_key in questions:
                duplicate_questions += 1
            else:
                questions.add(question_key)

            metadata = record.get("metadata", {})

            subject = metadata.get("subject","UNKNOWN")

            subjects[str(subject)] += 1

            prompt_lengths.append(len(user_content))

            answer_lengths.append(len(assistant_content))

            if first_record is None:
                first_record = record

            valid += 1

    print(f"\nTotal records       : {total}")
    print(f"Valid records       : {valid}")
    print(f"Invalid records     : {invalid}")
    print(f"Duplicate questions : {duplicate_questions}")

    if prompt_lengths:

        print(f"\nUser prompt length:")
        print(f"  Minimum : {min(prompt_lengths)} chars")
        print(f"  Maximum : {max(prompt_lengths)} chars")
        print(f"  Average : "f"{sum(prompt_lengths) / len(prompt_lengths):.2f} chars")

    if answer_lengths:

        print(f"\nAssistant answer length:")
        print(f"  Minimum : {min(answer_lengths)} chars")
        print(f"  Maximum : {max(answer_lengths)} chars")
        print(f"  Average : "f"{sum(answer_lengths) / len(answer_lengths):.2f} chars")

    print("\nTop medical subjects:")

    for subject, count in subjects.most_common(25):
        print(f"  {subject}: {count}")

    if first_record:

        print("\n" + "-" * 70)
        print("FIRST CONVERTED RECORD")
        print("-" * 70)

        print(json.dumps(first_record,indent=2,ensure_ascii=False))

    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "duplicates": duplicate_questions,
    }


def main():

    print("ABHILLM - MEDICAL DATASET VALIDATION")
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found:\n{DATA_DIR}")

    files = sorted(DATA_DIR.glob("*_chat.jsonl"))
    if not files:
        raise FileNotFoundError(f"No converted datasets found in:\n{DATA_DIR}")

    results = {}

    for file in files:
        results[file.name] = validate_file(file)

    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)

    for filename, result in results.items():
        print(f"\n{filename}")
        print(f"  Total     : {result['total']}")
        print(f"  Valid     : {result['valid']}")
        print(f"  Invalid   : {result['invalid']}")
        print(f"  Duplicates: {result['duplicates']}")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()