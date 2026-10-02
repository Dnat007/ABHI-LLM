import json
from pathlib import Path
from collections import Counter


FILE = Path(
    r"D:\AbhiLLM\datasets\medical\processed\pubmedqa\train_chat.jsonl"
)


def main():

    print("=" * 70)
    print("PUBMEDQA DATASET VALIDATION")
    print("=" * 70)

    total = 0
    valid = 0
    invalid = 0

    decisions = Counter()
    duplicate_questions = set()
    duplicates = 0

    user_lengths = []
    assistant_lengths = []

    first_record = None

    with open(FILE, "r", encoding="utf-8") as f:

        for line_no, line in enumerate(f, start=1):

            if not line.strip():
                continue

            total += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                print(f"Invalid JSON at line {line_no}")
                continue

            messages = record.get("messages")

            if not isinstance(messages, list):
                invalid += 1
                continue

            if len(messages) != 3:
                invalid += 1
                continue

            roles = [
                message.get("role")
                for message in messages
            ]

            if roles != ["system", "user", "assistant"]:
                invalid += 1
                continue

            system_text = messages[0].get("content", "")
            user_text = messages[1].get("content", "")
            assistant_text = messages[2].get("content", "")

            if not system_text or not user_text or not assistant_text:
                invalid += 1
                continue

            # Duplicate question detection
            question_key = user_text.lower().strip()

            if question_key in duplicate_questions:
                duplicates += 1
            else:
                duplicate_questions.add(question_key)

            metadata = record.get("metadata", {})

            decision = metadata.get("final_decision")

            if decision not in {"yes", "no", "maybe"}:
                invalid += 1
                continue

            decisions[decision] += 1

            user_lengths.append(len(user_text))
            assistant_lengths.append(len(assistant_text))

            if first_record is None:
                first_record = record

            valid += 1

    print("\nResults")
    print("-" * 70)

    print(f"Total records       : {total}")
    print(f"Valid records       : {valid}")
    print(f"Invalid records     : {invalid}")
    print(f"Duplicate questions : {duplicates}")

    if total:
        print(f"Validation rate     : {valid / total * 100:.2f}%")

    print("\nFinal decision distribution:")

    for decision in ["yes", "no", "maybe"]:
        count = decisions[decision]

        percentage = (
            count / valid * 100
            if valid
            else 0
        )

        print(
            f"  {decision:5} : "
            f"{count:4} "
            f"({percentage:.2f}%)"
        )

    if user_lengths:

        print("\nUser prompt length:")
        print(f"  Min : {min(user_lengths)}")
        print(f"  Max : {max(user_lengths)}")
        print(f"  Avg : {sum(user_lengths) / len(user_lengths):.2f}")

    if assistant_lengths:

        print("\nAssistant response length:")
        print(f"  Min : {min(assistant_lengths)}")
        print(f"  Max : {max(assistant_lengths)}")
        print(
            f"  Avg : "
            f"{sum(assistant_lengths) / len(assistant_lengths):.2f}"
        )

    if first_record:

        print("\n" + "=" * 70)
        print("FIRST RECORD")
        print("=" * 70)

        print(
            json.dumps(
                first_record,
                indent=2,
                ensure_ascii=False
            )
        )

    print("\n" + "=" * 70)

    if invalid == 0 and duplicates == 0:
        print("PUBMEDQA VALIDATION PASSED")
    else:
        print("PUBMEDQA VALIDATION COMPLETED — REVIEW ABOVE")

    print("=" * 70)


if __name__ == "__main__":
    main()