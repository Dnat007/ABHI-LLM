import json
import re
from pathlib import Path
from collections import Counter

RAW_DIR = Path(r"D:\AbhiLLM\datasets\medical\raw\medmcqa")
OUTPUT_DIR = Path(r"D:\AbhiLLM\datasets\medical\processed\medmcqa_clean")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = ["train.jsonl", "validation.jsonl"]

MAX_QUESTION_CHARS = 2500
MAX_EXPLANATION_CHARS = 12000


def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)

    # Remove excessive repeated punctuation
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"\-{4,}", "---", text)

    return text.strip()


def valid_option(value):
    value = clean_text(value)

    if not value:
        return False

    # Detect obvious placeholder options
    placeholders = {
        "none",
        "none of the above",
        "not recalled",
        "not known",
        "not applicable",
        "n/a",
    }

    return value.lower() not in placeholders


def process_file(filename):
    input_path = RAW_DIR / filename
    output_path = OUTPUT_DIR / filename

    total = 0
    kept = 0
    rejected = 0

    rejection_reasons = Counter()

    seen_questions = set()

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                rejected += 1
                rejection_reasons["invalid_json"] += 1
                continue

            question = clean_text(row.get("question"))

            options = [
                clean_text(row.get("opa")),
                clean_text(row.get("opb")),
                clean_text(row.get("opc")),
                clean_text(row.get("opd")),
            ]

            cop = row.get("cop")

            explanation = clean_text(row.get("exp"))
            
            # 1. Question validation
            if not question:
                rejected += 1
                rejection_reasons["empty_question"] += 1
                continue

            if len(question) > MAX_QUESTION_CHARS:
                rejected += 1
                rejection_reasons["question_too_long"] += 1
                continue

            # 2. Option validation
            if not all(valid_option(option) for option in options):
                rejected += 1
                rejection_reasons["invalid_options"] += 1
                continue

            # 3. Answer validation
            if not isinstance(cop, int) or not (0 <= cop <= 3):
                rejected += 1
                rejection_reasons["invalid_answer_index"] += 1
                continue
            
            # 4. Duplicate detection
            normalized_question = question.lower()

            if normalized_question in seen_questions:
                rejected += 1
                rejection_reasons["duplicate_question"] += 1
                continue

            seen_questions.add(normalized_question)
            
            # 5. Explanation handling
            if len(explanation) > MAX_EXPLANATION_CHARS:
                explanation = explanation[:MAX_EXPLANATION_CHARS].rstrip()
                
            # 6. Preserve metadata
            cleaned = {
                "id": row.get("id"),
                "question": question,
                "options": {
                    "A": options[0],
                    "B": options[1],
                    "C": options[2],
                    "D": options[3],
                },
                "correct_option": cop,
                "correct_answer": options[cop],
                "explanation": explanation,
                "choice_type": row.get("choice_type"),
                "subject": clean_text(row.get("subject_name")),
                "topic": clean_text(row.get("topic_name")),
                "source": "MedMCQA",
            }

            fout.write(json.dumps(cleaned,ensure_ascii=False) + "\n")

            kept += 1

    print("\n" + "=" * 70)
    print(f"{filename}")
    print("=" * 70)

    print(f"Total records : {total}")
    print(f"Kept records  : {kept}")
    print(f"Rejected      : {rejected}")

    if total:
        print(f"Retention     : {kept / total * 100:.2f}%")

    print("\nRejection reasons:")

    if rejection_reasons:
        for reason, count in rejection_reasons.most_common():
            print(f"  {reason}: {count}")
    else:
        print("  None")

    print(f"\nOutput: {output_path}")

    return kept


def main():

    print("=" * 70)
    print("MEDMCQA QUALITY FILTER")
    print("=" * 70)

    total_kept = 0

    for filename in FILES:
        total_kept += process_file(filename)

    print("\n" + "=" * 70)
    print("FILTER COMPLETE")
    print("=" * 70)
    print(f"Total retained records: {total_kept}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()