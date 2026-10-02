import json
from pathlib import Path


RAW_DIR = Path(
    r"D:\AbhiLLM\datasets\medical\raw\pubmedqa"
)

OUTPUT_DIR = Path(
    r"D:\AbhiLLM\datasets\medical\processed\pubmedqa"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SYSTEM_PROMPT = (
    "You are AbhiLLM, a medical-focused AI assistant. "
    "Analyze biomedical evidence carefully, distinguish evidence from uncertainty, "
    "and provide accurate, evidence-grounded answers."
)


def clean_text(text):
    if text is None:
        return ""

    text = str(text)
    return " ".join(text.split()).strip()


def extract_context(context):
    """
    PubMedQA context is a dictionary.
    We use only the actual research passages.
    """

    if not isinstance(context, dict):
        return ""

    contexts = context.get("contexts", [])

    if not isinstance(contexts, list):
        return ""

    cleaned = []

    for item in contexts:
        item = clean_text(item)

        if item:
            cleaned.append(item)

    return "\n\n".join(cleaned)


def convert_record(row):

    question = clean_text(row.get("question"))
    context = extract_context(row.get("context"))
    long_answer = clean_text(row.get("long_answer"))
    final_decision = clean_text(row.get("final_decision")).lower()

    if not question:
        return None

    if not context:
        return None

    if not long_answer:
        return None

    if final_decision not in {"yes", "no", "maybe"}:
        return None

    user_content = (
        "Question:\n"
        f"{question}\n\n"
        "Scientific context:\n"
        f"{context}\n\n"
        "Based on the provided evidence, determine the answer "
        "and explain the reasoning."
    )

    assistant_content = (
        f"Final decision: {final_decision.upper()}\n\n"
        f"Evidence-based explanation:\n{long_answer}"
    )

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
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
            "source": "PubMedQA",
            "pubid": row.get("pubid"),
            "final_decision": final_decision
        }
    }


def process_file(filename):

    input_path = RAW_DIR / filename
    output_path = OUTPUT_DIR / filename.replace(
        ".jsonl",
        "_chat.jsonl"
    )

    total = 0
    converted = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:

            if not line.strip():
                continue

            total += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            converted_record = convert_record(row)

            if converted_record is None:
                skipped += 1
                continue

            fout.write(
                json.dumps(
                    converted_record,
                    ensure_ascii=False
                ) + "\n"
            )

            converted += 1

    print("\n" + "=" * 70)
    print(f"FILE: {filename}")
    print("=" * 70)

    print(f"Total records : {total}")
    print(f"Converted     : {converted}")
    print(f"Skipped       : {skipped}")

    if total:
        print(f"Retention     : {converted / total * 100:.2f}%")

    print(f"Output        : {output_path}")

    return converted


def main():

    print("=" * 70)
    print("PUBMEDQA → ABHILLM CHAT FORMAT")
    print("=" * 70)

    total_converted = 0

    for filename in ["train.jsonl"]:
        total_converted += process_file(filename)

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(f"Total converted: {total_converted}")


if __name__ == "__main__":
    main()