from pathlib import Path
import json

RAW_DIR = Path(r"D:\AbhiLLM\datasets\medical\raw\medmcqa")

OUTPUT_DIR = Path(r"D:\AbhiLLM\datasets\medical\processed\medmcqa")

OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

SYSTEM_PROMPT = """You are AbhiLLM, a medical-focused AI assistant.
Answer medical questions accurately and clearly.
Use appropriate medical terminology.
When an explanation is available, explain why the answer is correct.
Do not invent medical facts.
Distinguish established information from uncertainty.
"""

OPTION_KEYS = ["opa", "opb", "opc", "opd"]

def clean_text(value):
    if value is None:
        return ""
    return str(value).strip()

# here i am finding the correct option
def get_correct_answer(row):
    try:
        correct_index = int(row["cop"])
    except (ValueError, TypeError):
        return None

    if correct_index < 0 or correct_index >= len(OPTION_KEYS):
        return None

    option_key = OPTION_KEYS[correct_index]
    answer = clean_text(row.get(option_key))
    if not answer:
        return None
    return answer

# perform the clean conversion A/B/C/D
def build_user_prompt(row):
    question = clean_text(row.get("question"))

    options = []

    for index, key in enumerate(OPTION_KEYS):
        option = clean_text(row.get(key))

        if option:
            letter = chr(ord("A") + index)
            options.append(f"{letter}. {option}")

    option_text = "\n".join(options)

    return (
        f"{question}\n\n"
        f"Options:\n"
        f"{option_text}"
    )

# correct answer to the model 
def build_assistant_answer(row):
    correct_answer = get_correct_answer(row)

    if correct_answer is None:
        return None

    explanation = clean_text(row.get("exp"))

    if explanation:
        return (f"The correct answer is: {correct_answer}\n\n"
            f"Explanation: {explanation}")

    return (f"The correct answer is: {correct_answer}")

def convert_record(row):
    question = clean_text(row.get("question"))

    if not question:
        return None
    answer = build_assistant_answer(row)
    if answer is None:
        return None
    user_prompt = build_user_prompt(row)

    metadata = {
        "source": "MedMCQA",
        "original_id": row.get("id"),
        "subject": row.get("subject_name"),
        "topic": row.get("topic_name"),
        "choice_type": row.get("choice_type"),
    }

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
        "metadata": metadata,
    }

def convert_split(input_file, output_file):
    total = 0
    converted = 0
    skipped = 0
    with input_file.open("r",encoding="utf-8") as source, output_file.open("w",encoding="utf-8") as destination:
        for line in source:
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            record = convert_record(row)
            if record is None:
                skipped += 1
                continue

            destination.write(json.dumps(record,ensure_ascii=False)+ "\n")

            converted += 1
    return total, converted, skipped

def main():
    files = sorted(RAW_DIR.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {RAW_DIR}")

    for input_file in files:
        output_file = (OUTPUT_DIR/ f"{input_file.stem}_chat.jsonl")
        total, converted, skipped = convert_split(input_file,output_file)
        print(f"  Total records : {total}")
        print(f"  Converted     : {converted}")
        print(f"  Skipped       : {skipped}")
        print(f"  Output        : {output_file}")

if __name__ == "__main__":
    main()