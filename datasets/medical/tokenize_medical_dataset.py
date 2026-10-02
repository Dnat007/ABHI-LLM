import json
from pathlib import Path
from transformers import AutoTokenizer

BASE_DIR = Path(r"D:\AbhiLLM")

INPUT_FILE = (
    BASE_DIR
    / "datasets"
    / "medical"
    / "processed"
    / "combined"
    / "train.jsonl"
)

OUTPUT_DIR = (
    BASE_DIR
    / "datasets"
    / "medical"
    / "tokenized"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "train_tokenized.jsonl"
MODEL_PATH = BASE_DIR / "models" / "Qwen3-4B"
MAX_LENGTH = 1024

def main():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH),local_files_only=True)
    total = 0
    processed = 0
    truncated = 0

    token_lengths = []

    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            messages = record["messages"]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            tokens = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=MAX_LENGTH,
            )

            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]

            original_tokens = tokenizer(
                text,
                add_special_tokens=False,
            )["input_ids"]

            original_length = len(original_tokens)
            final_length = len(input_ids)
            if original_length > MAX_LENGTH:
                truncated += 1

            token_lengths.append(final_length)
            output_record = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "metadata": record.get("metadata", {}),
                "original_token_length": original_length,
            }

            fout.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False,
                ) + "\n"
            )

            processed += 1
            if processed % 5000 == 0:
                print(f"{processed}/{total}")

    print("TOKENIZATION COMPLETE")
    print(f"\nTotal records       : {total}")
    print(f"Processed records   : {processed}")
    print(f"Truncated records   : {truncated}")

    if total:
        print(f"{truncated / total * 100:.2f}%")

    if token_lengths:
        print("\nFinal token length:")
        print(f"  Minimum : {min(token_lengths)}")
        print(f"  Maximum : {max(token_lengths)}")
        print(f"  Average : "f"{sum(token_lengths) / len(token_lengths):.2f}")
    print(OUTPUT_FILE)

if __name__ == "__main__":
    main()