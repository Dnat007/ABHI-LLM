import json
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "qiaojin/PubMedQA"
CONFIG_NAME = "pqa_labeled"

OUTPUT_DIR = Path(r"D:\AbhiLLM\datasets\medical\raw\pubmedqa")
HF_CACHE = Path(r"D:\AbhiLLM\cache\huggingface")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DOWNLOADING PUBMEDQA")
print("=" * 70)

print(f"Dataset : {DATASET_ID}")
print(f"Config  : {CONFIG_NAME}")

dataset = load_dataset(
    DATASET_ID,
    CONFIG_NAME,
    cache_dir=str(HF_CACHE)
)

print("\nAvailable splits:")
print(dataset)

for split_name, split_data in dataset.items():

    output_file = OUTPUT_DIR / f"{split_name}.jsonl"

    print(f"\nSaving {split_name}...")
    print(f"Records: {len(split_data)}")

    with open(output_file, "w", encoding="utf-8") as f:

        for record in split_data:

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    print(f"Saved: {output_file}")

print("\n" + "=" * 70)
print("PUBMEDQA DOWNLOAD COMPLETE")
print("=" * 70)