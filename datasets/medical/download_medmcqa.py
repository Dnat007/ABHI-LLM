from pathlib import Path
from datasets import load_dataset

DATASET_ID = "openlifescienceai/medmcqa"
OUTPUT_DIR = Path(r"D:\AbhiLLM\datasets\medical\raw\medmcqa")
HF_CACHE = Path(r"D:\AbhiLLM\cache\huggingface")
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
HF_CACHE.mkdir(parents=True,exist_ok=True)

print("\nDownloading/loading dataset...")
dataset = load_dataset(DATASET_ID,cache_dir=str(HF_CACHE))

print(dataset)

for split_name, split_data in dataset.items():
    print(f"SPLIT: {split_name}")
    print(f"Rows    : {len(split_data)}")
    print(f"Columns : {split_data.column_names}")

print("SAVING RAW DATA")


for split_name, split_data in dataset.items():
    output_file = OUTPUT_DIR / f"{split_name}.jsonl"
    print(f"Saving {split_name} -> "f"{output_file}")

    split_data.to_json(
        str(output_file),
        orient="records",
        lines=True,
        force_ascii=False
    )


first_split = list(dataset.keys())[0]

sample = dataset[first_split][0]

for key, value in sample.items():

    print(f"\n{key}:")
    print(value)
