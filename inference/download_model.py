from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_DIR = r"D:\AbhiLLM\models\Qwen3-4B"

print("Downloading Qwen3-4B...")
print(f"Destination: {MODEL_DIR}")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False,
)

print("\nDownload completed.")
print(f"Model location: {MODEL_DIR}")