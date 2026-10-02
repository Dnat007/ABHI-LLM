import json
import gc
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


BASE_DIR = Path(r"D:\AbhiLLM")

MODEL_PATH = BASE_DIR / "models" / "Qwen3-4B"
TRAIN_FILE = (
    BASE_DIR
    / "datasets"
    / "medical"
    / "tokenized"
    / "train.jsonl"
)


def load_one_sample():

    with open(TRAIN_FILE, "r", encoding="utf-8") as f:

        line = f.readline()

    return json.loads(line)


def main():

    print("=" * 70)
    print("ABHILLM QLoRA SMOKE TEST")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")

    print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(
        f"VRAM: {total_memory / 1024**3:.2f} GB"
    )

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    # ---------------------------------------------------------
    # 4-bit configuration
    # ---------------------------------------------------------

    print("Creating 4-bit NF4 configuration...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    print("\nLoading Qwen3-4B in 4-bit...")

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        quantization_config=bnb_config,
        device_map={"": 0},
        local_files_only=True,
    )

    print("Model loaded.")

    print(
        f"VRAM after model load: "
        f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
    )

    # ---------------------------------------------------------
    # Prepare for QLoRA
    # ---------------------------------------------------------

    print("\nPreparing model for k-bit training...")

    model = prepare_model_for_kbit_training(model)

    # ---------------------------------------------------------
    # LoRA
    # ---------------------------------------------------------

    print("Adding LoRA adapters...")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    # ---------------------------------------------------------
    # Load one training example
    # ---------------------------------------------------------

    print("\nLoading one training example...")

    sample = load_one_sample()

    input_ids = torch.tensor(
        [sample["input_ids"]],
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.tensor(
        [sample["attention_mask"]],
        dtype=torch.long,
        device=device,
    )

    labels = input_ids.clone()

    print(
        f"Input tokens: "
        f"{input_ids.shape[1]}"
    )

    # ---------------------------------------------------------
    # Enable training
    # ---------------------------------------------------------

    model.train()

    # Qwen models with gradient checkpointing
    # need input gradients enabled.
    model.enable_input_require_grads()

    # ---------------------------------------------------------
    # Forward + backward
    # ---------------------------------------------------------

    print("\nRunning forward pass...")

    torch.cuda.reset_peak_memory_stats()

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    loss = outputs.loss

    print(f"Loss: {loss.item():.4f}")

    print("\nRunning backward pass...")

    loss.backward()

    print("Backward pass successful.")

    # ---------------------------------------------------------
    # Optimizer step
    # ---------------------------------------------------------

    print("\nRunning optimizer step...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-4,
    )

    optimizer.step()

    print("Optimizer step successful.")

    # ---------------------------------------------------------
    # Memory
    # ---------------------------------------------------------

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    peak = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print("\n" + "-" * 70)
    print("GPU MEMORY")
    print("-" * 70)

    print(f"Allocated : {allocated:.2f} GB")
    print(f"Reserved  : {reserved:.2f} GB")
    print(f"Peak      : {peak:.2f} GB")

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    del outputs
    del model
    del optimizer

    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("QLoRA SMOKE TEST PASSED")
    print("=" * 70)

    print("\nYour environment successfully completed:")

    print("  [OK] Qwen3-4B 4-bit loading")
    print("  [OK] NF4 quantization")
    print("  [OK] LoRA adapter injection")
    print("  [OK] Forward pass")
    print("  [OK] Backward pass")
    print("  [OK] Optimizer step")

    print("\nNo full training was performed.")


if __name__ == "__main__":
    main()