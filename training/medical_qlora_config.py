from pathlib import Path

BASE_DIR = Path(r"D:\AbhiLLM")

CONFIG = {"model_name": str(BASE_DIR / "models" / "Qwen3-4B"),
          "train_file": str(BASE_DIR/ "datasets"/ "medical"/ "tokenized"/ "train.jsonl"),
          "validation_file": str(BASE_DIR/ "datasets"/ "medical"/ "tokenized"/ "validation.jsonl"),
          "output_dir": str(BASE_DIR/ "outputs"/ "medical_qlora"),

          "max_seq_length": 1024,
           #  here is my qlora
          "load_in_4bit": True,
          "bnb_4bit_quant_type": "nf4",
          "bnb_4bit_use_double_quant": True,
          "bnb_4bit_compute_dtype": "float16",

          "lora_r": 16,
          "lora_alpha": 32,
          "lora_dropout": 0.05,
          "lora_bias": "none",
          "lora_task_type": "CAUSAL_LM",

            #  here is my training
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "warmup_ratio": 0.03,
            # memory optimize
            "gradient_checkpointing": True,
            "optim": "paged_adamw_8bit",
            "fp16": True,
            "bf16": False,
            # LOGGING
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
            "save_total_limit": 2,
            # REPRODUCIBILITY
            "seed": 42,
}

def print_config():
    for key, value in CONFIG.items():
        print(f"{key:<35}: {value}")

if __name__ == "__main__":
    print_config()