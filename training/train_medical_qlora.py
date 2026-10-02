import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)

BASE_DIR = Path(r"D:\AbhiLLM")
MODEL_PATH = BASE_DIR / "models" / "Qwen3-4B"
TRAIN_FILE = (BASE_DIR/ "datasets"/ "medical"/ "tokenized"/ "train.jsonl")
VAL_FILE = (BASE_DIR/ "datasets"/ "medical"/ "tokenized"/ "validation.jsonl")
OUTPUT_DIR = (BASE_DIR/ "outputs"/ "medical_qlora")
MAX_LENGTH = 1024

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1

LOGGING_STEPS = 10
SAVE_STEPS = 500
EVAL_STEPS = 500
SEED = 42

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. ")

print("ABHILLM MEDICAL QLoRA TRAINING")

print("\nLoading datasets...")

dataset = load_dataset("json",
    data_files={
        "train": str(TRAIN_FILE),
        "validation": str(VAL_FILE),
    },
)


print(f"\nTrain samples  : "f"{len(dataset['train'])}")
print(f"Validation samples : "f"{len(dataset['validation'])}")

# remove metadata taht i have added in the preprocessing time 
print("\nRemoving metadata columns before training...")

columns_to_remove = [
    column
    for column in dataset["train"].column_names
    if column not in [
        "input_ids",
        "attention_mask",
    ]
]

dataset = dataset.remove_columns(columns_to_remove)

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_PATH),
    local_files_only=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("\nCreating 4-bit NF4 configuration...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print("\nLoading Qwen3-4B...")

model = AutoModelForCausalLM.from_pretrained(
    str(MODEL_PATH),
    quantization_config=bnb_config,
    device_map={"": 0},
    local_files_only=True,
)

print("Model loaded.")

print(
    f"VRAM after loading: "
    f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
)

print("\nPreparing model for QLoRA...")

# qlora
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
model.enable_input_require_grads()

# LoRA
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

model = get_peft_model(model,lora_config)
model.print_trainable_parameters()

print("\nCreating data collator...")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

print("\nCreating training configuration...")

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,

    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,

    learning_rate=LEARNING_RATE,
    fp16=True,
    bf16=False,

    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    logging_steps=LOGGING_STEPS,
    save_strategy="steps",

    save_steps=SAVE_STEPS,
    save_total_limit=2,

    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    report_to="none",

    seed=SEED,
    remove_unused_columns=False,
    dataloader_pin_memory=True,
    dataloader_num_workers=0,
)

print("\nCreating Trainer...")

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],

    processing_class=tokenizer,
    data_collator=data_collator,
)

print("STARTING MEDICAL QLoRA TRAINING")
print("\nTraining will now begin.\n")

# training section
train_result = trainer.train()

print("\nSaving final LoRA adapter...")

OUTPUT_DIR.mkdir(parents=True,exist_ok=True)

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
training_metrics = train_result.metrics

training_metrics_file = (OUTPUT_DIR / "training_metrics.json")

with open(training_metrics_file,"w",encoding="utf-8") as f:
    json.dump(training_metrics,f,indent=2,)


print("\nRunning final validation...")
evaluation_metrics = trainer.evaluate()
print("\nValidation metrics:")

for key, value in evaluation_metrics.items():
    print(f"  {key}: {value}")


evaluation_metrics_file = (OUTPUT_DIR / "evaluation_metrics.json")

with open(evaluation_metrics_file,"w",encoding="utf-8") as f:

    json.dump(evaluation_metrics,f,indent=2)


peak_memory = (torch.cuda.max_memory_allocated()/ 1024**3)

print(f"\nPeak GPU memory: "f"{peak_memory:.2f} GB")

print("MEDICAL QLoRA TRAINING COMPLETE")

print("\nLoRA adapter:")
print(OUTPUT_DIR)

print("\nTraining metrics:")
print(training_metrics_file)

print("\nEvaluation metrics:")
print(evaluation_metrics_file)

print("\n" + "=" * 70)