"""
NIC-CloudBot: SRE & Cloud Scaling SLM Fine-Tuning Script (QLoRA)
Designed to run on Kaggle / Google Colab with a Free T4/A100 GPU.
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

print("=" * 70)
print("  NIC-CloudBot: SLM QLoRA Fine-Tuning (Mistral-7B-Instruct)")
print("=" * 70)

# Check GPU Environment
print("CUDA Available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device Name:", torch.cuda.get_device_name(0))
    print("GPU Count      :", torch.cuda.device_count())

# 1. Load SRE Dataset
DATASET_PATH = "/kaggle/input/datasets/diveshvalecha/major-project-finetuning/nic_cloudbot_sre_train.json"
TEST_DATASET_PATH = "/kaggle/input/datasets/diveshvalecha/major-project-finetuning/nic_cloudbot_sre_test.json"

print(f"\n[1/5] Loading SRE Training Data from {DATASET_PATH}...")
with open(DATASET_PATH, 'r') as f:
    train_data = json.load(f)

with open(TEST_DATASET_PATH, 'r') as f:
    test_data = json.load(f)

train_ds = Dataset.from_list(train_data)
test_ds = Dataset.from_list(test_data)

print(f"    Train Samples: {len(train_ds):,}")
print(f"    Test Samples : {len(test_ds):,}")

# 2. Format Prompts for Instruction Tuning (ChatML / Alpaca format)
def format_prompt(sample):
    text = (
        f"<|im_start|>system\n{sample['instruction']}<|im_end|>\n"
        f"<|im_start|>user\n{sample['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['output']}<|im_end|>"
    )
    return {"text": text}

train_ds = train_ds.map(format_prompt)
test_ds = test_ds.map(format_prompt)

print("\nSample Formatted Prompt:")
print(train_ds[0]['text'][:400] + "\n...")

# 3. Model & Tokenizer Configuration
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # Or "meta-llama/Meta-Llama-3-8B-Instruct"

print(f"\n[2/5] Configuring 4-bit Quantization (QLoRA) for {MODEL_ID}...")
# Use float16 for Tesla T4 GPU compatibility (prevents bfloat16 errors on Kaggle)
compute_dtype = torch.float16 if torch.cuda.get_device_capability()[0] < 8 else torch.bfloat16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype
)

print("[3/5] Loading Tokenizer & Base Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare Model for LoRA
model = prepare_model_for_kbit_training(model)

# 4. LoRA Target Modules Configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
print("\nTrainable Parameters:")
model.print_trainable_parameters()

# 5. Tokenize Dataset for Stable Training
print("\n[4/5] Tokenizing dataset...")
def tokenize_fn(sample):
    result = tokenizer(
        sample["text"],
        truncation=True,
        max_length=768,
        padding=False
    )
    result["labels"] = result["input_ids"].copy()
    return result

train_tokenized = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
test_tokenized = test_ds.map(tokenize_fn, remove_columns=test_ds.column_names)

# 6. Training Arguments & Standard HuggingFace Trainer
OUTPUT_DIR = "./nic-cloudbot-slm-adapter"

from transformers import Trainer, DataCollatorForSeq2Seq

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    optim="paged_adamw_8bit",
    fp16=(compute_dtype == torch.float16),
    bf16=(compute_dtype == torch.bfloat16),
    logging_steps=25,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    warmup_steps=50,
    lr_scheduler_type="cosine",
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    pad_to_multiple_of=8,
    return_tensors="pt",
    padding=True
)

trainer = Trainer(
    model=model,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    data_collator=data_collator,
    args=training_args
)

print("\n[5/5] Starting Fine-Tuning...")
trainer.train()

# 7. Save Final Adapter and Tokenizer
print("\nSaving Fine-Tuned Adapter to", OUTPUT_DIR)
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("=" * 70)
print("SLM Fine-Tuning Complete! Adapter saved successfully.")
print("=" * 70)
