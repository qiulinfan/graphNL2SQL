#!/usr/bin/env python3
"""
NL2SQL Fine-tuning Training Script.

This script implements two-phase training:
1. WikiSQL warmup (1 epoch) - Learn basic SQL patterns
2. Spider training (3 epochs) - Learn complex multi-table queries

Usage:
    # Default two-phase training
    python train.py

    # With custom config
    python train.py --config small_gpu

    # Spider only (skip WikiSQL warmup)
    python train.py --mode spider_only

    # Resume from checkpoint
    python train.py --resume ./checkpoints/phase2_epoch_2
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
    TaskType,
)

from train_config import FullConfig, CONFIGS, get_default_config


# ============================================================================
# GPU Info
# ============================================================================

def print_gpu_info():
    """Print GPU information."""
    print("\n" + "=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)
    if torch.cuda.is_available():
        print(f"CUDA available: True")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA available: False - Training on CPU (will be slow)")
    print("=" * 60 + "\n")


# ============================================================================
# Data Loading
# ============================================================================

def load_jsonl(file_path: str) -> list[dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_training_data(config: FullConfig, phase: str = "wikisql") -> tuple[Dataset, Dataset]:
    """
    Load training data for specified phase.

    Args:
        config: Training configuration
        phase: "wikisql", "spider", or "combined"

    Returns:
        (train_dataset, eval_dataset)
    """
    data_dir = Path(config.data.data_dir)

    if phase == "wikisql":
        train_file = data_dir / config.data.wikisql_train
        eval_file = data_dir / config.data.wikisql_dev
    elif phase == "spider":
        train_file = data_dir / config.data.spider_train
        eval_file = data_dir / config.data.spider_dev
    else:  # combined
        train_file = data_dir / config.data.combined_train
        eval_file = data_dir / config.data.combined_dev

    print(f"Loading {phase} data from {train_file}")

    train_data = load_jsonl(str(train_file))
    eval_data = load_jsonl(str(eval_file))

    # Apply sample limits if specified
    if config.data.max_train_samples:
        train_data = train_data[:config.data.max_train_samples]
    if config.data.max_eval_samples:
        eval_data = eval_data[:config.data.max_eval_samples]

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


# ============================================================================
# Data Preprocessing
# ============================================================================

def create_prompt(example: dict) -> str:
    """
    Create training prompt from example.

    Format for instruction-tuned models:
    <|im_start|>system
    You are a SQL expert...
    <|im_end|>
    <|im_start|>user
    {schema + question}
    <|im_end|>
    <|im_start|>assistant
    {sql}
    <|im_end|>
    """
    # For models without chat template, use simpler format
    system_msg = (
        "You are a SQL expert. Given a database schema and a natural language question, "
        "generate the correct SQL query. Output only the SQL query."
    )

    # The input already contains schema and question
    user_input = example.get("input", "")

    # Get SQL output (handle both formats)
    sql_output = example.get("sql", example.get("output", ""))

    # Remove [SQL] prefix if present (from structured format)
    if sql_output.startswith("[SQL]\n"):
        sql_output = sql_output[6:]

    # Build prompt (using common chat format)
    prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
{sql_output}<|im_end|>"""

    return prompt


def preprocess_for_training(
    examples: dict,
    tokenizer,
    max_length: int = 1024
) -> dict:
    """
    Preprocess examples for causal language modeling.

    Args:
        examples: Batch of examples
        tokenizer: Tokenizer
        max_length: Maximum sequence length

    Returns:
        Tokenized examples with input_ids, attention_mask, and labels
    """
    # Create prompts
    prompts = [create_prompt({"input": inp, "sql": sql})
               for inp, sql in zip(examples["input"], examples["sql"])]

    # Tokenize
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )

    # For causal LM, labels = input_ids (model learns to predict next token)
    tokenized["labels"] = tokenized["input_ids"].copy()

    return tokenized


# ============================================================================
# Model Loading
# ============================================================================

def load_base_model(config: FullConfig):
    """
    Load base model with optional quantization.

    Returns:
        (model, tokenizer)
    """
    print(f"\nLoading model: {config.model.model_name}")

    # Configure quantization
    quantization_config = None
    if config.model.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("  Using 4-bit quantization (QLoRA)")
    elif config.model.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        print("  Using 8-bit quantization")

    # Determine torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.model.torch_dtype, torch.bfloat16)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        trust_remote_code=config.model.trust_remote_code,
        padding_side="right",
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=config.model.trust_remote_code,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )

    # Prepare for k-bit training if quantized
    if config.model.load_in_4bit or config.model.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing
        )

    print(f"  Model loaded. Parameters: {model.num_parameters():,}")

    return model, tokenizer


def apply_lora(model, config: FullConfig):
    """Apply LoRA adapters to model."""
    print("\nApplying LoRA...")

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def load_checkpoint(checkpoint_path: str, config: FullConfig):
    """
    Load model from checkpoint.

    Returns:
        (model, tokenizer)
    """
    print(f"\nLoading from checkpoint: {checkpoint_path}")

    # Load base model
    model, tokenizer = load_base_model(config)

    # Load LoRA weights
    model = PeftModel.from_pretrained(model, checkpoint_path)

    print("  Checkpoint loaded successfully")

    return model, tokenizer


# ============================================================================
# Training
# ============================================================================

def get_training_args(
    config: FullConfig,
    phase: str,
    num_epochs: int,
    output_dir: str,
    run_name: str
) -> TrainingArguments:
    """Create TrainingArguments for a training phase."""

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,

        # Batch size
        per_device_train_batch_size=config.training.per_device_train_batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,

        # Learning rate
        learning_rate=config.training.learning_rate,
        lr_scheduler_type=config.training.lr_scheduler_type,
        warmup_ratio=config.training.warmup_ratio,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,

        # Logging
        logging_steps=config.training.logging_steps,
        logging_dir=f"{output_dir}/logs",

        # Evaluation and Checkpointing
        # Both must match when load_best_model_at_end=True
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,

        # Mixed precision
        fp16=config.training.fp16,
        bf16=config.training.bf16,

        # Gradient checkpointing
        gradient_checkpointing=config.training.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.training.gradient_checkpointing else None,

        # Optimizer
        optim=config.training.optim,

        # Seed
        seed=config.training.seed,

        # DataLoader
        dataloader_num_workers=config.training.dataloader_num_workers,
        dataloader_pin_memory=config.training.dataloader_pin_memory,

        # WandB
        report_to="wandb" if config.wandb.enabled else "none",
        run_name=run_name,

        # Other
        remove_unused_columns=False,
        label_names=["labels"],
    )


def train_phase(
    model,
    tokenizer,
    config: FullConfig,
    phase: str,
    num_epochs: int,
    output_dir: str,
    run_name: str
):
    """
    Run a single training phase.

    Args:
        model: Model to train
        tokenizer: Tokenizer
        config: Configuration
        phase: "wikisql" or "spider"
        num_epochs: Number of epochs
        output_dir: Output directory for this phase
        run_name: WandB run name
    """
    print(f"\n{'=' * 60}")
    print(f"TRAINING PHASE: {phase.upper()}")
    print(f"Epochs: {num_epochs}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Load data
    train_dataset, eval_dataset = load_training_data(config, phase)

    # Preprocess data
    print("\nPreprocessing data...")

    def preprocess_fn(examples):
        return preprocess_for_training(
            examples,
            tokenizer,
            max_length=config.training.max_seq_length
        )

    train_dataset = train_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train data"
    )

    eval_dataset = eval_dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing eval data"
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # Training arguments
    training_args = get_training_args(
        config, phase, num_epochs, output_dir, run_name
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    final_path = f"{output_dir}/final"
    print(f"\nSaving final model to {final_path}")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)

    return model


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="NL2SQL Fine-tuning")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        choices=list(CONFIGS.keys()),
        help="Configuration preset"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="two_phase",
        choices=["two_phase", "combined", "spider_only", "wikisql_only"],
        help="Training mode"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override model name"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging"
    )

    args = parser.parse_args()

    # Load configuration
    config = CONFIGS[args.config]()

    # Apply overrides
    if args.model:
        config.model.model_name = args.model
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.no_wandb:
        config.wandb.enabled = False
    config.training_mode = args.mode
    config.resume_from_checkpoint = args.resume

    # Print info
    print_gpu_info()
    print(f"Configuration: {args.config}")
    print(f"Training mode: {config.training_mode}")
    print(f"Model: {config.model.model_name}")

    # Initialize WandB
    if config.wandb.enabled:
        import wandb
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = config.wandb.run_name or f"nl2sql_{timestamp}"
        wandb.init(
            project=config.wandb.project,
            name=run_name,
            tags=config.wandb.tags,
            config={
                "model": config.model.model_name,
                "lora_r": config.lora.r,
                "lora_alpha": config.lora.lora_alpha,
                "training_mode": config.training_mode,
                "batch_size": config.training.per_device_train_batch_size,
                "gradient_accumulation": config.training.gradient_accumulation_steps,
                "learning_rate": config.training.learning_rate,
            }
        )
    else:
        run_name = "nl2sql_training"

    # Load or resume model
    if args.resume:
        model, tokenizer = load_checkpoint(args.resume, config)
    else:
        model, tokenizer = load_base_model(config)
        model = apply_lora(model, config)

    # Create output directory
    output_base = Path(config.training.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Run training based on mode
    if config.training_mode == "two_phase":
        # Phase 1: WikiSQL warmup
        print("\n" + "=" * 60)
        print("PHASE 1: WikiSQL Warmup")
        print("=" * 60)
        model = train_phase(
            model, tokenizer, config,
            phase="wikisql",
            num_epochs=config.training.wikisql_epochs,
            output_dir=str(output_base / "phase1_wikisql"),
            run_name=f"{run_name}_phase1"
        )

        # Phase 2: Spider main training
        print("\n" + "=" * 60)
        print("PHASE 2: Spider Main Training")
        print("=" * 60)
        model = train_phase(
            model, tokenizer, config,
            phase="spider",
            num_epochs=config.training.spider_epochs,
            output_dir=str(output_base / "phase2_spider"),
            run_name=f"{run_name}_phase2"
        )

    elif config.training_mode == "spider_only":
        model = train_phase(
            model, tokenizer, config,
            phase="spider",
            num_epochs=config.training.spider_epochs,
            output_dir=str(output_base / "spider"),
            run_name=run_name
        )

    elif config.training_mode == "wikisql_only":
        model = train_phase(
            model, tokenizer, config,
            phase="wikisql",
            num_epochs=config.training.wikisql_epochs,
            output_dir=str(output_base / "wikisql"),
            run_name=run_name
        )

    else:  # combined
        model = train_phase(
            model, tokenizer, config,
            phase="combined",
            num_epochs=config.training.spider_epochs,
            output_dir=str(output_base / "combined"),
            run_name=run_name
        )

    # Finish WandB
    if config.wandb.enabled:
        import wandb
        wandb.finish()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print(f"Model saved to: {output_base}")
    print("=" * 60)


if __name__ == "__main__":
    main()
