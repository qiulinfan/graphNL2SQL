"""
Training Configuration for NL2SQL Fine-tuning.

This module defines all hyperparameters and settings for training.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Model configuration."""
    # Model selection - recommended ~5-7B models for NL2SQL
    # Options: "Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3.5-mini-instruct",
    #          "deepseek-ai/deepseek-coder-6.7b-instruct", "codellama/CodeLlama-7b-Instruct-hf"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # Quantization for memory efficiency
    load_in_4bit: bool = True
    load_in_8bit: bool = False

    # Model dtype
    torch_dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"

    # Trust remote code (needed for some models like Qwen)
    trust_remote_code: bool = True


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""
    # LoRA rank - higher = more capacity but more memory
    r: int = 32

    # Scaling factor (typically 1-2x r)
    lora_alpha: int = 64

    # Dropout for regularization
    lora_dropout: float = 0.05

    # Target modules for LoRA
    # For Qwen/LLaMA-style: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    # For lighter training: ["q_proj", "v_proj"]
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])

    # Bias training
    bias: str = "none"  # "none", "all", "lora_only"

    # Task type
    task_type: str = "CAUSAL_LM"


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    # Output directory
    output_dir: str = "./checkpoints"

    # Training phases
    # Phase 1: WikiSQL warmup (1 epoch)
    wikisql_epochs: int = 1
    # Phase 2: Spider main training (3 epochs)
    spider_epochs: int = 3

    # Batch size
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8  # Effective batch = 2 * 8 = 16

    # Learning rate
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05

    # Weight decay
    weight_decay: float = 0.01

    # Max gradient norm for clipping
    max_grad_norm: float = 1.0

    # Sequence lengths
    max_seq_length: int = 1024  # Input + Output combined

    # Logging
    logging_steps: int = 10

    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 200

    # Checkpointing
    save_strategy: str = "epoch"  # Save after each epoch
    save_steps: int = 500  # Also save every N steps as backup
    save_total_limit: int = 5  # Keep last 5 checkpoints

    # Best model tracking
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    # Mixed precision
    fp16: bool = False
    bf16: bool = True  # Prefer bf16 for newer GPUs

    # Gradient checkpointing (saves memory, slightly slower)
    gradient_checkpointing: bool = True

    # Optimizer
    optim: str = "adamw_torch"  # "adamw_torch", "adamw_8bit", "paged_adamw_8bit"

    # Random seed
    seed: int = 42

    # DataLoader
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True


@dataclass
class DataConfig:
    """Data configuration."""
    # Training data directory
    data_dir: str = "./training_data"

    # Data files
    wikisql_train: str = "wikisql_train.jsonl"
    wikisql_dev: str = "wikisql_dev.jsonl"
    spider_train: str = "spider_train.jsonl"
    spider_dev: str = "spider_dev.jsonl"

    # Combined files (optional)
    combined_train: str = "train.jsonl"
    combined_dev: str = "dev.jsonl"

    # Data processing
    max_train_samples: Optional[int] = None  # None = use all
    max_eval_samples: Optional[int] = 500    # Limit eval for speed


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    enabled: bool = True
    project: str = "nl2sql-finetuning"
    run_name: Optional[str] = None  # Auto-generated if None
    tags: List[str] = field(default_factory=lambda: ["nl2sql", "lora", "qwen"])


@dataclass
class FullConfig:
    """Complete training configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    # Training mode
    # "two_phase": WikiSQL warmup → Spider training (recommended)
    # "combined": Train on combined dataset
    # "spider_only": Train only on Spider
    training_mode: str = "two_phase"

    # Resume from checkpoint
    resume_from_checkpoint: Optional[str] = None


def get_default_config() -> FullConfig:
    """Get default configuration."""
    return FullConfig()


def get_small_gpu_config() -> FullConfig:
    """Configuration for smaller GPUs (8-12GB VRAM)."""
    config = FullConfig()
    config.model.load_in_4bit = True
    config.training.per_device_train_batch_size = 1
    config.training.gradient_accumulation_steps = 16
    config.training.gradient_checkpointing = True
    config.lora.r = 16
    config.lora.lora_alpha = 32
    return config


def get_large_gpu_config() -> FullConfig:
    """Configuration for larger GPUs (24GB+ VRAM)."""
    config = FullConfig()
    config.model.load_in_4bit = False
    config.model.load_in_8bit = True
    config.training.per_device_train_batch_size = 4
    config.training.gradient_accumulation_steps = 4
    config.lora.r = 64
    config.lora.lora_alpha = 128
    config.lora.target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    return config


# Quick configurations for common scenarios
CONFIGS = {
    "default": get_default_config,
    "small_gpu": get_small_gpu_config,
    "large_gpu": get_large_gpu_config,
}
