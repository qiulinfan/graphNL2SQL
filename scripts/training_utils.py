"""
Training Utilities for NL2SQL Fine-tuning.
NL2SQL 微调训练工具模块

This module encapsulates all training-related functions for use in notebooks
and scripts.
本模块封装了所有训练相关函数，供 notebook 和脚本使用。

Main Functions / 主要功能:
- load_config_from_json(): Load config from JSON file / 从 JSON 文件加载配置
- load_datasets(): Load WikiSQL and Spider datasets / 加载数据集
- load_model_and_tokenizer(): Load LLM with quantization / 加载量化后的 LLM
- setup_lora(): Configure LoRA adapters / 配置 LoRA 适配器
- train_phase1_wikisql(): WikiSQL warmup training / WikiSQL 预热训练
- train_phase2_spider(): Spider main training / Spider 主训练

Called by / 调用者:
- pipeline.ipynb: Main training notebook / 主训练 notebook
- train.py: Command-line training script / 命令行训练脚本
"""

import json
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

import torch
from datasets import Dataset
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
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# Optional WandB import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class TrainingConfig:
    """Training configuration with all hyperparameters."""
    # Model
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    use_bf16: bool = True
    use_fp16: bool = False
    
    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    
    # Training
    batch_size: int = 2
    gradient_accumulation: int = 8
    learning_rate: float = 2e-4
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.05
    max_seq_length: int = 1024
    gradient_checkpointing: bool = True
    
    # Epochs (supports float, e.g., 0.5 for half epoch)
    # 训练轮次（支持小数，如 0.5 表示半个 epoch）
    wikisql_epochs: float = 1.0
    spider_epochs: float = 3.0
    
    # Paths
    data_dir: str = "./training_data"
    output_dir: str = "./checkpoints"
    
    # Limits
    max_train_samples: Optional[int] = None
    max_eval_samples: int = 500
    
    # Saving
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    
    # Evaluation frequency (as fraction of epoch, e.g., 0.1 = every 0.1 epoch)
    # 验证频率（以 epoch 的分数表示，例如 0.1 = 每 0.1 个 epoch）
    eval_every_epoch_fraction: float = 0.1
    
    # WandB
    use_wandb: bool = True
    wandb_project: str = "nl2sql-finetuning"
    wandb_run_name: Optional[str] = None


def load_config_from_json(config_path: str = "config.json") -> TrainingConfig:
    """
    Load training configuration from JSON file.
    从 JSON 文件加载训练配置。
    
    Function / 功能:
        Reads config.json and creates a TrainingConfig object with all
        hyperparameters for model, LoRA, training, and WandB.
        读取 config.json 并创建包含模型、LoRA、训练和 WandB 所有超参数的配置对象。
    
    Called by / 调用者:
        - pipeline.ipynb: Load config before training (训练前加载配置)
    
    Args / 参数:
        config_path: Path to config.json file (配置文件路径)
        
    Returns / 返回:
        TrainingConfig: Configuration object with all settings (包含所有设置的配置对象)
    """
    with open(config_path, 'r') as f:
        data = json.load(f)
    
    return TrainingConfig(
        # Model
        model_name=data["model"]["model_name"],
        load_in_4bit=data["model"]["load_in_4bit"],
        load_in_8bit=data["model"]["load_in_8bit"],
        # LoRA
        lora_r=data["lora"]["r"],
        lora_alpha=data["lora"]["alpha"],
        lora_dropout=data["lora"]["dropout"],
        lora_target_modules=data["lora"]["target_modules"],
        # Training
        wikisql_epochs=data["training"]["wikisql_epochs"],
        spider_epochs=data["training"]["spider_epochs"],
        batch_size=data["training"]["batch_size"],
        gradient_accumulation=data["training"]["gradient_accumulation"],
        learning_rate=data["training"]["learning_rate"],
        lr_scheduler=data["training"]["lr_scheduler"],
        warmup_ratio=data["training"]["warmup_ratio"],
        max_seq_length=data["training"]["max_seq_length"],
        gradient_checkpointing=data["training"]["gradient_checkpointing"],
        use_bf16=data["training"].get("use_bf16", False),
        use_fp16=data["training"].get("use_fp16", False),
        # Data
        data_dir=data["data"]["data_dir"],
        output_dir=data["data"]["output_dir"],
        max_train_samples=data["data"]["max_train_samples"],
        max_eval_samples=data["data"]["max_eval_samples"],
        # WandB
        use_wandb=data["wandb"]["enabled"],
        wandb_project=data["wandb"]["project"],
        wandb_run_name=data["wandb"]["run_name"],
        # Save
        save_strategy=data["save"]["strategy"],
        save_total_limit=data["save"]["total_limit"],
        # Evaluation frequency
        eval_every_epoch_fraction=data.get("training", {}).get("eval_every_epoch_fraction", 0.1),
    )


def save_config_to_json(config: TrainingConfig, config_path: str = "config.json") -> None:
    """
    Save training configuration to JSON file.
    
    Args:
        config: TrainingConfig object
        config_path: Path to save config.json
    """
    data = {
        "model": {
            "model_name": config.model_name,
            "load_in_4bit": config.load_in_4bit,
            "load_in_8bit": config.load_in_8bit,
        },
        "lora": {
            "r": config.lora_r,
            "alpha": config.lora_alpha,
            "dropout": config.lora_dropout,
            "target_modules": config.lora_target_modules,
        },
        "training": {
            "wikisql_epochs": config.wikisql_epochs,
            "spider_epochs": config.spider_epochs,
            "batch_size": config.batch_size,
            "gradient_accumulation": config.gradient_accumulation,
            "learning_rate": config.learning_rate,
            "lr_scheduler": config.lr_scheduler,
            "warmup_ratio": config.warmup_ratio,
            "max_seq_length": config.max_seq_length,
            "gradient_checkpointing": config.gradient_checkpointing,
            "use_bf16": config.use_bf16,
            "use_fp16": config.use_fp16,
            "eval_every_epoch_fraction": config.eval_every_epoch_fraction,
        },
        "data": {
            "data_dir": config.data_dir,
            "output_dir": config.output_dir,
            "max_train_samples": config.max_train_samples,
            "max_eval_samples": config.max_eval_samples,
        },
        "wandb": {
            "enabled": config.use_wandb,
            "project": config.wandb_project,
            "run_name": config.wandb_run_name,
        },
        "save": {
            "strategy": config.save_strategy,
            "total_limit": config.save_total_limit,
        },
    }
    
    with open(config_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Configuration saved to {config_path}")


def print_config(config: TrainingConfig) -> None:
    """Print configuration summary."""
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"\nModel: {config.model_name}")
    print(f"  4-bit: {config.load_in_4bit}, 8-bit: {config.load_in_8bit}")
    print(f"\nLoRA:")
    print(f"  Rank: {config.lora_r}, Alpha: {config.lora_alpha}, Dropout: {config.lora_dropout}")
    print(f"  Target modules: {config.lora_target_modules}")
    print(f"\nTraining:")
    print(f"  Batch size: {config.batch_size} x {config.gradient_accumulation} = {config.batch_size * config.gradient_accumulation} effective")
    print(f"  Learning rate: {config.learning_rate}, Scheduler: {config.lr_scheduler}")
    print(f"  Epochs: {config.wikisql_epochs} WikiSQL + {config.spider_epochs} Spider")
    print(f"\nPaths:")
    print(f"  Data: {config.data_dir}")
    print(f"  Output: {config.output_dir}")
    print(f"\nWandB: {'Enabled' if config.use_wandb else 'Disabled'}")
    if config.use_wandb:
        print(f"  Project: {config.wandb_project}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_datasets(data_dir: str, verbose: bool = True) -> Tuple[List, List, List, List]:
    """
    Load WikiSQL and Spider datasets.
    
    Args:
        data_dir: Path to training data directory
        verbose: Whether to print dataset info
        
    Returns:
        Tuple of (wikisql_train, wikisql_dev, spider_train, spider_dev)
    """
    data_dir = Path(data_dir)
    
    if verbose:
        print("=" * 60)
        print("DATASET INFORMATION")
        print("=" * 60)
    
    # WikiSQL
    wikisql_train_path = data_dir / "wikisql_train.jsonl"
    wikisql_dev_path = data_dir / "wikisql_dev.jsonl"
    
    wikisql_train_data = []
    wikisql_dev_data = []
    
    if wikisql_train_path.exists():
        wikisql_train_data = load_jsonl(str(wikisql_train_path))
        wikisql_dev_data = load_jsonl(str(wikisql_dev_path))
        if verbose:
            print(f"\n WikiSQL Dataset:")
            print(f"   Train samples: {len(wikisql_train_data):,}")
            print(f"   Dev samples:   {len(wikisql_dev_data):,}")
            print(f"   Columns: {list(wikisql_train_data[0].keys())}")
    else:
        if verbose:
            print(f"\n WikiSQL not found at {wikisql_train_path}")
    
    # Spider
    spider_train_path = data_dir / "spider_train.jsonl"
    spider_dev_path = data_dir / "spider_dev.jsonl"
    
    spider_train_data = []
    spider_dev_data = []
    
    if spider_train_path.exists():
        spider_train_data = load_jsonl(str(spider_train_path))
        spider_dev_data = load_jsonl(str(spider_dev_path))
        if verbose:
            print(f"\n Spider Dataset:")
            print(f"   Train samples: {len(spider_train_data):,}")
            print(f"   Dev samples:   {len(spider_dev_data):,}")
            print(f"   Columns: {list(spider_train_data[0].keys())}")
            
            # Count multi-table and JOIN queries
            multi_table = sum(1 for ex in spider_train_data if ex.get('num_tables', 1) > 1)
            has_join = sum(1 for ex in spider_train_data if ex.get('has_join', False))
            print(f"   Multi-table: {multi_table:,} ({100*multi_table/len(spider_train_data):.1f}%)")
            print(f"   With JOIN:   {has_join:,} ({100*has_join/len(spider_train_data):.1f}%)")
    else:
        if verbose:
            print(f"\n Spider not found at {spider_train_path}")
    
    if verbose:
        print("\n" + "=" * 60)
    
    return wikisql_train_data, wikisql_dev_data, spider_train_data, spider_dev_data


def show_sample_examples(
    wikisql_train_data: List[Dict],
    spider_train_data: List[Dict]
) -> None:
    """Display sample examples from datasets."""
    print("=" * 60)
    print("SAMPLE EXAMPLES")
    print("=" * 60)
    
    if wikisql_train_data:
        print("\n WikiSQL Example:")
        example = wikisql_train_data[0]
        print(f"Question: {example.get('question', 'N/A')}")
        print(f"SQL: {example.get('sql', 'N/A')}")
        print(f"\nSchema preview:")
        schema = example.get('schema', 'N/A')
        print(schema[:500] + "..." if len(schema) > 500 else schema)
    
    if spider_train_data:
        print("\n" + "-" * 60)
        print("\n Spider Example:")
        example = spider_train_data[0]
        print(f"Question: {example.get('question', 'N/A')}")
        print(f"SQL: {example.get('sql', 'N/A')}")
        print(f"Database: {example.get('db_id', 'N/A')}")
        print(f"\nSchema preview:")
        schema = example.get('schema', 'N/A')
        print(schema[:500] + "..." if len(schema) > 500 else schema)


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def create_prompt(example: Dict) -> str:
    """Create training prompt from example."""
    system_msg = (
        "You are a SQL expert. Given a database schema and a natural language question, "
        "generate the correct SQL query. Output only the SQL query."
    )
    
    user_input = example.get("input", "")
    sql_output = example.get("sql", example.get("output", ""))
    
    # Remove [SQL] prefix if present
    if sql_output.startswith("[SQL]\n"):
        sql_output = sql_output[6:]
    
    prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
{sql_output}<|im_end|>"""
    
    return prompt


def preprocess_function(examples: Dict, tokenizer, max_length: int) -> Dict:
    """Preprocess examples for training."""
    prompts = [
        create_prompt({"input": inp, "sql": sql})
        for inp, sql in zip(examples["input"], examples["sql"])
    ]
    
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )
    
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def prepare_dataset(
    data: List[Dict],
    tokenizer,
    max_length: int,
    max_samples: Optional[int] = None,
    desc: str = "data"
) -> Dataset:
    """Prepare dataset for training."""
    if max_samples:
        data = data[:max_samples]
    
    dataset = Dataset.from_list(data)
    
    processed = dataset.map(
        lambda x: preprocess_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names,
        desc=f"Tokenizing {desc}"
    )
    
    return processed


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model_and_tokenizer(config: TrainingConfig) -> Tuple[Any, Any]:
    """
    Load model and tokenizer with quantization config.
    
    Args:
        config: Training configuration
        
    Returns:
        Tuple of (model, tokenizer)
    """
    print("=" * 60)
    print("LOADING MODEL")
    print("=" * 60)
    print(f"\nModel: {config.model_name}")
    
    # Configure quantization
    quantization_config = None
    if config.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization (QLoRA)")
    elif config.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        print("Using 8-bit quantization")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    print("Loading model (this may take a few minutes)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    except ValueError as e:
        if "CPU or the disk" in str(e) and config.load_in_4bit:
            print("\n⚠️  Warning: 4-bit quantization requires all modules on GPU.")
            print("   Trying with explicit device_map='cuda'...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    config.model_name,
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16 if config.use_bf16 else torch.float16,
                    device_map="cuda",  # Force all to GPU
                    trust_remote_code=True,
                )
            except Exception as e2:
                print(f"\n❌ Error: {e2}")
                print("\n💡 Solutions:")
                print("   1. Use 8-bit quantization instead (supports CPU offload)")
                print("   2. Reduce batch_size or max_seq_length")
                print("   3. Use a smaller model")
                print("   4. Free up GPU memory")
                raise
        else:
            raise
    
    # Prepare for k-bit training
    if config.load_in_4bit or config.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.gradient_checkpointing
        )
    
    print("\n Model loaded!")
    
    return model, tokenizer


def setup_lora(model, config: TrainingConfig) -> Any:
    """
    Configure and apply LoRA to model.
    
    Args:
        model: Base model
        config: Training configuration
        
    Returns:
        Model with LoRA adapters
    """
    print("=" * 60)
    print("CONFIGURING LoRA")
    print("=" * 60)
    
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    print(f"\nLoRA Configuration:")
    print(f"   Rank (r):        {config.lora_r}")
    print(f"   Alpha:           {config.lora_alpha}")
    print(f"   Dropout:         {config.lora_dropout}")
    print(f"   Target modules:  {config.lora_target_modules}")
    
    model = get_peft_model(model, lora_config)
    
    print("\n LoRA adapters added!")
    
    return model


def print_model_parameters(model) -> None:
    """Print detailed parameter count information."""
    print("=" * 60)
    print("MODEL PARAMETERS (AFTER LoRA)")
    print("=" * 60)
    
    # Use PEFT's built-in method
    model.print_trainable_parameters()
    
    # Compute manually for more detail
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    def format_params(num):
        if num >= 1e9:
            return f"{num/1e9:.2f}B"
        elif num >= 1e6:
            return f"{num/1e6:.2f}M"
        elif num >= 1e3:
            return f"{num/1e3:.2f}K"
        return str(num)
    
    print(f"\n Detailed Parameter Count:")
    print(f"   Total parameters:     {total_params:>15,} ({format_params(total_params)})")
    print(f"   Trainable parameters: {trainable_params:>15,} ({format_params(trainable_params)})")
    print(f"   Frozen parameters:    {frozen_params:>15,} ({format_params(frozen_params)})")
    print(f"   Trainable %:          {100 * trainable_params / total_params:>14.4f}%")
    
    # LoRA parameter breakdown
    lora_params = sum(p.numel() for n, p in model.named_parameters() if 'lora' in n.lower())
    print(f"\n LoRA Adapter Size:")
    print(f"   LoRA parameters:      {lora_params:>15,} ({format_params(lora_params)})")


# =============================================================================
# WANDB INTEGRATION
# =============================================================================

def init_wandb(config: TrainingConfig) -> str:
    """
    Initialize Weights & Biases logging.
    
    Args:
        config: Training configuration
        
    Returns:
        Run name string
    """
    if not config.use_wandb:
        print("WandB disabled")
        return "nl2sql_training"
    
    if not WANDB_AVAILABLE:
        print("Warning: wandb not installed, disabling")
        return "nl2sql_training"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = config.wandb_run_name or f"nl2sql_{timestamp}"
    
    wandb.init(
        project=config.wandb_project,
        name=run_name,
        config={
            "model": config.model_name,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "batch_size": config.batch_size,
            "gradient_accumulation": config.gradient_accumulation,
            "learning_rate": config.learning_rate,
            "max_seq_length": config.max_seq_length,
            "wikisql_epochs": config.wikisql_epochs,
            "spider_epochs": config.spider_epochs,
        }
    )
    print(f" WandB initialized! Run: {run_name}")
    
    return run_name


def finish_wandb() -> None:
    """Finish WandB run."""
    if WANDB_AVAILABLE:
        try:
            wandb.finish()
            print(" WandB run finished!")
        except:
            pass


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_phase1_wikisql(
    model,
    tokenizer,
    wikisql_train_data: List[Dict],
    wikisql_dev_data: List[Dict],
    config: TrainingConfig,
    run_name: str = "nl2sql"
) -> Trainer:
    """
    Phase 1: WikiSQL warmup training.
    
    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
        wikisql_train_data: WikiSQL training data
        wikisql_dev_data: WikiSQL dev data
        config: Training configuration
        run_name: WandB run name
        
    Returns:
        Trainer object
    """
    if not wikisql_train_data or config.wikisql_epochs <= 0:
        print("Skipping Phase 1 (WikiSQL not available or epochs=0)")
        return None
    
    print("=" * 60)
    print("PHASE 1: WikiSQL Warmup")
    print("=" * 60)
    
    # Prepare datasets
    print("\nPreparing WikiSQL data...")
    wikisql_train_dataset = prepare_dataset(
        wikisql_train_data, tokenizer, config.max_seq_length,
        config.max_train_samples, "WikiSQL train"
    )
    wikisql_eval_dataset = prepare_dataset(
        wikisql_dev_data, tokenizer, config.max_seq_length,
        config.max_eval_samples, "WikiSQL eval"
    )
    
    print(f"\n WikiSQL Dataset Ready:")
    print(f"   Train: {len(wikisql_train_dataset):,} samples")
    print(f"   Eval:  {len(wikisql_eval_dataset):,} samples")
    
    # Calculate steps per epoch for frequent evaluation
    # 计算每个 epoch 的步数，用于频繁验证
    effective_batch_size = config.batch_size * config.gradient_accumulation
    steps_per_epoch = len(wikisql_train_dataset) // effective_batch_size
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    
    # Evaluation frequency from config (default 0.1 epoch)
    # 从配置读取验证频率（默认 0.1 epoch）
    eval_fraction = getattr(config, 'eval_every_epoch_fraction', 0.1)
    
    # Determine strategy: use epoch-based if eval_fraction >= 1.0, otherwise steps-based
    # 确定策略：如果 eval_fraction >= 1.0 使用 epoch-based，否则使用 steps-based
    if eval_fraction >= 1.0:
        # Epoch-based strategy
        # 基于 epoch 的策略
        eval_strategy = "epoch"
        save_strategy = config.save_strategy  # Use config value (usually "epoch")
        eval_steps = None
        save_steps = None
        print(f"\n Training Configuration:")
        print(f"   Eval strategy:      epoch (every {eval_fraction:.1f} epoch)")
        print(f"   Save strategy:      {save_strategy}")
    else:
        # Steps-based strategy for frequent evaluation
        # 基于步数的策略用于频繁验证
        eval_steps = max(1, int(steps_per_epoch * eval_fraction))
        
        # Save checkpoint: ensure save_steps is a multiple of eval_steps
        # 保存 checkpoint：确保 save_steps 是 eval_steps 的整数倍
        # Target: save every ~0.5 epoch, but must be multiple of eval_steps
        # 目标：每 ~0.5 epoch 保存一次，但必须是 eval_steps 的倍数
        target_save_steps = max(1, int(steps_per_epoch * 0.5))
        # Round up to nearest multiple of eval_steps
        # 向上取整到最近的 eval_steps 倍数
        save_steps = ((target_save_steps + eval_steps - 1) // eval_steps) * eval_steps
        
        eval_strategy = "steps"
        save_strategy = "steps"  # Must match eval_strategy for load_best_model_at_end
        
        print(f"\n Training Steps Configuration:")
        print(f"   Steps per epoch:     {steps_per_epoch:,}")
        print(f"   Eval every:          {eval_steps:,} steps (~{eval_fraction} epoch)")
        print(f"   Save every:          {save_steps:,} steps (~{save_steps/steps_per_epoch:.2f} epoch, {save_steps//eval_steps}x eval)")
    
    # Training arguments
    phase1_output = f"{config.output_dir}/phase1_wikisql"
    
    training_args = TrainingArguments(
        output_dir=phase1_output,
        num_train_epochs=config.wikisql_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        warmup_ratio=config.warmup_ratio,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=config.use_bf16,
        fp16=config.use_fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
        report_to="wandb" if config.use_wandb and WANDB_AVAILABLE else "none",
        run_name=f"{run_name}_phase1",
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=wikisql_train_dataset,
        eval_dataset=wikisql_eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("\n Starting Phase 1 training...")
    trainer.train()
    
    # Save
    print(f"\n Saving Phase 1 model to {phase1_output}/final")
    trainer.save_model(f"{phase1_output}/final")
    tokenizer.save_pretrained(f"{phase1_output}/final")
    
    print("\n Phase 1 complete!")
    
    return trainer


def train_phase2_spider(
    model,
    tokenizer,
    spider_train_data: List[Dict],
    spider_dev_data: List[Dict],
    config: TrainingConfig,
    run_name: str = "nl2sql"
) -> Trainer:
    """
    Phase 2: Spider main training.
    
    Args:
        model: Model (possibly after Phase 1)
        tokenizer: Tokenizer
        spider_train_data: Spider training data
        spider_dev_data: Spider dev data
        config: Training configuration
        run_name: WandB run name
        
    Returns:
        Trainer object
    """
    if not spider_train_data or config.spider_epochs <= 0:
        print("Skipping Phase 2 (Spider not available or epochs=0)")
        return None
    
    print("=" * 60)
    print("PHASE 2: Spider Main Training")
    print("=" * 60)
    
    # Prepare datasets
    print("\nPreparing Spider data...")
    spider_train_dataset = prepare_dataset(
        spider_train_data, tokenizer, config.max_seq_length,
        config.max_train_samples, "Spider train"
    )
    spider_eval_dataset = prepare_dataset(
        spider_dev_data, tokenizer, config.max_seq_length,
        config.max_eval_samples, "Spider eval"
    )
    
    print(f"\n Spider Dataset Ready:")
    print(f"   Train: {len(spider_train_dataset):,} samples")
    print(f"   Eval:  {len(spider_eval_dataset):,} samples")
    
    # Calculate steps per epoch for frequent evaluation
    # 计算每个 epoch 的步数，用于频繁验证
    effective_batch_size = config.batch_size * config.gradient_accumulation
    steps_per_epoch = len(spider_train_dataset) // effective_batch_size
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    
    # Evaluation frequency from config (default 0.1 epoch)
    # 从配置读取验证频率（默认 0.1 epoch）
    eval_fraction = getattr(config, 'eval_every_epoch_fraction', 0.1)
    
    # Determine strategy: use epoch-based if eval_fraction >= 1.0, otherwise steps-based
    # 确定策略：如果 eval_fraction >= 1.0 使用 epoch-based，否则使用 steps-based
    if eval_fraction >= 1.0:
        # Epoch-based strategy
        # 基于 epoch 的策略
        eval_strategy = "epoch"
        save_strategy = config.save_strategy  # Use config value (usually "epoch")
        eval_steps = None
        save_steps = None
        print(f"\n Training Configuration:")
        print(f"   Eval strategy:      epoch (every {eval_fraction:.1f} epoch)")
        print(f"   Save strategy:      {save_strategy}")
    else:
        # Steps-based strategy for frequent evaluation
        # 基于步数的策略用于频繁验证
        eval_steps = max(1, int(steps_per_epoch * eval_fraction))
        
        # Save checkpoint: ensure save_steps is a multiple of eval_steps
        # 保存 checkpoint：确保 save_steps 是 eval_steps 的整数倍
        # Target: save every ~0.5 epoch, but must be multiple of eval_steps
        # 目标：每 ~0.5 epoch 保存一次，但必须是 eval_steps 的倍数
        target_save_steps = max(1, int(steps_per_epoch * 0.5))
        # Round up to nearest multiple of eval_steps
        # 向上取整到最近的 eval_steps 倍数
        save_steps = ((target_save_steps + eval_steps - 1) // eval_steps) * eval_steps
        
        eval_strategy = "steps"
        save_strategy = "steps"  # Must match eval_strategy for load_best_model_at_end
        
        print(f"\n Training Steps Configuration:")
        print(f"   Steps per epoch:     {steps_per_epoch:,}")
        print(f"   Eval every:          {eval_steps:,} steps (~{eval_fraction} epoch)")
        print(f"   Save every:          {save_steps:,} steps (~{save_steps/steps_per_epoch:.2f} epoch, {save_steps//eval_steps}x eval)")
    
    # Training arguments
    phase2_output = f"{config.output_dir}/phase2_spider"
    
    training_args = TrainingArguments(
        output_dir=phase2_output,
        num_train_epochs=config.spider_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        warmup_ratio=config.warmup_ratio,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=config.use_bf16,
        fp16=config.use_fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
        report_to="wandb" if config.use_wandb and WANDB_AVAILABLE else "none",
        run_name=f"{run_name}_phase2",
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=spider_train_dataset,
        eval_dataset=spider_eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("\n Starting Phase 2 training...")
    trainer.train()
    
    # Save
    print(f"\n Saving Phase 2 model to {phase2_output}/final")
    trainer.save_model(f"{phase2_output}/final")
    tokenizer.save_pretrained(f"{phase2_output}/final")
    
    print("\n Phase 2 complete!")
    
    return trainer


def finish_training(config: TrainingConfig) -> None:
    """Finish training and cleanup."""
    finish_wandb()
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nCheckpoints saved to: {config.output_dir}")
    print(f"\nTo test the model, run:")
    print(f"  python inference.py --model {config.output_dir}/phase2_spider/final")


# =============================================================================
# HIGH-LEVEL TRAINING FUNCTION
# =============================================================================

def run_full_training(config: TrainingConfig = None) -> Tuple[Any, Any, Trainer]:
    """
    Run the complete two-phase training pipeline.
    
    Args:
        config: Training configuration (uses defaults if None)
        
    Returns:
        Tuple of (model, tokenizer, final_trainer)
    """
    if config is None:
        config = TrainingConfig()
    
    # Load datasets
    wikisql_train, wikisql_dev, spider_train, spider_dev = load_datasets(
        config.data_dir, verbose=True
    )
    
    # Show samples
    show_sample_examples(wikisql_train, spider_train)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Setup LoRA
    model = setup_lora(model, config)
    
    # Print parameters
    print_model_parameters(model)
    
    # Initialize WandB
    run_name = init_wandb(config)
    
    # Phase 1: WikiSQL warmup
    train_phase1_wikisql(
        model, tokenizer,
        wikisql_train, wikisql_dev,
        config, run_name
    )
    
    # Phase 2: Spider main training
    trainer = train_phase2_spider(
        model, tokenizer,
        spider_train, spider_dev,
        config, run_name
    )
    
    # Finish
    finish_training(config)
    
    return model, tokenizer, trainer


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_default_config() -> TrainingConfig:
    """Get default training configuration."""
    return TrainingConfig()


def get_small_gpu_config() -> TrainingConfig:
    """Configuration for smaller GPUs (8-12GB VRAM)."""
    config = TrainingConfig()
    config.load_in_4bit = True
    config.batch_size = 1
    config.gradient_accumulation = 16
    config.lora_r = 16
    config.lora_alpha = 32
    return config


def get_large_gpu_config() -> TrainingConfig:
    """Configuration for larger GPUs (24GB+ VRAM)."""
    config = TrainingConfig()
    config.load_in_4bit = False
    config.load_in_8bit = True
    config.batch_size = 4
    config.gradient_accumulation = 4
    config.lora_r = 64
    config.lora_alpha = 128
    config.lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    return config


# Predefined configurations
CONFIGS = {
    "default": get_default_config,
    "small_gpu": get_small_gpu_config,
    "large_gpu": get_large_gpu_config,
}

