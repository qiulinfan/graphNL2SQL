#!/usr/bin/env python3
"""
Inference Script for NL2SQL Model.

This script loads a trained model and generates SQL queries from natural language.

Usage:
    # Interactive mode
    python inference.py --model ./checkpoints/phase2_spider/final

    # Single query
    python inference.py --model ./checkpoints/phase2_spider/final \
        --question "How many students are there?" \
        --schema "[TABLES]\nstudent:\n    id (PK)\n    name\n    age"

    # Evaluate on dev set
    python inference.py --model ./checkpoints/phase2_spider/final --evaluate
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

try:
    from .training_utils import TrainingConfig, load_config_from_json
except ImportError:
    from training_utils import TrainingConfig, load_config_from_json

# Compatibility aliases
FullConfig = TrainingConfig
def get_default_config():
    return TrainingConfig()


def load_model_for_inference(
    checkpoint_path: str,
    base_model_name: Optional[str] = None,
    load_in_4bit: bool = True,
):
    """
    Load trained model for inference.

    Args:
        checkpoint_path: Path to LoRA checkpoint
        base_model_name: Base model name (auto-detected from checkpoint if None)
        load_in_4bit: Use 4-bit quantization

    Returns:
        (model, tokenizer)
    """
    checkpoint_path = Path(checkpoint_path)

    # Try to load adapter config to get base model name
    adapter_config_path = checkpoint_path / "adapter_config.json"
    if adapter_config_path.exists() and base_model_name is None:
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path")

    if base_model_name is None:
        base_model_name = "Qwen/Qwen2.5-7B-Instruct"
        print(f"Warning: Could not detect base model, using default: {base_model_name}")

    print(f"Loading base model: {base_model_name}")

    # Quantization config
    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="left",  # For generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load LoRA weights
    print(f"Loading LoRA weights from: {checkpoint_path}")
    model = PeftModel.from_pretrained(model, str(checkpoint_path))
    model.eval()

    print("Model loaded successfully!")

    return model, tokenizer


def generate_sql(
    model,
    tokenizer,
    question: str,
    schema: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    do_sample: bool = False,
) -> str:
    """
    Generate SQL from question and schema.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        question: Natural language question
        schema: Database schema (in structured format)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        do_sample: Whether to sample or use greedy decoding

    Returns:
        Generated SQL query
    """
    # Build prompt
    system_msg = (
        "You are a SQL expert. Given a database schema and a natural language question, "
        "generate the correct SQL query. Output only the SQL query."
    )

    # Format input (should match training format)
    instruction = (
        "Given the following database schema and question, "
        "generate the SQL query that answers the question."
    )
    user_input = f"{instruction}\n\n{schema}\n\n[QUESTION]\n{question}"

    prompt = f"""<|im_start|>system
{system_msg}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
[SQL]
"""

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract SQL from response
    # Look for content after [SQL] and before <|im_end|>
    if "[SQL]" in generated:
        sql_part = generated.split("[SQL]")[-1]
        if "<|im_end|>" in sql_part:
            sql_part = sql_part.split("<|im_end|>")[0]
        return sql_part.strip()

    # Fallback: return everything after assistant tag
    if "<|im_start|>assistant" in generated:
        return generated.split("<|im_start|>assistant")[-1].strip()

    return generated.strip()


def interactive_mode(model, tokenizer):
    """Run interactive query mode."""
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("Enter 'quit' to exit, 'schema' to set schema")
    print("=" * 60)

    current_schema = """[TABLES]
table:
    id (PK)
    name
    value"""

    while True:
        print(f"\nCurrent schema:\n{current_schema}\n")

        user_input = input("Enter question (or 'schema'/'quit'): ").strip()

        if user_input.lower() == 'quit':
            break
        elif user_input.lower() == 'schema':
            print("Enter new schema (end with empty line):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            current_schema = "\n".join(lines)
            continue

        # Generate SQL
        print("\nGenerating SQL...")
        sql = generate_sql(model, tokenizer, user_input, current_schema)
        print(f"\nGenerated SQL:\n{sql}")


def evaluate_on_dataset(
    model,
    tokenizer,
    data_path: str,
    max_samples: int = 100,
    output_file: Optional[str] = None,
):
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        tokenizer: Tokenizer
        data_path: Path to JSONL evaluation data
        max_samples: Maximum samples to evaluate
        output_file: Optional file to save results
    """
    print(f"\nEvaluating on: {data_path}")

    # Load data
    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    if max_samples:
        data = data[:max_samples]

    results = []
    correct = 0
    total = 0

    for i, example in enumerate(data):
        question = example.get("question", "")
        schema = example.get("schema", "")
        gold_sql = example.get("sql", "")

        # Generate
        pred_sql = generate_sql(model, tokenizer, question, schema)

        # Simple exact match (normalized)
        gold_norm = " ".join(gold_sql.lower().split())
        pred_norm = " ".join(pred_sql.lower().split())
        is_correct = gold_norm == pred_norm

        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "correct": is_correct,
        })

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i + 1}/{len(data)}, Accuracy: {correct/total:.2%}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nFinal Accuracy: {accuracy:.2%} ({correct}/{total})")

    # Save results
    if output_file:
        with open(output_file, 'w') as f:
            json.dump({
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
                "results": results,
            }, f, indent=2)
        print(f"Results saved to: {output_file}")

    return accuracy, results


def main():
    parser = argparse.ArgumentParser(description="NL2SQL Inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (auto-detected if not specified)"
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question to answer"
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Schema for single question"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate on dev set"
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="./training_data/spider_dev.jsonl",
        help="Evaluation data path"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum samples for evaluation"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for evaluation results"
    )
    parser.add_argument(
        "--no-4bit",
        action="store_true",
        help="Disable 4-bit quantization"
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer = load_model_for_inference(
        args.model,
        base_model_name=args.base_model,
        load_in_4bit=not args.no_4bit,
    )

    if args.evaluate:
        # Evaluation mode
        evaluate_on_dataset(
            model, tokenizer,
            args.eval_data,
            max_samples=args.max_samples,
            output_file=args.output,
        )
    elif args.question:
        # Single question mode
        schema = args.schema or "[TABLES]\ntable:\n    column1\n    column2"
        sql = generate_sql(model, tokenizer, args.question, schema)
        print(f"\nGenerated SQL:\n{sql}")
    else:
        # Interactive mode
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
