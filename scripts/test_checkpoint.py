#!/usr/bin/env python3
"""
Test a saved checkpoint after training interruption.

Usage:
    # Test final model
    python scripts/test_checkpoint.py --checkpoint ./checkpoints/phase2_spider/final

    # Test a specific checkpoint
    python scripts/test_checkpoint.py --checkpoint ./checkpoints/phase2_spider/checkpoint-516

    # Test with EGD
    python scripts/test_checkpoint.py --checkpoint ./checkpoints/phase2_spider/final --use-egd

    # Test on custom data
    python scripts/test_checkpoint.py --checkpoint ./checkpoints/phase2_spider/final --eval-data ./training_data/dev.jsonl
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.testing_utils import (
    load_finetuned_model,
    evaluate_with_execution,
    load_jsonl,
)
from scripts.training_utils import load_config_from_json


def main():
    parser = argparse.ArgumentParser(
        description="Test a saved checkpoint after training interruption"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., ./checkpoints/phase2_spider/final or ./checkpoints/phase2_spider/checkpoint-516)",
    )
    parser.add_argument(
        "--eval-data",
        type=str,
        default="./training_data/dev.jsonl",
        help="Path to evaluation data (default: ./training_data/dev.jsonl)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--use-egd",
        action="store_true",
        help="Use Execution-Guided Decoding",
    )
    parser.add_argument(
        "--egd-candidates",
        type=int,
        default=5,
        help="Number of candidates for EGD (default: 5)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name (auto-detected if not specified)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Use 4-bit quantization",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Use 8-bit quantization",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config.json (for auto-detecting base model and quantization settings)",
    )

    args = parser.parse_args()

    # Load config if available
    config = None
    if Path(args.config).exists():
        try:
            config = load_config_from_json(args.config)
            print(f"Loaded config from {args.config}")
            if args.base_model is None:
                args.base_model = config.model_name
            if not args.load_in_4bit and not args.load_in_8bit:
                args.load_in_4bit = config.load_in_4bit
                args.load_in_8bit = config.load_in_8bit
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            print("Using command-line arguments only")

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        parent_dir = checkpoint_path.parent
        if parent_dir.exists():
            for item in sorted(parent_dir.iterdir()):
                if item.is_dir():
                    print(f"  - {item}")
        sys.exit(1)

    # Check eval data exists
    eval_data_path = Path(args.eval_data)
    if not eval_data_path.exists():
        print(f"Error: Evaluation data not found: {eval_data_path}")
        sys.exit(1)

    print("=" * 60)
    print("TESTING CHECKPOINT")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Eval data: {eval_data_path}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print(f"EGD: {args.use_egd}")
    if args.use_egd:
        print(f"EGD candidates: {args.egd_candidates}")
    print()

    # Load model
    print("Loading model...")
    try:
        model, tokenizer = load_finetuned_model(
            str(checkpoint_path),
            base_model_name=args.base_model,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Load evaluation data
    print("Loading evaluation data...")
    eval_data = load_jsonl(str(eval_data_path))
    print(f"Loaded {len(eval_data)} examples")

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    results = evaluate_with_execution(
        model=model,
        tokenizer=tokenizer,
        eval_data=eval_data,
        max_samples=args.max_samples,
        use_egd=args.use_egd,
        egd_candidates=args.egd_candidates,
        verbose=True,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Exact Match (EM):     {results['exact_match_accuracy']:.2%} ({results['exact_match_count']}/{results['total']})")
    print(f"Execution Match (EX): {results['execution_match_accuracy']:.2%} ({results['execution_match_count']}/{results['total']})")
    if results.get('execution_errors', 0) > 0:
        print(f"Execution Errors:     {results['execution_errors']}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

