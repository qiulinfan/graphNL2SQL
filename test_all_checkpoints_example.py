"""
Example: Test All Checkpoints
测试所有checkpoints的示例

This script demonstrates how to use test_all_checkpoints() to evaluate
all checkpoints and find the best one.
此脚本演示如何使用 test_all_checkpoints() 评估所有checkpoints并找到最佳的一个。
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.testing_utils import test_all_checkpoints, load_jsonl
from scripts.training_utils import load_datasets

# =============================================================================
# Configuration
# =============================================================================

# Checkpoint directory (adjust to your path)
# Checkpoint目录（根据你的路径调整）
CHECKPOINT_DIR = "./checkpoints/phase2_spider"

# Base model name (auto-detected if None)
# 基础模型名称（如果为None则自动检测）
BASE_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# Evaluation data path
# 评估数据路径
EVAL_DATA_PATH = "./training_data/spider_dev.jsonl"

# Maximum samples to evaluate per checkpoint (None for all)
# 每个checkpoint评估的最大样本数（None表示全部）
MAX_SAMPLES = 100

# Use EGD (Execution-Guided Decoding)
# 是否使用EGD（执行引导解码）
USE_EGD = False
EGD_CANDIDATES = 5

# =============================================================================
# Load Evaluation Data
# =============================================================================

print("Loading evaluation data...")
eval_data = load_jsonl(EVAL_DATA_PATH)
print(f"Loaded {len(eval_data)} evaluation samples")

if MAX_SAMPLES:
    eval_data = eval_data[:MAX_SAMPLES]
    print(f"Using first {MAX_SAMPLES} samples for testing")

# =============================================================================
# Test All Checkpoints
# =============================================================================

print("\n" + "=" * 80)
print("Testing All Checkpoints")
print("=" * 80)
print()

results = test_all_checkpoints(
    checkpoint_dir=CHECKPOINT_DIR,
    eval_data=eval_data,
    base_model_name=BASE_MODEL_NAME,
    max_samples=None,  # Already limited above
    load_in_4bit=True,
    load_in_8bit=False,
    use_egd=USE_EGD,
    egd_candidates=EGD_CANDIDATES,
    verbose=True,
)

# =============================================================================
# Access Results
# =============================================================================

print("\n" + "=" * 80)
print("Results Summary")
print("=" * 80)

# Best checkpoint by EX
best_ex = results["best_ex"]
if best_ex:
    print(f"\n🏆 Best Checkpoint (by EX):")
    print(f"   Name: {best_ex['checkpoint']}")
    print(f"   Step: {best_ex['step']}")
    print(f"   EM: {best_ex['em_accuracy']:.2f}%")
    print(f"   EX: {best_ex['ex_accuracy']:.2f}%")
    print(f"   Path: {best_ex['path']}")

# Best checkpoint by EM
best_em = results["best_em"]
if best_em and best_em != best_ex:
    print(f"\n⭐ Best Checkpoint (by EM):")
    print(f"   Name: {best_em['checkpoint']}")
    print(f"   Step: {best_em['step']}")
    print(f"   EM: {best_em['em_accuracy']:.2f}%")
    print(f"   EX: {best_em['ex_accuracy']:.2f}%")
    print(f"   Path: {best_em['path']}")

# All checkpoints summary
print(f"\n📊 All Checkpoints ({len(results['summary'])} total):")
for item in results["summary"]:
    print(f"   {item['step']:>6} | {item['checkpoint']:<30} | "
          f"EM: {item['em_accuracy']:>6.2f}% | EX: {item['ex_accuracy']:>6.2f}%")

print("\n" + "=" * 80)
print("Done!")
print("=" * 80)

