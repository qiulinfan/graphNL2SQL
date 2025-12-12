


# 1. Downloading
python download_data.py

# 2. Preparing
# Full command with all options
python prepare_training_data.py \
    --style basic \              # Schema linearization style
    --format all \               # Output: jsonl, alpaca, chat, or all
    --semantic \                 # Enable semantic column links
    --semantic-threshold 0.8 \   # Similarity threshold (0.0-1.0)
    --wikisql-balanced 5000 \    # Balanced WikiSQL sampling
    --spider                     # Include Spider dataset

# Balanced WikiSQL + full Spider
python prepare_training_data.py --wikisql-balanced 5000 --spider

# Spider only
python prepare_training_data.py --spider --skip-wikisql




# 3. Training
python train.py --config small_gpu  # 8-12GB VRAM
python train.py --config default    # 16-20GB VRAM
python train.py --config large_gpu  # 24GB+ VRAM


# 4. Test
python inference.py --model ./checkpoints/phase2_spider/final