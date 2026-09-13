#!/usr/bin/env bash
# LoRA fine-tune Qwen3-1.7B on the fact_pipeline Minecraft corpus, on an Apple-Silicon Mac (MLX).
# Runs on an Apple-Silicon Mac. Data dir must contain train.jsonl / valid.jsonl (mlx chat format,
# produced by convert_to_mlx_lora.py). QLoRA-style: LoRA adapter over the 4-bit base.
#
# Hyperparameters reflect the training methodology (see METHODOLOGY_qwen3_1.7b.md):
#  - LoRA, not full FT (16GB box; LoRA gets ~90% of the benefit, proven working).
#  - --num-layers 28 = all blocks: fact recall lives in FFN across depth (GPT2-XL
#    lesson), so don't restrict to the last few layers.
#  - Moderate epochs + val every 50 steps: watch val loss and STOP when it turns
#    up — the tool-harness discussion showed low-data-high-epoch overfits args;
#    same risk for fact memorization vs generalization.
#  - Held-out valid split already separated by convert_to_mlx_lora.py.
set -euo pipefail

VENV=${VENV:-$HOME/.venvs/mlx/bin}
MODEL=${MODEL:-$HOME/qwen3-lora/base-4bit}
DATA=${DATA:-$HOME/qwen3-lora/data-mc}
ADAPTER=${ADAPTER:-$HOME/qwen3-lora/adapter-mc}
EPOCHS=${EPOCHS:-3}
BATCH=${BATCH:-4}
LR=${LR:-1e-4}
MAXLEN=${MAXLEN:-1536}
NUM_LAYERS=${NUM_LAYERS:-28}

n_train=$(wc -l < "$DATA/train.jsonl")
# iters = epochs * ceil(n_train / batch)
iters=$(( EPOCHS * ( (n_train + BATCH - 1) / BATCH ) ))
echo "train=$n_train  batch=$BATCH  epochs=$EPOCHS  -> iters=$iters  (lr=$LR, layers=$NUM_LAYERS, maxlen=$MAXLEN)"

exec "$VENV/python" -m mlx_lm.lora \
  --model "$MODEL" \
  --train --data "$DATA" \
  --fine-tune-type lora \
  --num-layers "$NUM_LAYERS" \
  --batch-size "$BATCH" \
  --iters "$iters" \
  --learning-rate "$LR" \
  --max-seq-length "$MAXLEN" \
  --steps-per-report 10 \
  --steps-per-eval 50 \
  --val-batches 10 \
  --save-every 200 \
  --grad-checkpoint \
  --adapter-path "$ADAPTER"
