#!/usr/bin/env python3
"""LoRA fine-tune Qwen3-1.7B on CUDA (PyTorch/transformers/peft/trl), replacing
the MLX training used on Apple Silicon (MLX only, unavailable on this Linux
box). Reads the SAME data format used throughout this session: JSONL with a
"messages" list per line (chat-format multi-turn tool-call traces).

    python3 train_lora_cuda.py --train train.jsonl --valid valid.jsonl \
        --out-dir adapter-tool-v11-cuda --epochs 3 --lr 1e-4 --batch-size 4
"""
import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-1.7B"


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    args = ap.parse_args()

    print(f"loading tokenizer + model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_rows = load_jsonl(args.train)
    valid_rows = load_jsonl(args.valid)
    print(f"train={len(train_rows)} valid={len(valid_rows)}")

    train_ds = Dataset.from_list(train_rows)
    valid_ds = Dataset.from_list(valid_rows)

    sft_config = SFTConfig(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_length=args.max_len,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=5,
        bf16=True,
        report_to=[],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    print(f"done -- adapter saved to {args.out_dir}")


if __name__ == "__main__":
    main()
