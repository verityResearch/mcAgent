"""LoRA fine-tune of GPT2-XL (1.5B) on a fact_pipeline verified dataset, on CUDA.

Adapted from training/finetune_mellum2.py's data-prep and masking approach
(see that file for the full rationale): no chat template (GPT2 is a plain
base LM, not instruction-tuned), no DeepSpeed (unnecessary at this model
size -- 1.5B fits in a single consumer GPU with LoRA + fp16). Base model
loads in fp16, not fp32: a 6GB card (e.g. RTX 2060) can't hold GPT2-XL's
~6GB fp32 weights plus LoRA optimizer state and activations, but fp16
weights (~3GB) leave enough headroom, and CUDA's fp16 support (unlike MPS's)
is mature. LoRA adapter params still train in fp32 (PEFT's default), so
optimizer precision is unaffected.

Training text format: "Question: {instruction}\nAnswer: {answer}<|endoftext|>",
loss masked to the answer span only, same principle as the Mellum2 script.

Usage:
    python3 training/finetune_gpt2xl.py \
        --dataset fact_pipeline/data/verified/dataset_gpt2_factseeded_large.jsonl \
        --output-dir out/gpt2xl_lora/
"""
import argparse
import json

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

MODEL_ID = "gpt2-xl"
MAX_LENGTH = 1024


class SFTDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.items = []
        dropped = 0
        for example in examples:
            prompt_text = f"Question: {example['instruction']}\nAnswer:"
            full_text = f"{prompt_text} {example['answer']}{tokenizer.eos_token}"

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(
                full_text, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]

            if len(prompt_ids) >= len(full_ids):
                dropped += 1
                continue
            assert full_ids[:len(prompt_ids)] == prompt_ids, (
                "prompt-only and full-text encodings disagreed on their shared prefix")

            labels = list(full_ids)
            for i in range(len(prompt_ids)):
                labels[i] = -100
            self.items.append({"input_ids": full_ids, "labels": labels})

        if dropped:
            print(f"WARNING: dropped {dropped}/{len(examples)} example(s) whose prompt alone "
                  f"reached MAX_LENGTH={MAX_LENGTH}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch, pad_token_id):
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, attention_mask = [], [], []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
        attention_mask.append([1] * len(item["input_ids"]) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


def load_examples(dataset_path):
    examples = []
    skipped = 0
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            marker = "</thought>\n"
            if marker not in row["response"]:
                skipped += 1
                continue
            answer = row["response"].split(marker, 1)[1]
            examples.append({"instruction": row["instruction"], "answer": answer})
    if skipped:
        print(f"WARNING: skipped {skipped} row(s) missing the '</thought>' delimiter")
    return examples


def build_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        # GPT2 uses a fused Conv1D for QKV (c_attn) rather than separate
        # q/k/v projections; c_proj is the attention output projection. Both
        # attn.c_proj and mlp.c_proj share the literal name "c_proj", so
        # target_modules=["c_attn","c_proj"] alone (v1's config) already hit
        # attn.c_proj AND mlp.c_proj -- confirmed by inspecting the adapted
        # module names after get_peft_model, not assumed. What v1 missed was
        # mlp.c_fc, the FFN's up-projection into its larger intermediate
        # dimension (6400 vs. 1600) -- interpretability research (e.g. Geva
        # et al.) points to FFN layers as where factual associations are
        # actually stored, so v1's fact-seeded answers fabricating specific
        # facts (piston recipe, zombie drops, llama feed) while nailing
        # response *format* is consistent with LoRA never having reached the
        # one FFN sublayer most likely to encode new facts. Adding "c_fc"
        # completes FFN coverage. PEFT has built-in Conv1D support for all
        # three.
        target_modules=["c_attn", "c_proj", "c_fc"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-steps", type=int, default=-1)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This script targets CUDA; no CUDA device is visible.")

    examples = load_examples(args.dataset)
    print(f"Loaded {len(examples)} training examples from {args.dataset}")

    model, tokenizer = build_model_and_tokenizer()
    model = model.to("cuda")
    dataset = SFTDataset(examples, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
