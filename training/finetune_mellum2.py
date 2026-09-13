"""QLoRA fine-tune of Mellum2 Instruct on a fact_pipeline verified dataset.

fact_pipeline/src/generation/converter.py writes each admitted sample's
``response`` as ``"<thought>\\n{reasoning}\\n</thought>\\n{answer}"``. Mellum2
Instruct's own chat template strips everything up to a literal ``</think>``
marker from assistant turns (a different tag -- Instruct is meant to answer
directly, with no visible chain-of-thought; that's the Thinking variant's
job), so training data here uses only the ``answer`` half, not the reasoning
wrapper -- training on the full wrapper would push a model that's deployed
and templated as direct-answer towards visible-CoT behavior it wasn't served
for.

Usage:
    venv/bin/python finetune_mellum2.py --dataset path/to/dataset.jsonl --output-dir out/
"""
import argparse
import json
import os

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "JetBrains/Mellum2-12B-A2.5B-Instruct"
MAX_LENGTH = 2048
DEEPSPEED_CONFIG = os.path.join(os.path.dirname(__file__), "deepspeed_zero3_offload.json")


class SFTDataset(Dataset):
    """Tokenizes instruction/answer pairs, masking loss to the assistant turn only.

    Tokenization happens once here at construction, not lazily per __getitem__
    call: an example whose instruction alone reaches MAX_LENGTH leaves
    truncation with no room for any answer tokens, so masking everything
    before the prompt would zero out every label (silent dead weight, or a
    source of NaN loss if a whole batch lands this way). Counting and
    dropping those examples up front needs to see every example's real
    prompt/full lengths before training starts, not discover it mid-epoch.
    """

    def __init__(self, examples, tokenizer):
        self.items = []
        dropped = 0
        for example in examples:
            messages = [
                {"role": "user", "content": example["instruction"]},
                {"role": "assistant", "content": example["answer"]},
            ]
            prompt_text = tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=True)
            full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(
                full_text, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]

            if len(prompt_ids) >= len(full_ids):
                dropped += 1
                continue
            # The chat template renders full_text as prompt_text + answer + the
            # turn's closing tokens, so prompt_ids should be an exact token-level
            # prefix of full_ids -- but that's only guaranteed at the character
            # level; a subword tokenizer could in principle merge across the
            # boundary. Assert rather than trust two independent tokenizer()
            # calls to agree by construction.
            assert full_ids[:len(prompt_ids)] == prompt_ids, (
                "tokenizer's prompt-only and full-conversation encodings disagreed on their "
                "shared prefix -- loss masking below would be wrong")

            # Loss only on the assistant's actual answer tokens -- the prompt
            # (user question plus the template's own assistant-turn preamble)
            # is context, not something the model should be scored on
            # reproducing.
            labels = list(full_ids)
            for i in range(len(prompt_ids)):
                labels[i] = -100
            self.items.append({"input_ids": full_ids, "labels": labels})

        if dropped:
            print(f"WARNING: dropped {dropped}/{len(examples)} example(s) whose instruction alone "
                  f"reached MAX_LENGTH={MAX_LENGTH}, leaving no room for any answer tokens")

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
            # A row missing the converter's delimiter would make split()[-1]
            # silently return the ENTIRE response -- <thought> wrapper and
            # reasoning trace included -- as the training target, which is
            # exactly what this function exists to prevent. Skip loudly
            # instead of training on it.
            marker = "</thought>\n"
            if marker not in row["response"]:
                skipped += 1
                continue
            answer = row["response"].split(marker, 1)[1]
            examples.append({"instruction": row["instruction"], "answer": answer})
    if skipped:
        print(f"WARNING: skipped {skipped} row(s) missing the '</thought>' delimiter -- "
              f"not the fact_pipeline converter's output shape")
    return examples


def build_model_and_tokenizer():
    """Load the base model and attach LoRA adapters.

    Bitsandbytes 4-bit quantization was tried first and abandoned: this
    model's real 4-bit footprint deterministically comes within ~20-500MB of
    a fully-free 5060 Ti's 16GB, and 14 distinct workarounds across two
    quantization libraries (second GPU, CPU offload, a pre-quantized AWQ
    checkpoint, ...) each hit a genuine bug in the quantization ecosystem's
    still-immature MoE support for this brand-new architecture -- see project
    memory project_mellum2_qlora_vram_blocker for the full account. DeepSpeed
    ZeRO-3 with CPU offload sidesteps that class of bug entirely: instead of
    quantizing to fit one GPU, it shards the plain bf16 model's parameters
    and optimizer states across GPU VRAM and the 372GB of CPU RAM available
    on this machine, streaming shards to GPU as each layer computes --
    exactly the "model too big for one GPU, plenty of system RAM" scenario it
    was built for, and a far more mature code path than the newer
    transformers quantized-loading pipeline that caused all 14 earlier
    failures. Verified working end-to-end (2-step smoke run, real adapter
    saved) once two environment gaps were closed: `ninja` for JIT-compiling
    DeepSpeed's CPU Adam kernel, and `python3.14-dev` for its C extension
    headers -- see requirements.txt.

    Must be called with a DeepSpeed ZeRO-3 TrainingArguments already
    constructed (see main()) -- Trainer's DeepSpeed integration hooks
    from_pretrained via a global HfDeepSpeedConfig so weights are sharded as
    they're loaded, not materialized in full on one device first. Loading the
    model before TrainingArguments exists would silently skip that hook.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        # Attention-only: targeting all 64 experts' FFN layers would multiply
        # adapter parameter count ~64x for a MoE model this size, for no
        # established benefit over attention-only LoRA on instruction-tuning-
        # style data (new response patterns/knowledge integration, not new
        # routing behavior).
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
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
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Cap total optimizer steps (for a quick smoke test); -1 trains full epochs")
    # The deepspeed launcher (deepspeed --num_gpus=N script.py ...) always
    # passes --local_rank; TrainingArguments picks the real value up from the
    # LOCAL_RANK env var it also sets, so this only needs to be accepted, not
    # used.
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    examples = load_examples(args.dataset)
    print(f"Loaded {len(examples)} training examples from {args.dataset}")

    # TrainingArguments must be constructed before the model loads: Trainer's
    # DeepSpeed integration registers a global HfDeepSpeedConfig here that
    # from_pretrained (called inside build_model_and_tokenizer, next) checks
    # to decide whether to shard weights across GPU/CPU as they're read in.
    # Building this after the model would silently train unsharded instead.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to=[],
        deepspeed=DEEPSPEED_CONFIG,
    )

    model, tokenizer = build_model_and_tokenizer()
    dataset = SFTDataset(examples, tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    trainer.train()

    # model.save_pretrained() below is only correct because this script is
    # single-GPU (world_size 1): under true multi-rank ZeRO-3 sharding, a
    # LoRA adapter's parameters are created by get_peft_model() AFTER
    # from_pretrained's zero.Init() context has already closed, so they are
    # never gathered before PeftModel.save_pretrained() reads them --
    # verified against peft 0.20.0's save_pretrained/get_peft_model_state_dict
    # source, neither of which has any DeepSpeed/ZeRO-3 awareness. This is a
    # well-documented failure mode for PEFT+DeepSpeed-ZeRO3 upstream
    # (huggingface/peft#1366, huggingface/peft#453, huggingface/trl#4416),
    # producing a truncated or empty adapter file with no error raised. At
    # world_size 1 "partitioning" a tensor across one rank is a no-op (the
    # local shard IS the full tensor), which is what makes the plain call
    # below safe here -- but only here. Assert the assumption explicitly so
    # a future --num_gpus=2 attempt (this box has two GPUs) fails loudly
    # instead of silently writing a corrupted adapter.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    assert world_size == 1, (
        f"model.save_pretrained() below is only correct at world_size=1 (got {world_size}) -- "
        "see the comment at this assertion for why a naive save silently corrupts under true "
        "multi-rank ZeRO-3 sharding of a LoRA-wrapped model")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()
