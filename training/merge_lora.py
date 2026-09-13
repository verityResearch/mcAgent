"""Merge a trained LoRA adapter into Mellum2's base weights for GGUF export.

llama.cpp's convert_hf_to_gguf.py works on a plain (non-PEFT) HF checkpoint,
so the adapter must be merged into full-precision weights first -- this
script loads the base model in bf16 (no quantization, no DeepSpeed), applies
the adapter, merges it via merge_and_unload(), and saves the result as a
standalone HF checkpoint.

Usage:
    venv/bin/python merge_lora.py --adapter-dir path/to/adapter --output-dir out/
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "JetBrains/Mellum2-12B-A2.5B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    print(f"Loading base model {MODEL_ID}...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print(f"Loading adapter from {args.adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print("Merging adapter into base weights...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}...")
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
