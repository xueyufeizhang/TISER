import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os


BASE_MODEL_PATH = 'mistralai/Mistral-7B-Instruct-v0.3'
LORA_ADAPTER_PATH = 'model/Mistral/Adapter'
OUTPUT_DIR = 'model/Mistral/Mistral-7B-Instruct-v0.3-TISER'

def merge():
    print(f"Loading basemodel: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Loading LoRA adapter: {LORA_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

    print("Merging...")
    merged_model = model.merge_and_unload()

    print(f"Saving final model: {OUTPUT_DIR}")
    merged_model.save_pretrained(OUTPUT_DIR, max_shard_size="5GB")
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\nCompletely Merged!")

if __name__ == "__main__":
    merge()