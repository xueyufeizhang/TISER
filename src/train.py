import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, PeftModelForCausalLM
from trl import SFTConfig, SFTTrainer

def train():
    model_id = 'mistralai/Mistral-7B-Instruct-v0.3'
    data_path = 'data/TISER_formatted_train.jsonl'
    output_dir = 'model/Mistral/Mistral-7B-Instruct-v0.3-TISER'

    print(f"Loading model: {model_id}...")

  # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        dtype=torch.bfloat16,
        attn_implementation='flash_attention_2', # double check
        trust_remote_code=True
    )
  
  # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

  # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules='all-linear'
    )
  # model = PeftModelForCausalLM.from_pretrained(model, peft_config)
  
  # Load dataset
    print(f"Loading dataset from {data_path}...")
    train_data = load_dataset('json', data_files=data_path, split='train')
    print(f"Total training samples: {len(train_data)}")

  # Training paramaters
    train_args = SFTConfig(
        output_dir=output_dir,
        bf16=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,

        logging_steps=10,
        save_strategy='epoch',
        report_to='wandb',
        run_name='Mistral-7B-Instruct-v0.3-TISER',
        max_length=2048,
        packing=True,
        optim='paged_adamw_8bit'
    )

  # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_data,
        processing_class=tokenizer,
        peft_config=peft_config
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training complete!")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"Error occurs: {e}")
    finally:
        torch.cuda.empty_cache()
        with open("train_finished.txt", "w") as f:
            f.write("done")
        print("Shutdown pending...")
        os.system("vastai stop instance 29300840")
        