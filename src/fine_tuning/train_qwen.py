import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

model_id = "Qwen/Qwen2.5-7B"
data_path = "data/TISER_train.json"
output_dir = "model/Qwen/Qwen2.5-7B-LoRA"

# Hyperparameters
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 8
GRAD_ACCUMULATION = 2
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3

def train():
    print(f"Loading model: {model_id}...")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model (Bfloat16 + Flash Attention 2)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map='auto',
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2", 
        trust_remote_code=True
    )


    def formatting_prompts_func(data):
        text = {
            "prompt": data['prompt'],
            "completion": data['output']
        }
        return text
    

    # LoRA configuration
    peft_config = LoraConfig(
        r=16,                       # LoRA Rank
        lora_alpha=32,              # Usally twice the LoRA Rank
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules='all-linear' # Finetune all linear
    )

    # Load dataset
    print(f"Loading dataset from {data_path}...")
    dataset = load_dataset('json', data_files=data_path, split='train')
    dataset = dataset.map(formatting_prompts_func)
    print(f"Total training samples: {len(dataset)}")

    # Set training parameters
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="wandb",
        run_name="Qwen2.5-7B-TISER",
        
        # Sequence length
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        dataset_kwargs={"add_special_tokens": False},
        completion_only_loss=True,
        dataset_text_field="text",
    )

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
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
        os.system("vastai stop instance 31221501")