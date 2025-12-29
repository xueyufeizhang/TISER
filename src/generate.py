import torch
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = 'xueyufeizhang/Qwen2.5-7B-Instruct-TISER'
# model_id = '/workspace/Mistral-7B-Instruct-v0.3-TISER'

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map='auto',
    attn_implementation='flash_attention_2',
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)
model.eval()

with open('/workspace/TISER/data/TISER_test.json', 'r') as f:
    for i, line in enumerate(f):
        if i == 1:
            line = line.strip()
            entry = json.loads(line)
            system_text = entry.get('prompt', "")
            user_text = entry.get('question', "")
            assistant_text = entry.get('output', "")
            messages =  [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
            break


text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1
    )
response_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)]
response = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]

print("-" * 30)
print(f"System: {system_text}")
print(f"User: {user_text}")
print(f"Golden assistant: {assistant_text}")
print("-" * 30)
print(f"Generated assistant: {response}")