import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

# ================= 配置区域 =================
# 1. 原始底座模型路径 (你下载 Qwen2.5 的文件夹)
BASE_MODEL_PATH = 'mistralai/Mistral-7B-Instruct-v0.3'

# 2. 训练好的 LoRA 权重路径 (包含 adapter_model.bin 的文件夹)
LORA_ADAPTER_PATH = 'model/Mistral/Adapter'

# 3. 合并后的输出路径 (准备发给队友的文件夹)
OUTPUT_DIR = 'model/Mistral/Mistral-7B-Instruct-v0.3-TISER'
# ===========================================

def merge():
    print(f"开始加载底座模型: {BASE_MODEL_PATH}")
    # 使用 bfloat16 加载以保持精度，A800 完美支持
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto", # 自动利用 A800 显存
        trust_remote_code=True
    )

    print(f"开始加载 LoRA 权重: {LORA_ADAPTER_PATH}")
    # 将 LoRA 挂载到底座模型上
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)

    print("正在进行权重合并 (Merging)...")
    # merge_and_unload 是核心步骤：它会物理上将 LoRA 矩阵加回到模型权重中
    merged_model = model.merge_and_unload()

    print(f"正在保存完整模型至: {OUTPUT_DIR}")
    # 保存完整的权重、配置文件
    merged_model.save_pretrained(OUTPUT_DIR, max_shard_size="5GB") # 分片保存，方便传输
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n✅ 合并完成！")
    print(f"现在你可以直接打包目录 {OUTPUT_DIR} 发给队友了。")

if __name__ == "__main__":
    merge()