import json
import torch
import re
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import time


class TISER_Evaluator:
    def __init__(self, model_path: str, prompt_mode: str = "standard"):
        """
        prompt_mode:
          - "tiser": 直接使用 sample["prompt"]
          - "standard": 用 question + temporal context 组装标准prompt
        """
        assert prompt_mode in ("tiser", "standard")
        self.prompt_mode = prompt_mode

        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model()

    def _load_model(self):
        print(f"加载模型: {self.model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")

            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map={"": 0},
                    trust_remote_code=True,
                    load_in_8bit=True,
                    low_cpu_mem_usage=True
                )
                print("✅ 使用8位量化加载模型（单卡 cuda:0）")
            except Exception as e:
                print(f"⚠️ 8位量化失败，改用FP16加载: {e}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map={"": 0},
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                print("✅ 使用FP16加载模型（单卡 cuda:0）")
        else:
            self.device = torch.device("cpu")
            print("⚠️ 使用CPU（速度会很慢）")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True
            )

        self.model.eval()
        print("hf_device_map:", getattr(self.model, "hf_device_map", None))
        print("✅ 模型加载成功")

    # -------------------- Prompt 构造（Standard）--------------------

    @staticmethod
    def _extract_temporal_context_from_prompt(prompt: str) -> str:
        """
        从 TISER prompt 里提取 "Temporal context:" 后面的部分。
        如果找不到，返回空字符串。
        """
        if not prompt:
            return ""
        # 取 "Temporal context:" 到行尾（或到下一个 ###/Answer 之前）
        m = re.search(r"Temporal context:\s*(.*)", prompt, re.DOTALL | re.IGNORECASE)
        if not m:
            return ""
        tail = m.group(1).strip()

        # 避免把 "### Answer:" 等也抓进来
        stop = re.search(r"\n\s*###\s*(Answer|Question)\s*:", tail, re.IGNORECASE)
        if stop:
            tail = tail[: stop.start()].strip()

        return tail.strip()

    def build_prompt(self, sample: dict, role: str) -> str:
        """
        根据 prompt_mode 生成最终输入 prompt
        """
        if self.prompt_mode == "tiser":
            if role == 'system':
                return sample.get("prompt", "").split("### Question:")[1].split("### Answer:")[0].strip()
            elif role == 'user':
                return sample.get("question", "")

        # standard
        question = (sample.get("question", "") or "").strip()

        # 优先从 sample["context"]，没有就从 sample["prompt"] 里抠 temporal context
        context = (sample.get("context", "") or "").strip()
        if not context:
            context = self._extract_temporal_context_from_prompt(sample.get("prompt", ""))

        # 组装标准 prompt（尽量简短，避免模型输出解释）
        # 你也可以按论文的标准格式再微调措辞，但核心是：给 question + temporal context + 要求只输出答案
        if role == 'system':
            prompt = (
                "You are a helpful assistant for temporal reasoning.\n"
                "Answer the question using ONLY the temporal context.\n"
                "Output ONLY the final answer (entity/event) with no explanation.\n\n"
            )
        elif role == 'user':
            prompt = (
                f"Question: {question}\n"
                f"Temporal context: {context}\n"
            )
        return prompt

    # -------------------- 评测逻辑 --------------------

    @staticmethod
    def normalize_answer(s: str) -> str:
        s = (s or "").lower().strip()
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"\s*,\s*", ", ", s)
        s = re.sub(r"[^\w\s,]", "", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def extract_answer(self, generated_text: str) -> str:
        """
        Standard 模式下，模型往往直接输出一行答案；
        TISER 模式下可能带 explanation。这里统一做：截断到 Explanation/Reasoning 之前，取第一行。
        """
        if not generated_text:
            return ""

        m = re.search(r"<answer>\s*(.*?)\s*</answer>", generated_text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip().strip('"').strip("'").strip()

        cut_markers = ["### Explanation", "### Reasoning", "Explanation:", "Reasoning:"]
        cut_pos = len(generated_text)
        for mk in cut_markers:
            pos = generated_text.find(mk)
            if pos != -1:
                cut_pos = min(cut_pos, pos)

        head = generated_text[:cut_pos].strip()
        lines = [l.strip() for l in head.splitlines() if l.strip()]
        ans = lines[0] if lines else head
        ans = re.sub(r"(?i)^\s*(the\s+answer\s+is|answer\s*:)\s*", "", ans).strip()
        return ans.strip().strip('"').strip("'").strip()

    def calculate_em_f1(self, predicted: str, ground_truth: str):
        pred_raw = (predicted or "").strip()
        truth_raw = (ground_truth or "").strip()

        em = 1 if self.normalize_answer(pred_raw) == self.normalize_answer(truth_raw) else 0

        pred = pred_raw.lower().strip()
        truth = truth_raw.lower().strip()

        pred_tokens = set(pred.split())
        truth_tokens = set(truth.split())

        if not pred_tokens or not truth_tokens:
            return em, 0.0

        common = pred_tokens.intersection(truth_tokens)
        if len(common) == 0:
            return em, 0.0

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return em, f1

    def generate_answer(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 64) -> str:
        # inputs = self.tokenizer(
        #     prompt,
        #     return_tensors="pt",
        #     truncation=True,
        #     max_length=2048
        # )
        # inputs = {k: v.to(self.device) for k, v in inputs.items()}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = system_prompt + "\n" + user_prompt + "\n"

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        generated = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return generated

    def evaluate_dataset(self, dataset_samples, dataset_name: str, max_samples=None, verbose: bool = False):
        if max_samples and len(dataset_samples) > max_samples:
            samples = dataset_samples[:max_samples]
            print(f"  ⚠️  限制评估: 前{max_samples}个样本 (共{len(dataset_samples)}个)")
        else:
            samples = dataset_samples

        total_em, total_f1 = 0.0, 0.0
        processed = 0

        pbar = tqdm(samples, desc=f"  {dataset_name.ljust(20)}", unit="样本")

        for i, sample in enumerate(pbar):
            ground_truth = sample.get("answer", "")
            if not ground_truth:
                continue

            system_prompt = self.build_prompt(sample, 'system')
            user_prompt = self.build_prompt(sample, 'user')
            if not system_prompt or not user_prompt:
                continue

            try:
                generated = self.generate_answer(system_prompt, user_prompt)
                predicted = self.extract_answer(generated)

                em, f1 = self.calculate_em_f1(predicted, ground_truth)
                total_em += em
                total_f1 += f1
                processed += 1

                pbar.set_postfix({
                    "EM": f"{total_em/processed:.3f}" if processed > 0 else "0.000",
                    "F1": f"{total_f1/processed:.3f}" if processed > 0 else "0.000"
                })

                if verbose and i < 3:
                    print(f"\n   样本 {i+1}:")
                    print(f"     模式: {self.prompt_mode}")
                    print(f"     生成: {generated[:200]}...")
                    print(f"     预测: {predicted[:160]}...")
                    print(f"     正确答案: {ground_truth}")
                    print(f"     EM: {em}, F1: {f1:.4f}")

            except Exception as e:
                if verbose and i < 3:
                    print(f"\n   样本 {i+1} 失败: {e}")
                continue

        pbar.close()

        if processed == 0:
            return 0.0, 0.0, 0

        return total_em / processed, total_f1 / processed, processed


def load_test_data(json_path: str):
    print(f"加载测试数据: {json_path}")

    name_mapping = {
        "tgqa_test": "TGQA",
        "tempreason_l2_test": "TempReason (L2)",
        "tempreason_l3_test": "TempReason (L3)",
        "timeqa_easy_test": "TimeQA (easy)",
        "timeqa_hard_test": "TimeQA (hard)",
        "tot_semantic_test": "ToT_Semantic"
    }

    organized_data = defaultdict(list)
    total_samples = 0

    with open(json_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="读取JSONL文件"):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                ds_name = sample.get("dataset_name", "").strip().lower()
                standard_name = name_mapping.get(ds_name, ds_name)
                organized_data[standard_name].append(sample)
                total_samples += 1
            except Exception:
                continue

    print(f"✅ 加载完成: {total_samples} 个总样本")

    table1_datasets = ["TGQA", "TempReason (L2)", "TempReason (L3)", "TimeQA (easy)", "TimeQA (hard)"]
    print("\n📊 数据集分布:")
    for ds in table1_datasets:
        cnt = len(organized_data.get(ds, []))
        print(f"  {ds.ljust(25)}: {cnt:>6} 个样本")

    return organized_data


def print_table1_format(results):
    print("\n" + "=" * 90)
    print("评估结果 (Table 1 格式)")
    print("=" * 90)

    header = f"{'数据集':<25} {'Exact Match (EM)':<20} {'F1 Score':<20} {'评估样本数':<15}"
    print(header)
    print("-" * 85)

    table1_order = ["TGQA", "TempReason (L2)", "TempReason (L3)", "TimeQA (easy)", "TimeQA (hard)"]

    em_scores, f1_scores = [], []
    for ds_name in table1_order:
        if ds_name in results:
            res = results[ds_name]
            em = res["EM"]
            f1 = res["F1"]
            samples = res["samples_processed"]
            total = res["total_samples"]
            print(f"{ds_name:<25} {em:<20.3f} {f1:<20.3f} {samples}/{total:<15}")
            if samples > 0:
                em_scores.append(em)
                f1_scores.append(f1)
        else:
            print(f"{ds_name:<25} {'-':<20} {'-':<20} {'0/0':<15}")

    if em_scores:
        macro_em = float(np.mean(em_scores))
        macro_f1 = float(np.mean(f1_scores))
        print("-" * 85)
        print(f"{'Macro Average':<25} {macro_em:<20.3f} {macro_f1:<20.3f}")
        return macro_em, macro_f1

    return 0.0, 0.0


def main():
    MODEL_PATH = r"D:\projects\nlp\Qwen2.5\xueyufeizhangQwen2.5-7B-Instruct-TISER"
    TEST_DATA_PATH = "./data/TISER_test.json"

    # ✅ 改这里：standard / tiser
    PROMPT_MODE = "standard"   # ← 你要 standard 就用这个
    # PROMPT_MODE = "tiser"

    FULL_EVALUATION = True
    MAX_SAMPLES_PER_DATASET = None if FULL_EVALUATION else 100

    print(f"模型路径: {MODEL_PATH}")
    print(f"测试数据: {TEST_DATA_PATH}")
    print(f"Prompt模式: {PROMPT_MODE}")
    print()

    start_time = time.time()
    test_data = load_test_data(TEST_DATA_PATH)
    data_load_time = time.time() - start_time

    evaluator = TISER_Evaluator(MODEL_PATH, prompt_mode=PROMPT_MODE)

    print("\n" + "=" * 90)
    print("开始评估")
    print("=" * 90)

    results = {}
    table1_datasets = ["TGQA", "TempReason (L2)", "TempReason (L3)", "TimeQA (easy)", "TimeQA (hard)"]
    total_evaluation_time = 0.0

    for ds_name in table1_datasets:
        if ds_name not in test_data or len(test_data[ds_name]) == 0:
            print(f"\n⚠️  跳过: '{ds_name}' 没有数据")
            results[ds_name] = {"EM": 0, "F1": 0, "samples_processed": 0, "total_samples": 0}
            continue

        samples = test_data[ds_name]
        total_samples = len(samples)

        print(f"\n评估数据集: {ds_name}")
        print(f"总样本数: {total_samples}")

        eval_start = time.time()
        em_score, f1_score, processed = evaluator.evaluate_dataset(
            samples, ds_name, MAX_SAMPLES_PER_DATASET, verbose=False
        )
        eval_time = time.time() - eval_start

        results[ds_name] = {
            "EM": em_score,
            "F1": f1_score,
            "samples_processed": processed,
            "total_samples": total_samples,
            "eval_time": eval_time
        }
        print(f"  完成: EM={em_score:.4f}, F1={f1_score:.4f}, 时间={eval_time:.1f}秒")
        total_evaluation_time += eval_time

    print("\n" + "=" * 90)
    macro_em, macro_f1 = print_table1_format(results)

    output_file = f"table1_results_{PROMPT_MODE}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "prompt_mode": PROMPT_MODE,
            "results": results,
            "macro_average": {"EM": macro_em, "F1": macro_f1},
            "evaluation_settings": {
                "model_path": MODEL_PATH,
                "test_data_path": TEST_DATA_PATH,
                "full_evaluation": FULL_EVALUATION,
                "max_samples_per_dataset": MAX_SAMPLES_PER_DATASET
            },
            "timing": {
                "data_loading": data_load_time,
                "total_evaluation": total_evaluation_time,
                "total": time.time() - start_time
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 详细结果已保存到: {output_file}")


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()
