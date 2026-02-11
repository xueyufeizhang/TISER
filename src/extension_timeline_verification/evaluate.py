import json
import torch
import re
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import time
import networkx as nx

class TISER_Evaluator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """加载模型"""
        print(f"加载模型: {self.model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        load_in_8bit=True,
                        low_cpu_mem_usage=True
                    )
                except:
                    print("⚠️ 8位量化失败，使用普通加载")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True
                    )
            else:
                self.device = torch.device("cpu")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    trust_remote_code=True
                )
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise

    # ==========================================
    # [Extension Core] 逻辑校验
    # ==========================================
    def verify_timeline_logic(self, response):
        timeline_match = re.search(r'<timeline>(.*?)</timeline>', response, re.DOTALL | re.IGNORECASE)
        if not timeline_match:
            return False, "Missing <timeline> tags."
        
        timeline_text = timeline_match.group(1).strip()
        if not timeline_text:
            return False, "Timeline is empty."

        events = defaultdict(dict)
        pattern = r'\((.*?)\)\s*(starts|ends)\s*at\s*(\d+)'
        matches = re.findall(pattern, timeline_text)
        
        if not matches:
            pattern_loose = r'(.*?)\s*(starts|ends)\s*at\s*(\d+)'
            matches = re.findall(pattern_loose, timeline_text)
            if not matches:
                return False, "Timeline format unparseable."

        for event_name, type_, year_str in matches:
            event_name = event_name.strip()
            year = int(year_str)
            if type_ == 'starts':
                events[event_name]['start'] = year
            elif type_ == 'ends':
                events[event_name]['end'] = year

        G = nx.DiGraph()
        for event, times in events.items():
            start = times.get('start')
            end = times.get('end')
            if start is not None and end is not None:
                if start > end:
                    return False, f"Logical Error: Event '{event}' ends ({end}) before it starts ({start})."
                u, v = f"{event}_start", f"{event}_end"
                G.add_edge(u, v)

        try:
            nx.find_cycle(G)
            return False, "Logical Error: Timeline contains a causal loop (Cycle detected)."
        except nx.NetworkXNoCycle:
            pass
            
        return True, None

    def _raw_generate(self, input_text, max_new_tokens):
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False, 
                temperature=0.0, pad_token_id=self.tokenizer.eos_token_id
            )
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        if input_text in full_output:
            generated_part = full_output.replace(input_text, "", 1).strip()
        else:
            generated_part = full_output
        return generated_part

    # ==========================================
    # [Updated] 生成函数 (增加日志记录)
    # ==========================================
    def generate_answer_with_system2(self, prompt, max_new_tokens=512, max_retries=1):
        """
        返回: (final_response, retry_count, trace_log)
        trace_log 包含 System 2 介入的详细过程，用于分析 Bad Case
        """
        history = prompt
        
        # 1. 初始生成
        initial_response = self._raw_generate(history, max_new_tokens)
        
        # 校验
        is_valid, error_msg = self.verify_timeline_logic(initial_response)
        
        trace_log = {
            'triggered': False,
            'initial_output': initial_response,
            'error_detected': None,
            'final_output': initial_response
        }

        retry_count = 0
        current_response = initial_response

        while not is_valid and retry_count < max_retries:
            retry_count += 1
            trace_log['triggered'] = True
            trace_log['error_detected'] = error_msg
            
            # 构建介入 Prompt
            intervention_prompt = (
                f"\n\n[System Alert]: I detected a logical error in your timeline: {error_msg}. "
                "Please regenerate the response (reasoning, timeline, reflection, and answer) correctly."
            )
            
            history = history + current_response + intervention_prompt
            
            # 重新生成
            current_response = self._raw_generate(history, max_new_tokens)
            
            # 再次校验
            is_valid, error_msg = self.verify_timeline_logic(current_response)
        
        trace_log['final_output'] = current_response
        return current_response, retry_count, trace_log
    
    def extract_answer(self, response, prompt=None):
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL | re.IGNORECASE)
        if answer_match: return answer_match.group(1).strip()
        if prompt and prompt in response:
            generated = response.split(prompt)[-1].strip()
            lines = [line.strip() for line in generated.split('\n') if line.strip()]
            if lines: return lines[-1]
        return response.strip()
    
    def calculate_em_f1(self, predicted, ground_truth):
        pred = predicted.strip().lower()
        truth = ground_truth.strip().lower()
        em = 1 if pred == truth else 0
        pred_tokens = set(pred.split())
        truth_tokens = set(truth.split())
        if not pred_tokens or not truth_tokens: return em, 0.0
        common = pred_tokens.intersection(truth_tokens)
        if len(common) == 0: return em, 0.0
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return em, f1
    
    def evaluate_dataset(self, dataset_samples, dataset_name, max_samples=None, verbose=False):
        if max_samples: samples = dataset_samples[:max_samples]
        else: samples = dataset_samples
        
        total_em, total_f1 = 0.0, 0.0
        processed = 0
        system2_triggered_count = 0 
        
        # [New] 用于保存所有触发了 System 2 的案例日志
        intervention_logs = []
        
        pbar = tqdm(samples, desc=f"  {dataset_name.ljust(20)}", unit="smpl")
        
        for i, sample in enumerate(pbar):
            prompt = sample.get('prompt', '')
            ground_truth = sample.get('answer', '')
            if not prompt or not ground_truth: continue
            
            try:
                # 获取 trace_log
                response, retries, trace_log = self.generate_answer_with_system2(prompt, max_retries=1)
                
                # 如果触发了修正，保存到日志列表
                if retries > 0:
                    system2_triggered_count += 1
                    intervention_logs.append({
                        'id': i,
                        'dataset': dataset_name,
                        'prompt': prompt,
                        'ground_truth': ground_truth,
                        'initial_wrong_output': trace_log['initial_output'],
                        'error_message': trace_log['error_detected'],
                        'corrected_output': trace_log['final_output']
                    })
                
                predicted = self.extract_answer(response, prompt)
                em, f1 = self.calculate_em_f1(predicted, ground_truth)
                total_em += em
                total_f1 += f1
                processed += 1
                
                pbar.set_postfix({'EM': f'{total_em/processed:.3f}', 'Sys2': f'{system2_triggered_count}'})
                    
            except Exception as e:
                continue
        
        pbar.close()
        
        if processed == 0: return 0.0, 0.0, 0, []
        
        avg_em = total_em / processed
        avg_f1 = total_f1 / processed
        
        if system2_triggered_count > 0:
            print(f"  [Stats] 触发修正: {system2_triggered_count}/{processed} ({system2_triggered_count/processed:.1%})")
        
        # 返回 logs 供 main 函数保存
        return avg_em, avg_f1, processed, intervention_logs

def load_test_data(json_path):
    print(f"加载测试数据: {json_path}")
    name_mapping = {
        'tgqa_test': 'TGQA', 'tempreason_l2_test': 'TempReason (L2)',
        'tempreason_l3_test': 'TempReason (L3)', 'timeqa_easy_test': 'TimeQA (easy)',
        'timeqa_hard_test': 'TimeQA (hard)', 'tot_semantic_test': 'ToT_Semantic'
    }
    organized_data = defaultdict(list)
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    ds_name = sample.get('dataset_name', '').strip().lower()
                    standard_name = name_mapping.get(ds_name, ds_name)
                    organized_data[standard_name].append(sample)
                except: continue
    return organized_data

def print_table1_format(results):
    print("\n" + "="*90 + "\n评估结果 (Table 1)\n" + "="*90)
    print(f"{'数据集':<25} {'EM':<20} {'F1':<20} {'样本数':<15}")
    print("-" * 85)
    table1_order = ['TGQA', 'TempReason (L2)', 'TempReason (L3)', 'TimeQA (easy)', 'TimeQA (hard)']
    em_scores, f1_scores = [], []
    for ds_name in table1_order:
        if ds_name in results:
            res = results[ds_name]
            print(f"{ds_name:<25} {res['EM']:.3f:<20} {res['F1']:.3f:<20} {res['samples_processed']}/{res['total_samples']}")
            if res['samples_processed'] > 0:
                em_scores.append(res['EM'])
                f1_scores.append(res['F1'])
        else:
            print(f"{ds_name:<25} {'-':<20} {'-':<20} 0/0")
    if em_scores:
        print("-" * 85)
        print(f"{'Macro Average':<25} {np.mean(em_scores):.3f:<20} {np.mean(f1_scores):.3f:<20}")

def main():
    # ========== 请修改路径 ==========
    MODEL_PATH = "C:/Users/Ronnie/Desktop/Python_Test/pythonProject/1"
    TEST_DATA_PATH = "./TISER/data/TISER_test.json"
    FULL_EVALUATION = False  
    MAX_SAMPLES_PER_DATASET = 20 if not FULL_EVALUATION else None

    evaluator = TISER_Evaluator(MODEL_PATH)
    test_data = load_test_data(TEST_DATA_PATH)
    
    results = {}
    all_intervention_cases = []  # [New] 总日志列表

    table1_datasets = ['TGQA', 'TempReason (L2)', 'TempReason (L3)', 'TimeQA (easy)', 'TimeQA (hard)']
    
    print("\n开始评估 (Intervention Mode: ON)")
    
    for ds_name in table1_datasets:
        if ds_name not in test_data: continue
        samples = test_data[ds_name]
        print(f"\n评估: {ds_name} (Total: {len(samples)})")
        
        # 获取 logs
        em, f1, num, logs = evaluator.evaluate_dataset(
            samples, ds_name, MAX_SAMPLES_PER_DATASET, verbose=True
        )
        
        results[ds_name] = {'EM': em, 'F1': f1, 'samples_processed': num, 'total_samples': len(samples)}
        all_intervention_cases.extend(logs) # 收集日志
    
    print_table1_format(results)
    
    # [New] 保存 Bad Case 分析日志
    log_file = "system2_intervention_logs.json"
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(all_intervention_cases, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 介入案例分析已保存到: {log_file}")
    print(f"   共捕获 {len(all_intervention_cases)} 个修正案例，请打开该文件查看 'initial_wrong_output' vs 'corrected_output'。")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()