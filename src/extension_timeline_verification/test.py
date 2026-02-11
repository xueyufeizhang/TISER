import json
import torch
import re
import os
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict
import time
import networkx as nx  # 引入图论库用于逻辑校验

class TISER_Extension:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self._load_model()
    
    def _load_model(self):
        """加载模型，使用量化以节省显存"""
        print(f"加载模型: {self.model_path}")
        
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # 检查GPU并设置设备
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"✅ 使用GPU: {torch.cuda.get_device_name(0)}")
                
                # 尝试使用8位量化
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=torch.float16,
                        device_map="auto",
                        trust_remote_code=True,
                        load_in_8bit=True,  # 8位量化
                        low_cpu_mem_usage=True
                    )
                    print("✅ 使用8位量化加载模型")
                except:
                    # 如果8位量化失败，使用普通加载
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
                print("⚠️ 使用CPU（速度会很慢）")
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
    # [Extension Core] System 2 Thinking 逻辑校验
    # ==========================================
    def verify_timeline_logic(self, response):
        """
        解析模型输出的时间线，构建图谱并检测逻辑错误。
        返回: (is_valid, error_message)
        """
        # 1. 提取 <timeline> 部分
        timeline_match = re.search(r'<timeline>(.*?)</timeline>', response, re.DOTALL | re.IGNORECASE)
        if not timeline_match:
            # 如果连标签都没有，视为格式错误
            return False, "Missing <timeline> tags."
        
        timeline_text = timeline_match.group(1).strip()
        if not timeline_text:
            return False, "Timeline is empty."

        # 2. 解析事件和时间 (使用正则提取 TISER 格式)
        # 格式示例: (Event A) starts at 1990. (Event B) ends at 2000.
        events = defaultdict(dict)
        # 匹配模式: (事件内容) starts/ends at 数字
        pattern = r'\((.*?)\)\s*(starts|ends)\s*at\s*(\d+)'
        matches = re.findall(pattern, timeline_text)
        
        if not matches:
            # 尝试宽松匹配（不带括号的情况）
            pattern_loose = r'(.*?)\s*(starts|ends)\s*at\s*(\d+)'
            matches = re.findall(pattern_loose, timeline_text)
            if not matches:
                return False, "Timeline format unparseable. Expected: '(Event) starts/ends at Year'."

        # 3. 构建数据结构
        for event_name, type_, year_str in matches:
            event_name = event_name.strip()
            year = int(year_str)
            if type_ == 'starts':
                events[event_name]['start'] = year
            elif type_ == 'ends':
                events[event_name]['end'] = year

        # 4. 图论逻辑校验 (Graph-Theoretic Checks)
        G = nx.DiGraph() # 创建有向图
        
        for event, times in events.items():
            start = times.get('start')
            end = times.get('end')
            
            # 校验 A: 时间倒流 (Start > End)
            if start is not None and end is not None:
                if start > end:
                    return False, f"Logical Error: Event '{event}' ends ({end}) before it starts ({start})."
                
                # 在图中添加边: Start_Node -> End_Node (代表时间流向)
                u = f"{event}_start"
                v = f"{event}_end"
                G.add_edge(u, v, weight=(end-start))

        # 校验 B: 环检测 (虽然这里是简单时间轴，但如果模型生成了奇怪的因果链，这里可以扩展)
        try:
            nx.find_cycle(G)
            return False, "Logical Error: Timeline contains a causal loop (Cycle detected in event graph)."
        except nx.NetworkXNoCycle:
            pass
            
        return True, None

    def _raw_generate(self, input_text, max_new_tokens):
        """底层的生成函数，供 System 2 循环调用"""
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0, # 保持确定性
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 提取新生成的部分
        if input_text in full_output:
            generated_part = full_output.replace(input_text, "", 1).strip()
        else:
            generated_part = full_output
            
        return generated_part

    def generate_answer_with_system2(self, prompt, max_new_tokens=512, max_retries=1):
        """
        [System 2 Extension] 带有"介入修正"的生成过程。
        """
        # --- 第一次尝试 (System 1) ---
        history = prompt
        response = self._raw_generate(history, max_new_tokens)
        
        # 运行逻辑校验
        is_valid, error_msg = self.verify_timeline_logic(response)
        
        retry_count = 0
        while not is_valid and retry_count < max_retries:
            # --- 进入 System 2 Intervention Mode ---
            retry_count += 1
            
            # 构建“报错+重试”的 Prompt
            intervention_prompt = (
                f"\n\n[System Alert]: I detected a logical error in your timeline: {error_msg}. "
                "Please regenerate the response (reasoning, timeline, reflection, and answer) correctly."
            )
            
            # 将 错误回答 + 报错信息 拼接到历史中
            history = history + response + intervention_prompt
            
            # 重新生成
            response = self._raw_generate(history, max_new_tokens)
            
            # 再次校验
            is_valid, error_msg = self.verify_timeline_logic(response)
        
        return response, retry_count
    
    def extract_answer(self, response, prompt=None):
        """从模型响应中提取答案"""
        # 方法1: 查找<answer>标签
        answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
        
        # 方法2: 如果有prompt，尝试提取prompt之后的内容 (用于fallback)
        if prompt and prompt in response:
            generated = response.split(prompt)[-1].strip()
            lines = [line.strip() for line in generated.split('\n') if line.strip()]
            if lines:
                return lines[-1]
        
        return response.strip()
    
    def calculate_em_f1(self, predicted, ground_truth):
        """计算Exact Match和F1分数"""
        pred = predicted.strip().lower()
        truth = ground_truth.strip().lower()
        
        em = 1 if pred == truth else 0
        
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
    
    def evaluate_dataset(self, dataset_samples, dataset_name, max_samples=None, verbose=False):
        """评估单个数据集 (集成 System 2)"""
        if max_samples and len(dataset_samples) > max_samples:
            samples = dataset_samples[:max_samples]
            print(f"  ⚠️  限制评估: 前{max_samples}个样本 (共{len(dataset_samples)}个)")
        else:
            samples = dataset_samples
        
        total_em, total_f1 = 0.0, 0.0
        processed = 0
        system2_triggered_count = 0 # 统计触发次数
        
        pbar = tqdm(samples, desc=f"  {dataset_name.ljust(20)}", unit="smpl")
        
        for i, sample in enumerate(pbar):
            prompt = sample.get('prompt', '')
            ground_truth = sample.get('answer', '')
            
            if not prompt or not ground_truth:
                continue
            
            try:
                # === 使用新的 System 2 生成函数 ===
                response, retries = self.generate_answer_with_system2(prompt, max_retries=1)
                
                if retries > 0:
                    system2_triggered_count += 1
                
                # 提取答案
                predicted = self.extract_answer(response, prompt)
                
                # 计算指标
                em, f1 = self.calculate_em_f1(predicted, ground_truth)
                total_em += em
                total_f1 += f1
                processed += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'EM': f'{total_em/processed:.3f}',
                    'Sys2': f'{system2_triggered_count}' # 实时显示介入次数
                })
                
                # 清理显存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                if verbose and i < 3:
                    print(f"\n   样本 {i+1} 失败: {e}")
                continue
        
        pbar.close()
        
        if processed == 0:
            return 0.0, 0.0, 0
        
        avg_em = total_em / processed
        avg_f1 = total_f1 / processed
        
        if system2_triggered_count > 0:
            print(f"  [System 2 Stats] 触发逻辑修正: {system2_triggered_count}/{processed} 次 ({system2_triggered_count/processed:.1%})")
        
        return avg_em, avg_f1, processed

def load_test_data(json_path):
    """加载测试数据并组织"""
    print(f"加载测试数据: {json_path}")
    
    # 数据集名称映射
    name_mapping = {
        'tgqa_test': 'TGQA',
        'tempreason_l2_test': 'TempReason (L2)',
        'tempreason_l3_test': 'TempReason (L3)',
        'timeqa_easy_test': 'TimeQA (easy)',
        'timeqa_hard_test': 'TimeQA (hard)',
        'tot_semantic_test': 'ToT_Semantic'
    }
    
    organized_data = defaultdict(list)
    total_samples = 0
    
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取JSONL文件"):
            line = line.strip()
            if line:
                try:
                    sample = json.loads(line)
                    ds_name = sample.get('dataset_name', '').strip().lower()
                    
                    # 映射到标准名称
                    standard_name = name_mapping.get(ds_name, ds_name)
                    organized_data[standard_name].append(sample)
                    total_samples += 1
                except:
                    continue
    
    print(f"✅ 加载完成: {total_samples} 个总样本")
    
    # 打印统计信息
    table1_datasets = ['TGQA', 'TempReason (L2)', 'TempReason (L3)', 'TimeQA (easy)', 'TimeQA (hard)']
    
    print("\n📊 数据集分布:")
    for ds in table1_datasets:
        cnt = len(organized_data.get(ds, []))
        print(f"  {ds.ljust(25)}: {cnt:>6} 个样本")
    
    return organized_data

def print_table1_format(results):
    """以Table 1格式打印结果"""
    print("\n" + "="*90)
    print("评估结果 (Table 1 格式)")
    print("="*90)
    
    # 表头
    header = f"{'数据集':<25} {'Exact Match (EM)':<20} {'F1 Score':<20} {'评估样本数':<15}"
    print(header)
    print("-" * 85)
    
    table1_order = ['TGQA', 'TempReason (L2)', 'TempReason (L3)', 'TimeQA (easy)', 'TimeQA (hard)']
    
    em_scores, f1_scores = [], []
    
    for ds_name in table1_order:
        if ds_name in results:
            res = results[ds_name]
            em = res['EM']
            f1 = res['F1']
            samples = res['samples_processed']
            total = res['total_samples']
            
            # 格式化输出
            em_str = f"{em:.3f}"
            f1_str = f"{f1:.3f}"
            sample_str = f"{samples}/{total}"
            
            print(f"{ds_name:<25} {em_str:<20} {f1_str:<20} {sample_str:<15}")
            
            if samples > 0:
                em_scores.append(em)
                f1_scores.append(f1)
        else:
            print(f"{ds_name:<25} {'-':<20} {'-':<20} {'0/0':<15}")
    
    # 计算宏平均
    if em_scores:
        macro_em = np.mean(em_scores)
        macro_f1 = np.mean(f1_scores)
        print("-" * 85)
        print(f"{'Macro Average':<25} {macro_em:.3f:<20} {macro_f1:.3f:<20}")
    
    return macro_em, macro_f1 if em_scores else (0, 0)

def main():
    # ========== 配置 ==========
    MODEL_PATH = "C:/Users/Ronnie/Desktop/Python_Test/pythonProject/1"
    TEST_DATA_PATH = "./TISER/data/TISER_test.json"
    
    # 评估设置
    # 为了测试Extension是否正常工作，建议先跑一小部分
    FULL_EVALUATION = False  
    MAX_SAMPLES_PER_DATASET = 20 if not FULL_EVALUATION else None

    print(f"🚀 启动 Neuro-Symbolic TISER Evaluator (System 2 Enabled)")
    print(f"模型路径: {MODEL_PATH}")
    print(f"测试数据: {TEST_DATA_PATH}")
    print(f"模式: {'完整评估' if FULL_EVALUATION else f'快速测试 (最多{MAX_SAMPLES_PER_DATASET}样本/数据集)'}")
    print()
    
    # 1. 加载测试数据
    start_time = time.time()
    test_data = load_test_data(TEST_DATA_PATH)
    data_load_time = time.time() - start_time
    
    # 2. 初始化评估器
    evaluator = TISER_Extension(MODEL_PATH)
    
    # 3. 评估每个数据集
    print("\n" + "="*90)
    print("开始评估 (Intervention Mode: ON)")
    print("="*90)
    
    results = {}
    table1_datasets = ['TGQA', 'TempReason (L2)', 'TempReason (L3)', 'TimeQA (easy)', 'TimeQA (hard)']
    
    total_evaluation_time = 0
    
    for ds_name in table1_datasets:
        if ds_name not in test_data or len(test_data[ds_name]) == 0:
            print(f"\n⚠️  跳过: '{ds_name}' 没有数据")
            results[ds_name] = {'EM': 0, 'F1': 0, 'samples_processed': 0, 'total_samples': 0}
            continue
        
        samples = test_data[ds_name]
        total_samples = len(samples)
        
        print(f"\n评估数据集: {ds_name}")
        print(f"总样本数: {total_samples}")
        
        # 评估
        eval_start = time.time()
        em_score, f1_score, processed = evaluator.evaluate_dataset(
            samples, ds_name, MAX_SAMPLES_PER_DATASET, verbose=True
        )
        eval_time = time.time() - eval_start
        
        results[ds_name] = {
            'EM': em_score,
            'F1': f1_score,
            'samples_processed': processed,
            'total_samples': total_samples,
            'eval_time': eval_time
        }
        
        print(f"  完成: EM={em_score:.4f}, F1={f1_score:.4f}, 时间={eval_time:.1f}秒")
        total_evaluation_time += eval_time
    
    # 4. 打印Table 1格式的结果
    print("\n" + "="*90)
    macro_em, macro_f1 = print_table1_format(results)
    
    # 5. 保存详细结果 (你要求的完整逻辑)
    output_file = "table1_results_detailed_sys2.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'macro_average': {'EM': macro_em, 'F1': macro_f1},
            'evaluation_settings': {
                'model_path': MODEL_PATH,
                'test_data_path': TEST_DATA_PATH,
                'full_evaluation': FULL_EVALUATION,
                'max_samples_per_dataset': MAX_SAMPLES_PER_DATASET,
                'extension': 'System 2 Thinking (Graph Validation)'
            },
            'timing': {
                'data_loading': data_load_time,
                'total_evaluation': total_evaluation_time,
                'total': time.time() - start_time
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 详细结果已保存到: {output_file}")
    
    # 6. 与论文结果对比参考
    print("\n" + "="*90)
    print("论文 Baseline 参考 (TISER Mistral-7B):")
    print("  TGQA: 0.805 | TimeQA(Easy): 0.975 | Macro: 0.887")
    print("  如果你的 EM 略高于此，说明 Extension 有效！")

if __name__ == "__main__":
    # 设置警告过滤
    import warnings
    warnings.filterwarnings("ignore")
    
    # 运行主函数
    main()