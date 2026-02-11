import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================
# 1) Prompting
# ================
TISER_SYSTEM = (
    "You are an AI assistant specialized in temporal reasoning. "
    "You must follow the TISER format strictly."
)

TISER_INSTRUCTION = (
    "You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries. "
    "Follow these steps:\n"
    "1. Reason through the problem step by step within the <reasoning> tags.\n"
    "2. Given your previous reasoning, identify relevant temporal events in the given context for answering "
    "the given question within <timeline> tags. Assume relations in the context are unidirectional.\n"
    "3. Reflect on your reasoning and the timeline to check for any errors or improvements within the <reflection> tags.\n"
    "4. Make any necessary adjustments based on your reflection. If there is additional reasoning required, "
    "go back to Step 1, otherwise move to the next step.\n"
    "5. Provide your final, concise answer within the <answer> tags.\n"
    "Output MUST contain <reasoning>, <timeline>, <reflection>, <answer> exactly once each.\n"
)

DEFAULT_SYSTEM = "You are a helpful assistant."


def build_user_prompt(sample_prompt: str, inference: str) -> str:
    if inference == "tiser":
        return f"{TISER_INSTRUCTION}\n\n{sample_prompt}".strip()
    return sample_prompt.strip()


# ================
# 2) Metrics (token-level F1 per Table 1)
# ================
def normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def token_f1(pred: str, truth):
    if isinstance(truth, (list, tuple)):
        best_em, best_f1 = 0, 0.0
        for t in truth:
            em, f1 = token_f1(pred, t)
            if f1 > best_f1 or (f1 == best_f1 and em > best_em):
                best_em, best_f1 = em, f1
        return best_em, best_f1

    pred = normalize_text(pred)
    truth = normalize_text(str(truth))
    em = 1 if pred == truth else 0

    pred_toks = pred.split()
    truth_toks = truth.split()
    if len(pred_toks) == 0 or len(truth_toks) == 0:
        return em, 0.0

    pred_cnt = Counter(pred_toks)
    truth_cnt = Counter(truth_toks)
    common = pred_cnt & truth_cnt
    num_same = sum(common.values())
    if num_same == 0:
        return em, 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(truth_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return em, f1


def extract_answer(response: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", response, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # fallback: last non-empty line
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    return lines[-1] if lines else response.strip()


# ================
# 3) Data loading
# ================
DATASETS = [
    ("TGQA", "TGQA/TGR_test.json"),
    ("TempReason (L2)", "TempReason/TGR_l2_test.json"),
    ("TempReason (L3)", "TempReason/TGR_l3_test.json"),
    ("TimeQA (easy)", "TimeQA/TGR_easy_test.json"),
    ("TimeQA (hard)", "TimeQA/TGR_hard_test.json"),
]


def load_json_list(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt_from_sample(x: dict):
    if "prompt" in x and x["prompt"]:
        return str(x["prompt"]).strip()

    story = None
    for k in ["story", "context", "passage", "background", "article", "support"]:
        if k in x and x[k]:
            story = str(x[k]).strip()
            break

    question = None
    for k in ["question", "query", "input"]:
        if k in x and x[k]:
            question = str(x[k]).strip()
            break

    parts = []
    if story:
        parts.append(story)
    if question:
        parts.append(f"Question: {question}")

    if parts:
        return "\n\n".join(parts).strip()

    return None


def extract_answers_from_sample(x: dict):
    for k in ["answers", "answer", "output", "label"]:
        if k not in x:
            continue
        a = x[k]
        if a is None:
            continue
        if isinstance(a, dict) and "text" in a:
            a = a["text"]
        if isinstance(a, (list, tuple)):
            out = []
            for item in a:
                if isinstance(item, dict) and "text" in item:
                    out.append(str(item["text"]))
                else:
                    out.append(str(item))
            return [s for s in out if s]
        return [str(a)]
    return None


def pack_samples(raw):
    out = []
    for x in raw:
        prompt = build_prompt_from_sample(x)
        answers = extract_answers_from_sample(x)
        if not prompt or not answers:
            continue
        out.append({"prompt": prompt, "answers": answers})
    return out


# ================
# 4) Evaluator
# ================
class Table1Evaluator:
    def __init__(self, model_path: str, inference: str):
        self.model_path = model_path
        self.inference = inference
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if not getattr(self.tokenizer, "chat_template", None):
            template_path = Path(model_path) / "chat_template.jinja"
            if template_path.exists():
                self.tokenizer.chat_template = template_path.read_text(encoding="utf-8")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.tokenizer.padding_side = "left"

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        kwargs = dict(torch_dtype=dtype, device_map="auto", trust_remote_code=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, attn_implementation="flash_attention_2", **kwargs
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

        self.model.eval()

    @torch.no_grad()
    def generate(self, user_prompt: str, max_new_tokens=512) -> str:
        system = TISER_SYSTEM if self.inference == "tiser" else DEFAULT_SYSTEM
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = system + "\n" + user_prompt + "\n"

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        gen_ids = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def eval_dataset(self, samples, name: str, max_new_tokens=512, limit=0):
        total_em, total_f1, n = 0.0, 0.0, 0
        pbar = tqdm(samples, desc=name.ljust(18), unit="smpl")
        for s in pbar:
            if limit and n >= limit:
                break
            prompt = s["prompt"]
            answers = s["answers"]
            user_prompt = build_user_prompt(prompt, self.inference)

            resp = self.generate(user_prompt, max_new_tokens=max_new_tokens)
            pred = extract_answer(resp)

            em, f1 = token_f1(pred, answers)
            total_em += em
            total_f1 += f1
            n += 1
            if n > 0:
                pbar.set_postfix({"EM": f"{total_em/n:.3f}", "F1": f"{total_f1/n:.3f}"})
        pbar.close()
        return (total_em / n if n else 0.0), (total_f1 / n if n else 0.0), n


# ================
# 5) Table print
# ================
def print_table1(results, inference: str):
    order = [ds for ds, _ in DATASETS]
    ems, f1s = [], []
    print("\n" + "=" * 94)
    print(f"Table 1 reproduction (Inference = {inference.upper()}; EM + token-level F1)")
    print("=" * 94)
    print(f"{'Dataset':<22} {'EM':<10} {'F1':<10} {'N':<8}")
    print("-" * 94)
    for k in order:
        if k not in results:
            print(f"{k:<22} {'-':<10} {'-':<10} {'0':<8}")
            continue
        em, f1, n = results[k]
        print(f"{k:<22} {em*100:>7.1f}%   {f1*100:>7.1f}%   {n:<8}")
        if n > 0:
            ems.append(em * 100)
            f1s.append(f1 * 100)
    if ems:
        print("-" * 94)
        print(f"{'Macro Avg':<22} {np.mean(ems):>7.1f}%   {np.mean(f1s):>7.1f}%")
    print("=" * 94 + "\n")


def infer_model_tag(model_path: str) -> str:
    base = Path(model_path).name.lower()
    if "qwen" in base:
        return "qwen"
    if "mistral" in base or "mis" in base:
        return "mis"
    # Fallback: keep alnum only
    tag = re.sub(r"[^a-z0-9]+", "", base)
    return tag or "model"


def write_csv(results, inference: str, csv_out: Path):
    name_map = {
        "TGQA": "TGQA",
        "TempReason (L2)": "TempReason-L2",
        "TempReason (L3)": "TempReason-L3",
        "TimeQA (easy)": "TimeQA-easy",
        "TimeQA (hard)": "TimeQA-hard",
    }
    order = [ds for ds, _ in DATASETS]
    rows = []
    ems, f1s = [], []
    for ds in order:
        if ds not in results:
            continue
        em, f1, n = results[ds]
        rows.append((name_map.get(ds, ds), em, f1))
        if n > 0:
            ems.append(em)
            f1s.append(f1)
    if ems:
        rows.append(("MacroAvg", float(np.mean(ems)), float(np.mean(f1s))))

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bucket", "EM", "F1"])
        for bucket, em, f1 in rows:
            writer.writerow([bucket, em, f1])


def resolve_default_paths():
    here = Path(__file__).resolve()
    model_path = here.parent
    data_root = here.parents[2] / "sxiong_TGQA_repo"
    return model_path, data_root


def main():
    default_model, default_data_root = resolve_default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=str(default_model))
    parser.add_argument("--data_root", type=str, default=str(default_data_root))
    parser.add_argument("--inference", type=str, choices=["tiser", "standard"], default="tiser")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0, help="Limit samples per dataset (0 = no limit)")
    parser.add_argument("--csv_out", type=str, default=None, help="Write results to CSV")
    parser.add_argument("--csv_tag", type=str, default=None, help="Tag for default CSV filename")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    evaluator = Table1Evaluator(args.model_path, args.inference)

    results = {}
    for name, rel_path in DATASETS:
        file_path = data_root / rel_path
        if not file_path.exists():
            print(f"Missing file: {file_path}")
            continue
        raw = load_json_list(file_path)
        samples = pack_samples(raw)
        if not samples:
            print(f"No samples loaded for {name} from {file_path}")
            continue
        em, f1, n = evaluator.eval_dataset(samples, name, max_new_tokens=args.max_new_tokens, limit=args.limit)
        results[name] = (em, f1, n)

    print_table1(results, args.inference)

    if args.csv_out:
        csv_out = Path(args.csv_out)
    else:
        tag = args.csv_tag or infer_model_tag(args.model_path)
        csv_out = Path.cwd() / f"sum_{tag}_{args.inference}_min.csv"
    write_csv(results, args.inference, csv_out)
    print(f"CSV saved: {csv_out}")


if __name__ == "__main__":
    main()
