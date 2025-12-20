import json
import os

input_file = 'data/TISER_train.json'
output_file = 'data/TISER_formatted_train.jsonl'

def format_tiser_data_simple(input_path):
    formatted_data = []
    print(f"正在读取 {input_path} ...")

    try:
        with open(input_file, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue

                try:
                    entry = json.loads(line)
                    system_text = entry.get('prompt', "")
                    user_text = entry.get('question', "")
                    assistant_text = entry.get('output', "")
            
                except json.JSONDecodeError:
                    print(f"Worning: Fail to read line No.{i+1}.")
                    continue

    except FileNotFoundError:
        print(f"Error: fail to find the file {input_path}.")
        return []
    
    return formatted_data


if __name__ == "__main__":
    format_tiser_data_simple(input_file)