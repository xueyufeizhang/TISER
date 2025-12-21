import json
import os

input_file = 'data/TISER_train.json'
output_file = 'data/TISER_formatted_train.jsonl'

def format_tiser_data_simple(input_path):
    formatted_data = []
    print(f"Loading {input_path} ...")

    try:
        with open(input_file, 'r') as f:
            for i, line in enumerate(f):
            # for line in f:
                line = line.strip()
                if not line: continue

                try:
                    entry = json.loads(line)
                    system_text = entry.get('prompt', "")
                    user_text = entry.get('question', "")
                    assistant_text = entry.get('output', "")

                    sample = {
                        "messages": {
                            {"role": "system", "content": system_text},
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": assistant_text}
                        }
                    }
                    formatted_data.append(sample)
            
                except json.JSONDecodeError:
                    print(f"Worning: Fail to read line No.{i+1}.")
                    continue

    except FileNotFoundError:
        print(f"Error: fail to find the file {input_path}.")
        return []
    
    return formatted_data


if __name__ == "__main__":
    processed = format_tiser_data_simple(input_file)

    if processed:
        print(f"Format completed! {len(processed)} data are saved in total.")

        with open(output_file, 'w', encoding='utf-8') as f:
            for item in processed:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"File already saved at: {output_file}")

        print("\n=== Formatted data preview (Check this!) ===")
        sample_msg = processed[0]['messages']
        print(f"[System]: {sample_msg[0]['content'][:50]}...")
        print(f"[User]: {sample_msg[1]['content'][:100]}...")
        print(f"[Assistant]: {sample_msg[2]['content'][:200]}...")

    else:
        print("Fail to format, please check file name.")

    