import json
import time
from openai import OpenAI
from tqdm import tqdm

input_file = 'data/TISER_train.json'
output_file = 'data/TIGER_train.json'
# new_instructions = "You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries. Follow these steps:\n\n        Step 1. Reason through the problem step by step within the <reasoning> tags.\n        Step 2. Given your previous reasoning, identify relevant temporal events in the given context for answering the given question within <timeline> tags. Assume relations in the context are unidirectional.\n        Step 3. Reflect on your reasoning and the timeline to check for any errors or improvements within the <reflection> tags.\n        Step 4. Make any necessary adjustments based on your reflection. If there is additional reasoning required, go back to Step 1 (reason through the problem step-by-step), otherwise move to the next step (Step 5).\n        Step 5. Provide your final, concise answer within the <answer> tags. If the answer is a number, just output the number nothing else. Otherwise output the entity or event, without any additional comments.\n\n        Important: The <reasoning>, <reflection> and <timeline> sections are for your internal reasoning process. All the reflection and the timeline have to be contained inside the thinking section.\n        Do not use enumerations or lists when writing, use plain text instead such as paragraphs.\n        The response to the query must be entirely contained within the <answer> tags.\n\n        Use the following format for your response:\n        \n        <reasoning>\n        [Your step-by-step reasoning goes here. This is your internal thought process.]\n        <timeline>\n        [Relevant temporal events for answering the given question.]\n        </timeline>\n        <reflection>\n        [Your reflection on your reasoning, checking for errors or improvements]\n        </reflection>\n        [Any adjustments to your thinking based on your reflection]\n        </reasoning>\n        <answer>\n        [Your final, concise answer to the query.]\n        </answer>\n        \n        Question: When did the event (Mike Johnson was born in Kingston, Wyoming) start?\n        \n        Temporal context: (Mike Johnson was born in Kingston, Wyoming) starts at 1929. (Emily Adams was born in Leicester) starts at 1934. (Emily Adams was married to Mike Johnson) starts at 1961. (Mike Johnson was married to Emily Adams) starts at 1961. (Emily Adams died in Willowdale) starts at 1970. (Emily Adams was married to Mike Johnson) ends at 1970. (Mike Johnson was married to Emily Adams) ends at 1970. (Mike Johnson died in Oceanview) starts at 2014"
# new_instructions = f"""You are an AI assistant that uses a Graph-Based Temporal Reasoning approach with reflection to answer queries. Follow these steps:\n\n        Step 1. Reason through the problem step by step within the <reasoning> tags. Explicitly mention constructing a "temporal dependency graph" or "network" to analyze the relationships between events.\n        Step 2. Based on your reasoning, construct a structured temporal graph within <timeline_graph> tags.\n- Output a valid JSON object.\n- The JSON must contain two lists: "events" (nodes) and "relations" (edges).\n- For "relations", use strict Allen's Interval Algebra types: [BEFORE, MEETS, OVERLAPS, STARTS, FINISHES, EQUALS, DURING, CONTAINS].\n- Calculate the precise relationship between events based on their start and end times (e.g., if A:1990-1995 and B:1992-1998, the relation is OVERLAPS).\n        Step 3. Reflect on your graph to check for logical consistency within the <reflection> tags.\n- Perform a "Cycle Check": explicitly verify that there are no logical loops (e.g., A before B, B before A).\n- Verify if the graph structure supports the final answer.\n        Step 4. Make any necessary adjustments based on your reflection. If the graph is invalid, correct it.\n        Step 5. Provide your final, concise answer within the <answer> tags. If the answer is a number, just output the number. Otherwise, output the entity or event without comments.\n\n        Important: The <reasoning>, <timeline_graph>, and <reflection> sections are for your internal reasoning process. The response to the query must be entirely contained within the <answer> tags.\n\n        Use the following format for your response:\n        \n        <reasoning>\n        [Your step-by-step reasoning about constructing the graph goes here.]\n        <timeline_graph>\n        {{"events": [{{"id": "E1", "label": "Event description"}}, {{"id": "E2", "label": "Event description"}}], "relations": [{{"source": "E1", "target": "E2", "relation": "BEFORE"}}, {{"source": "E2", "target": "E3", "relation": "CONTAINS"}}]}}\n        </timeline_graph>\n        <reflection>\n        [Your logic check: "Graph cycle check: Passed/Failed. Logic verification: ..."]\n        </reflection>\n        </reasoning>\n        <answer>\n        [Your final, concise answer to the query.]\n        </answer>\n        \n        Question: {question}\n        \n        Temporal context: {temp_context}"""

def extract_context_split(full_text):
    marker = "Temporal context: "
    try:
        # maxsplit=1 保证即使后面还有这个词（虽然不太可能），也只从第一个切分
        # [1] 取切分后的第二部分，即 marker 之后的内容
        content = full_text.split(marker, 1)[1]
        return content.strip()  # 去除首尾的空格和换行符
    except IndexError:
        return ""


def tiger_data_augmentation(input_path):
    print(f"Loading {input_path} ...")
    
    try:
        with open(input_path, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue

                try:
                    entry = json.loads(line)
                    original_prompt = entry.get('prompt', "")
                    question = entry.get('question', "")
                    temp_context = extract_context_split(original_prompt)
                    system_text = f"""You are an AI assistant that uses a Graph-Based Temporal Reasoning approach with reflection to answer queries. Follow these steps:\n\n        Step 1. Reason through the problem step by step within the <reasoning> tags. Explicitly mention constructing a "temporal dependency graph" or "network" to analyze the relationships between events.\n        Step 2. Based on your reasoning, construct a structured temporal graph within <timeline_graph> tags.\n        - Output a valid JSON object.\n        - The JSON must contain two lists: "events" (nodes) and "relations" (edges).\n        - For "relations", use strict Allen's Interval Algebra types: [BEFORE, MEETS, OVERLAPS, STARTS, FINISHES, EQUALS, DURING, CONTAINS].\n        - Calculate the precise relationship between events based on their start and end times (e.g., if A:1990-1995 and B:1992-1998, the relation is OVERLAPS).\n        Step 3. Reflect on your graph to check for logical consistency within the <reflection> tags.\n        - Perform a "Cycle Check": explicitly verify that there are no logical loops (e.g., A before B, B before A).\n        - Verify if the graph structure supports the final answer.\n        Step 4. Make any necessary adjustments based on your reflection. If the graph is invalid, correct it.\n        Step 5. Provide your final, concise answer within the <answer> tags. If the answer is a number, just output the number. Otherwise, output the entity or event without comments.\n\n        Important: The <reasoning>, <timeline_graph>, and <reflection> sections are for your internal reasoning process. The response to the query must be entirely contained within the <answer> tags.\n\n        Use the following format for your response:\n        \n        <reasoning>\n        [Your step-by-step reasoning about constructing the graph goes here.]\n        </reasoning>\n        <timeline_graph>\n        {{"events": [{{"id": "E1", "label": "Event description"}}, {{"id": "E2", "label": "Event description"}}], "relations": [{{"source": "E1", "target": "E2", "relation": "BEFORE"}}, {{"source": "E2", "target": "E3", "relation": "CONTAINS"}}]}}\n        </timeline_graph>\n        <reflection>\n        [Your logic check: "Graph cycle check: Passed/Failed. Logic verification: ..."]\n        </reflection>\n        <answer>\n        [Your final, concise answer to the query.]\n        </answer>\n        \n        Question: {question}\n        \n        Temporal context: {temp_context}"""
                    user_text = question
                    original_output = entry.get('output', "")
                    print(original_prompt)
                    print('-'*50)
                    print(system_text)
                    print('='*50)
                    if i == 0:
                        break
                except json.JSONDecodeError:
                    print(f"Worning: Fail to read line No.{i+1}.")
                    continue
    except FileNotFoundError:
        print(f"Error: fail to find the file {input_path}.")
        return []
    


if __name__ == "__main__":
    tiger_data_augmentation(input_file)