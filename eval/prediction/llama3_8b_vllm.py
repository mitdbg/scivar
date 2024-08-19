import json
import os
import time
from datetime import datetime
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def extract_output_text(response):
    # Check if the response is not empty and has the expected structure
    if response and isinstance(response, list) and hasattr(response[0], 'outputs'):
        first_request_output = response[0]
        if first_request_output.outputs and isinstance(first_request_output.outputs, list):
            first_completion_output = first_request_output.outputs[0]
            return first_completion_output.text
    return None  # Return None if the structure is not as expected

# Define the function to extract variable information using LLaMA
def get_var_whole_extraction(text, model, tokenizer, mode="pure", tool_extract=None, gpt_extraction_file=None):
    print(f"mode: {mode}")
    if mode == "pure":
        prompt = get_var_whole_prompt(text)
    elif mode == "tool":
        assert tool_extract is not None
        prompt = get_var_whole_tool_prompt(text, tool_extract)
    elif mode == "3shot":
        prompt = get_var_whole_prompt(text, prompt_template_path='../prompts/text_var_val_prompt-3shots.txt')
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"Formatted prompt: {formatted_prompt}")
    output = model.generate(formatted_prompt, SamplingParams(max_tokens=1024, temperature=0))
    res = extract_output_text(output)
    print(f"Output: {res}")
    return res

def get_var_whole_prompt(text, prompt_template_path='../prompts/text_var_val_prompt.txt'):
    # Assuming the prompt template is stored in a text file
    # prompt_template_path = '../prompts/text_var_val_prompt.txt'
    with open(prompt_template_path, 'r') as file:
        prompt = file.read()
    prompt = prompt.replace("[TEXT]", text)
    return prompt

def get_var_whole_tool_prompt(text, tool):
    text_file = open(os.path.join(os.path.dirname(__file__), '../prompts/text_var_val_tool_prompt.txt'), "r")
    prompt = text_file.read()
    # print(prompt)
    text_file.close()

    prompt = prompt.replace("[TEXT]", text)
    prompt = prompt.replace("[TOOL]", tool)
    # print(prompt)
    return prompt

# Load the model and tokenizer
model = LLM("/home/gridsan/cliu/hf/Meta-Llama-3-8B-Instruct", dtype="float16")
tokenizer = AutoTokenizer.from_pretrained("/home/gridsan/cliu/hf/Meta-Llama-3-8B-Instruct")

# Load JSON data from file
data_dir = "/home/gridsan/cliu/mitaskem/mitaskem/src/benchmark/"
json_file_path = os.path.join(data_dir, 'askem_variable_dataset.json')
with open(json_file_path, 'r') as file:
    data = json.load(file)
print(f"Total records: {len(data)}")

tool_file_path = os.path.join(data_dir, 'variable_dataset_skema.json')
with open(tool_file_path, 'r') as file:
    tool_data = json.load(file)

mode = "3shot"

results = []
cnt = 0

# Process each entry in the JSON data
for entry in data:
    if cnt >= 10000:
        break
    print(f"Processing entry {cnt} at {datetime.now()}")
    all_text = entry['all_text']

    # Measure the runtime of the function
    start_time = time.time()
    if mode == "pure":
        extracted_info = get_var_whole_extraction(all_text, model, tokenizer, mode="pure")
    elif mode == "3shot":
        extracted_info = get_var_whole_extraction(all_text, model, tokenizer, mode="3shot")
    elif mode == "tool":
        assert tool_data[cnt]['all_text'] == all_text
        tool_extract = tool_data[cnt]['extracted_info']
        # Convert object tool_extract to string
        tool_extract_str = json.dumps(tool_extract)
        extracted_info = get_var_whole_extraction(all_text, model, tokenizer, mode="tool", tool_extract=tool_extract_str)

    duration = time.time() - start_time
    duration = "{:.2f}".format(duration)
    print(f"Duration: {duration} seconds")

    results.append({
        'id': cnt,
        'all_text': all_text,
        'page': entry['page'],
        'file': entry['file'],
        'duration': duration,
        'extracted_info': extracted_info
    })
    cnt += 1

# Save the results to a JSON file
output_file_path = os.path.join(data_dir, mode+'_variable_dataset_llama.json')
with open(output_file_path, 'w') as outfile:
    json.dump(results, outfile, indent=4)

print(f"Total records processed: {cnt}")