import json
import os
import time
from datetime import datetime
from vllm import LLM, SamplingParams


def extract_output_text(response):
    if response and isinstance(response, list) and hasattr(response[0], 'outputs'):
        first_request_output = response[0]
        if first_request_output.outputs and isinstance(first_request_output.outputs, list):
            first_completion_output = first_request_output.outputs[0]
            return first_completion_output.text
    return None


def get_var_whole_extraction(text, model, sampling_params, mode="pure", tool_extract=None, gpt_extraction=None):
    print(f"mode: {mode}")
    if mode == "pure":
        prompt = get_var_whole_prompt(text)
    elif mode == "tool":
        assert tool_extract is not None
        prompt = get_var_whole_tool_prompt(text, tool_extract)
    elif mode == "3shot":
        prompt = get_var_whole_prompt(text, prompt_template_path='../prompts/text_var_val_prompt-3shots.txt')

    print(f"Formatted prompt: {prompt}")

    outputs = model.generate([prompt], sampling_params)
    res = extract_output_text(outputs)
    print(f"Output: {res}")
    return res


def get_var_whole_tool_prompt(text, tool):
    text_file = open(os.path.join(os.path.dirname(__file__), '../prompts/text_var_val_tool_prompt.txt'), "r")
    prompt = text_file.read()
    # print(prompt)
    text_file.close()

    prompt = prompt.replace("[TEXT]", text)
    prompt = prompt.replace("[TOOL]", tool)
    # print(prompt)
    return prompt

def get_var_whole_prompt(text, prompt_template_path='../prompts/text_var_val_prompt.txt'):
    # prompt_template_path = '../prompts/text_var_val_prompt.txt'
    print(f"Using prompt template path: {prompt_template_path}")
    with open(prompt_template_path, 'r') as file:
        prompt = file.read()
    prompt = prompt.replace("[TEXT]", text)
    return prompt


# Load the model
model = LLM("/home/gridsan/cliu/hf/Mistral-7B-Instruct-v0.2", dtype="float16")

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
sampling_params = SamplingParams(temperature=0, max_tokens=1024)

# Process each entry in the JSON data
for entry in data:
    if cnt >= 10000:
        break
    print(f"Processing entry {cnt} at {datetime.now()}")
    all_text = entry['all_text']
    # Measure the runtime of the function
    start_time = time.time()
    if mode == "pure":
        extracted_info = get_var_whole_extraction(all_text, model, sampling_params, mode="pure")
    elif mode == "3shot":
        extracted_info = get_var_whole_extraction(all_text, model, sampling_params, mode="3shot")
    elif mode == "tool":
        assert tool_data[cnt]['all_text'] == all_text
        tool_extract = tool_data[cnt]['extracted_info']
        # Convert object tool_extract to string
        tool_extract_str = json.dumps(tool_extract)
        extracted_info = get_var_whole_extraction(all_text, model, sampling_params, mode="tool", tool_extract=tool_extract_str)
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
output_file_path = os.path.join(data_dir, mode+'_variable_dataset_mistral.json')
with open(output_file_path, 'w') as outfile:
    json.dump(results, outfile, indent=4)
print(f"Total records processed: {cnt}")