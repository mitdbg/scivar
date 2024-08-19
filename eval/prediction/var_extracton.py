import json
import os
import time

from mitaskem.src.eval.data_quality import get_var_whole_extraction, get_var_whole_tool_extraction, \
    get_var_extract_tool_extraction
def process_json_file(json_file_path, gpt_key, model_name='gpt-3.5-turbo-16k', mode="pure", tool_extract_json=None, gpt_extraction_file=None):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    if mode == "tool":
        with open(tool_extract_json, 'r') as file:
            tool_data = json.load(file)

    elif mode == "merge":
        with open(gpt_extraction_file, 'r') as file:
            gpt_data = json.load(file)
        with open(tool_extract_json, 'r') as file:
            tool_data = json.load(file)

    results = []
    cnt = 0
    for entry in data:
        print(f"Processing entry {cnt}")
        all_text = entry['all_text']
        # count the runtime of the function
        start_time = time.time()
        if mode == "pure":
            extracted_info = get_var_whole_extraction(all_text, gpt_key,model=model_name)
        elif mode == "3shot":
            extracted_info = get_var_whole_extraction(all_text, gpt_key, model=model_name, mode="3shot")
        elif mode == "tool":
            assert tool_data[cnt]['all_text'] == all_text
            tool_extract = tool_data[cnt]['extracted_info']
            # convert object tool_extract to string
            tool_extract_str = json.dumps(tool_extract)
            extracted_info = get_var_whole_tool_extraction(all_text, tool_extract_str, gpt_key, model=model_name)
        elif mode == "merge":
            assert gpt_data[cnt]['all_text'] == all_text
            assert tool_data[cnt]['all_text'] == all_text
            gpt_extract = gpt_data[cnt]['extracted_info']
            tool_extract = tool_data[cnt]['extracted_info']
            tool_extract_str = json.dumps(tool_extract)
            extracted_info = get_var_extract_tool_extraction(gpt_extract, tool_extract_str, gpt_key, model=model_name)
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

    output_file_path = os.path.join(os.path.dirname(json_file_path), mode+'_variable_dataset_'+model_name+'.json')
    with open(output_file_path, 'w') as outfile:
        json.dump(results, outfile, indent=4)

# Example usage
gpt_key = os.environ.get('OPENAI_API_KEY')
# model_name = 'gpt-3.5-turbo-16k'
# model_name = 'gpt-4-1106-preview'
# model_name = 'gpt-4o'
# model_name = 'gpt-4o-mini'
# mode as "pure" or "tool" or "merge"
mode = "3shot"

json_file_path = "askem_variable_dataset.json"
tool_file_path = "variable_dataset_skema.json"

for model_name in ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo-16k', 'gpt-4-1106-preview']:
    if mode == "pure":
        process_json_file(json_file_path, gpt_key, model_name,mode="pure")
    elif mode =="3shot":
        process_json_file(json_file_path, gpt_key, model_name,mode="3shot")
    elif mode == "tool":
        process_json_file(json_file_path, gpt_key, model_name,mode="tool", tool_extract_json=tool_file_path)
    elif mode == "merge":
        gpt_extraction_file = "/Users/chunwei/research/scivar/eval/llms/variable_dataset_"+model_name+".json"
        process_json_file(json_file_path, gpt_key, model_name,mode="merge", gpt_extraction_file=gpt_extraction_file, tool_extract_json=tool_file_path)
