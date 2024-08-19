import json


with open("/Users/chunwei/research/scivar/candidates_mistral_3shot.json", 'r') as f:
    data = json.load(f)

# Function to create the formatted output
def create_formatted_output(data, start_id=0, end_id=50000):
    output = []
    custom_id = 0
    print(f"Total number of items: {len(data)}")

    for item in data:
        if custom_id <start_id:
            custom_id += 1
            continue
        if item['ann_type'] == 'var desc':
            prompt = f"The following pair of text describe a variable and its description.\n'{item['annotation']}'\n'{item['prediction']}'\nPlease check if they mean the same. Answer y or n"
        elif item['ann_type'] == 'var val':
            prompt = f"The following pair of text describe a variable and its value.\n'{item['annotation']}'\n'{item['prediction']}'\nPlease check if they mean the same. Answer y or n"
        else:
            print(f"Skipping item with ann_type: {item['ann_type']}")
            continue

        # item['pair_id'] = custom_id

        formatted_item = {
            "custom_id": f"request_{custom_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4-turbo",
                "messages": [
                    {"role": "system", "content": "You are a human evaluator."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1
            }
        }
        output.append(json.dumps(formatted_item))
        custom_id += 1
        if custom_id == end_id:
            break
    # write the candidate json back with the pair_id
    # with open("/Users/chunwei/research/scivar/candidates_id_3shot.json", 'w') as f:
    #     json.dump(data, f, indent=4)
    return output


#  split the data into batches of 50k
for i in range(0, len(data), 50000):
    start_id = i
    end_id = start_id + 50000
    # Generate the formatted output
    formatted_output = create_formatted_output(data, start_id=start_id, end_id=end_id)

    # Write the formatted output to a jsonl file
    with open(f"3shot_candidates_formatted_batch-mistral-3shot-{start_id}-{end_id}.jsonl", 'w') as f:
        for item in formatted_output:
            f.write(item + '\n')