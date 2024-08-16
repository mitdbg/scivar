import json


def extract_baseline(json_file, baseline = "3shot_variable_dataset_mistral"):
    with open(json_file, 'r') as file:
        data = json.load(file)

    extracted = []
    cnt = 0
    for entry in data:
        if entry['method'] == baseline:
            entry['pair_id'] = cnt
            extracted.append(entry)
            cnt += 1
    return extracted

if __name__ == "__main__":
    json_file = "../candidates.json"
    extracted = extract_baseline(json_file, baseline = "3shot_variable_dataset_mistral")
    print(f"Extracted {len(extracted)} entries")
    output_file = f"../candidates_mistral_3shot.json"
    with open(output_file, 'w') as outfile:
        json.dump(extracted, outfile, indent=4)
    print(f"Saved to {output_file}")