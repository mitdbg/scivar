import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def read_jsonl_file(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]


def parse():
    basepath = "/Users/chunwei/research/scivar/"
    # Paths to the files
    candidates_path = f'{basepath}candidates_id-pz.json'
    results_path = f'{basepath}candidate-id-results-pz.jsonl'

    # Read the data from files
    candidates_data = read_json_file(candidates_path)
    results_data = read_jsonl_file(results_path)

    # Ensure both files have the same number of entries
    if len(candidates_data) != len(results_data):
        raise ValueError("The number of entries in the files do not match.")
    res = {}
    res['model'] = results_data[0]['response']['body']['model']
    res['total'] = len(candidates_data)
    res['results'] = {}
    # Iterate through both files simultaneously
    for candidate, result in zip(candidates_data, results_data):
        print("Candidate ID:", candidate['pair_id'], "Result ID:", result['custom_id'].split('_')[-1])
        assert(candidate['pair_id'] == int(result['custom_id'].split('_')[-1]))
        if candidate['method'] not in res['results']:
            res['results'][candidate['method']] = {}
        if candidate['text_id'] not in res['results'][candidate['method']]:
            res['results'][candidate['method']][candidate['text_id']] = {"annotation_set":{}, "prediction_set":{}, "match_set":[], "others":[]}
        cur_text = res['results'][candidate['method']][candidate['text_id']]
        if candidate['annotation'] not in cur_text['annotation_set']:
            cur_text['annotation_set'][candidate['annotation']] = []
        if candidate['prediction'] not in cur_text['prediction_set']:
            cur_text['prediction_set'][candidate['prediction']] = []

        if result['response']['body']['choices'][0]['message']['content'] == 'y':
            cur_text['annotation_set'][candidate['annotation']].append(int(result['custom_id'].split('_')[-1]))
            cur_text['prediction_set'][candidate['prediction']].append(int(result['custom_id'].split('_')[-1]))
            cur_text['match_set'].append(int(result['custom_id'].split('_')[-1]))
        elif result['response']['body']['choices'][0]['message']['content'] != 'n':
            cur_text['others'].append(int(result['custom_id'].split('_')[-1]))



    # Write the results to a file
    with open('300k/results_stats.json', 'w') as file:
        json.dump(res, file, indent=4)


# with task input as "var val" or "var desc"
def parse_task(task):
    basepath = "/Users/chunwei/research/scivar/"
    # Paths to the files
    candidates_path = f'{basepath}candidates_id-pz.json'
    results_path = f'{basepath}candidate-id-results-pz.jsonl'

    # Read the data from files
    candidates_data = read_json_file(candidates_path)
    results_data = read_jsonl_file(results_path)

    # Ensure both files have the same number of entries
    if len(candidates_data) != len(results_data):
        raise ValueError("The number of entries in the files do not match.")
    res = {}
    res['model'] = results_data[0]['response']['body']['model']
    res['total'] = len(candidates_data)
    res['results'] = {}
    # Iterate through both files simultaneously
    for candidate, result in zip(candidates_data, results_data):
        if candidate['ann_type'] != task:
            continue
        print("Candidate ID:", candidate['pair_id'], "Result ID:", result['custom_id'].split('_')[-1])
        assert(candidate['pair_id'] == int(result['custom_id'].split('_')[-1]))
        if candidate['method'] not in res['results']:
            res['results'][candidate['method']] = {}
        if candidate['text_id'] not in res['results'][candidate['method']]:
            res['results'][candidate['method']][candidate['text_id']] = {"annotation_set":{}, "prediction_set":{}, "match_set":[], "others":[]}
        cur_text = res['results'][candidate['method']][candidate['text_id']]
        if candidate['annotation'] not in cur_text['annotation_set']:
            cur_text['annotation_set'][candidate['annotation']] = []
        if candidate['prediction'] not in cur_text['prediction_set']:
            cur_text['prediction_set'][candidate['prediction']] = []

        if result['response']['body']['choices'][0]['message']['content'] == 'y':
            cur_text['annotation_set'][candidate['annotation']].append(int(result['custom_id'].split('_')[-1]))
            cur_text['prediction_set'][candidate['prediction']].append(int(result['custom_id'].split('_')[-1]))
            cur_text['match_set'].append(int(result['custom_id'].split('_')[-1]))
        elif result['response']['body']['choices'][0]['message']['content'] != 'n':
            cur_text['others'].append(int(result['custom_id'].split('_')[-1]))



    # Write the results to a file
    with open(f'300k/results_stats_{task}.json', 'w') as file:
        json.dump(res, file, indent=4)


def calculate_f1(json_file_path, task=""):
    # Load the JSON data from the file
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Navigate through the nested 'results' dictionary
    for dataset_key, dataset_value in data['results'].items():
        for record_key, record_value in dataset_value.items():
            # Initialize counts
            gd = len(record_value['annotation_set'])
            tgd = sum(1 for _, v in record_value['annotation_set'].items() if v)
            pred = len(record_value['prediction_set'])
            tpred = sum(1 for _, v in record_value['prediction_set'].items() if v)

            # Calculate recall and precision
            recall = tgd / gd if gd != 0 else 0
            precision = tpred / pred if pred != 0 else 0

            # Update the record with new fields
            record_value['gd'] = gd
            record_value['tgd'] = tgd
            record_value['pred'] = pred
            record_value['tpred'] = tpred
            record_value['recall'] = recall
            record_value['precision'] = precision
            record_value['f1'] = 2 * (recall * precision) / (recall + precision) if recall + precision != 0 else 0

    if task=="":
        # Save the modified data back to the JSON file or another file
        with open('300k/f1_results_stats.json', 'w') as file:
            json.dump(data, file, indent=4)
    else:
        with open(f'300k/f1_results_stats_{task}.json', 'w') as file:
            json.dump(data, file, indent=4)

def cleaning(file_path):
    import re

    with open(file_path, 'r') as file:
        text = file.read()


    # Regex pattern to find and replace
    pattern = r'("prediction":\s*")\d+\.\s'

    # Replace the matching part with the captured group
    # The captured group is the part of the regex inside the parentheses, which is "prediction": "
    modified_text = re.sub(pattern, r'\1', text)

    # Write the modified text back to the file
    with open(file_path, 'w') as file:
        file.write(modified_text)

def f1_violin_plot(file_path, task=""):
    # Load JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Dictionary mapping the full model names to concise labels
    model_name_mapping = {
        'pure_variable_dataset_gpt-3.5-turbo-16k': 'pure_GPT3.5T',
        'tool_variable_dataset_gpt-3.5-turbo-16k': 'tool_GPT3.5T',
        'pure_variable_dataset_gpt-4-1106-preview': 'pure_GPT4T',
        'tool_variable_dataset_gpt-4-1106-preview': 'tool_GPT4T',
        'pure_variable_dataset_gpt-4o': 'pure_GPT4o',
        'tool_variable_dataset_gpt-4o': 'tool_GPT4o',
        'pure_variable_dataset_gpt-4o-mini': 'pure_GPT4o-mini',
        'tool_variable_dataset_gpt-4o-mini': 'tool_GPT4o-mini',
        'pure_variable_dataset_llama': 'pure_llama',
        'tool_variable_dataset_llama': 'tool_llama',
        'pure_variable_dataset_mistral': 'pure_mistral',
        'tool_variable_dataset_mistral': 'tool_mistral',
        'rules_extractions': 'rules',
        'variable_dataset_pz_formatted': 'Palimpzest',
        '3shot_variable_dataset_gpt-3.5-turbo-16k': '3shot_GPT3.5T',
        '3shot_variable_dataset_gpt-4-1106-preview': '3shot_GPT4T',
        '3shot_variable_dataset_gpt-4o': '3shot_GPT4o',
        '3shot_variable_dataset_gpt-4o-mini': '3shot_GPT4o-mini',
        '3shot_variable_dataset_llama': '3shot_llama',
        '3shot_variable_dataset_mistral': '3shot_mistral'
    }

    # Extracting data into a DataFrame
    rows = []
    for model, results in data['results'].items():
        for idx, scores in results.items():
            row = {
                'Model': model_name_mapping[model],  # Map model names
                'Config': idx,
                'Recall': scores['recall'],
                'Precision': scores['precision'],
                'F1': scores['f1']
            }
            # if any numbers are negative OR greater than 1.0, through an error
            if any(score < 0 or score > 1.0 for score in [scores['recall'], scores['precision'], scores['f1']]):
                raise ValueError("Invalid score detected.")
            rows.append(row)

    df = pd.DataFrame(rows)

    # Define the desired order of the models
    desired_order = [
        "pure_GPT3.5T", "tool_GPT3.5T", "3shot_GPT3.5T",
        "pure_GPT4T", "tool_GPT4T", "3shot_GPT4T",
        "pure_GPT4o", "tool_GPT4o", "3shot_GPT4o",
        "pure_GPT4o-mini", "tool_GPT4o-mini", "3shot_GPT4o-mini",
        "pure_llama", "tool_llama", "3shot_llama",
        "pure_mistral", "tool_mistral", "3shot_mistral",
        "rules", "Palimpzest"
    ]

    # Create a categorical type with the desired order and assign it to the 'Model' column
    model_type = pd.CategoricalDtype(categories=desired_order, ordered=True)
    df['Model'] = df['Model'].astype(model_type)

    # Sorting by model name
    df.sort_values('Model', inplace=True)

    # Plotting Recall
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Model', y='Recall', data=df, cut=0)
    point_plot = sns.pointplot(x='Model', y='Recall', data=df, color='red', scale=0.5, ci=None, markers='D', linestyles="")
    plt.title('Recall Scores by Model Configuration')
    plt.xlabel('Model Configuration')
    plt.xticks(rotation=30)
    plt.ylabel('Recall')
    plt.ylim(0, 1)

    # Annotate each point with its value
    for x, y in zip(range(len(df['Model'].unique())), df.groupby('Model')['Recall'].mean()):
        plt.text(x, y, f'{y:.3f}', color='red', ha='center', va='bottom')

    plt.tight_layout()  # Adjust layout to make room for label
    plt.savefig(f'300k/recall_scores_violin{task}.pdf')

    # Plotting Precision
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Model', y='Precision', data=df, cut=0)
    point_plot = sns.pointplot(x='Model', y='Precision', data=df, color='red', scale=0.5, ci=None, markers='D',
                               linestyles="")
    plt.title('Precision Scores by Model Configuration')
    plt.xlabel('Model Configuration')
    plt.xticks(rotation=30)
    plt.ylabel('Precision')
    plt.ylim(0, 1)

    # Annotate each point with its value for Precision
    for x, y in zip(range(len(df['Model'].unique())), df.groupby('Model')['Precision'].mean()):
        plt.text(x, y, f'{y:.3f}', color='red', ha='center', va='bottom')

    plt.tight_layout()  # Adjust layout to make room for label
    plt.savefig(f'300k/precision_scores_violin{task}.pdf')

    # Plotting F1 Score
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Model', y='F1', data=df, cut=0)
    point_plot = sns.pointplot(x='Model', y='F1', data=df, color='red', scale=0.5, ci=None, markers='D', linestyles="")
    plt.title('F1 Scores by Model Configuration')
    plt.xlabel('Model Configuration')
    plt.xticks(rotation=30)
    plt.ylabel('F1')
    plt.ylim(0, 1)

    # Annotate each point with its value for F1 Score
    for x, y in zip(range(len(df['Model'].unique())), df.groupby('Model')['F1'].mean()):
        plt.text(x, y, f'{y:.3f}', color='red', ha='center', va='bottom')

    plt.tight_layout()  # Adjust layout to make room for label
    plt.savefig(f'300k/f1_scores_violin{task}.pdf')

    # Save the average precision, recall, F1 score for each model into a CSV file
    csvdf = df.groupby('Model')[['Recall', 'Precision', 'F1']].mean()
    csvdf = csvdf.round(3)
    csvdf.to_csv(f'300k/average_scores{task}.csv')


if __name__ == "__main__":
    parse()
    calculate_f1('300k/results_stats.json')
    f1_violin_plot('300k/f1_results_stats.json')



    # parse and calculate f1 for var desc
    task = "var val"
    parse_task(task)
    calculate_f1(f'300k/results_stats_{task}.json', task=task)
    f1_violin_plot(f'300k/f1_results_stats_{task}.json', task)

    task = "var desc"
    parse_task(task)
    calculate_f1(f'300k/results_stats_{task}.json', task=task)
    f1_violin_plot(f'300k/f1_results_stats_{task}.json', task)



    # cleaning("/Users/chunwei/research/scivar/candidates_id-PZ.json")
