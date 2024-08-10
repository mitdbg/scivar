from nltk.tokenize import word_tokenize
from collections import Counter
from itertools import chain
from pathlib import Path
import json, copy, re
import pandas as pd
from tqdm import tqdm
from evaluate import load
import itertools as it

bertscore = load("bertscore")

def take(n, iterable):
    "Return first n items of the iterable as a list."
    return list(it.islice(iterable, n))

if __name__ == "__main__":

	output_path = Path("candidates_w_bertscore.json")

	if not output_path.exists():

		with open('candidates.json') as f:
			candidates = json.load(f)

		iter_candidates = iter(candidates)
		n = 10000
		block = take(n, iter_candidates)
		tick = tqdm(desc="Processing", unit="pairs", total=len(candidates))
		while block:
			refs, preds = list(), list()
			for b in block:
				refs.append(b['annotation'])
				preds.append(b['prediction'])
			ss = bertscore.compute(references=refs, predictions=preds, lang='en', batch_size=10000, device="mps", model_type="microsoft/deberta-large-mnli")
			for b, p, r, f1 in zip(block, ss['precision'], ss['recall'], ss['f1']):
				b['bert_p'] = p
				b['bert_r'] = r
				b['bert_f1'] = f1

			tick.update(n)
			block = take(n, iter_candidates)


		with output_path.open("w") as f:
			json.dump(candidates, f)

	else:
		
		with output_path.open() as f:
			candidates = json.load(f)

	# Now aggregate the results
	grouped = it.groupby(tqdm(candidates, desc="Matching predictions"), lambda b: (b['method'], b['text_id'], b['annotation']))
	results = list()
	unused_preds = set()
	for (method, text_id, annotation), blocks in grouped:
		
		item = {
			"method": method,
			"text_id": text_id,
			"ref":annotation,
			"pred":None,
			
		}

		p = 0.
		r = 0.
		f1 = 0.
		max_f1 = 0.
		pred = None

		
		for block in blocks:
			pp = block["prediction"]
			unused_preds.add((method, text_id, pp))
			if block["bert_f1"] > max_f1:
				p = block['bert_p']
				r = block['bert_r']
				f1 = block['bert_f1']
				pred = pp
				max_f1 = f1

		if (method, text_id, pred) in unused_preds:
			unused_preds.remove((method, text_id, pred))


		item['pred'] = pred
		item['p'] = p
		item['r'] = r
		item['f1'] = f1

		results.append(item)

	for (method, text_id, pred) in unused_preds:
		results.append({
			"method":method,
			"text_id":text_id,
			"ref":None,
			"pred":pred,
			"p":0.,
			"r":0.,
			"f1":0.
		})

	frame = pd.DataFrame(results)
	frame.to_csv("results_bertscore.csv")



