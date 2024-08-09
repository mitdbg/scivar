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


	with open("candidates_w_bertscore.json", "w") as f:
		json.dump(candidates, f)

