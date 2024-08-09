from nltk.tokenize import word_tokenize
from collections import Counter
from itertools import chain
from pathlib import Path
import json, copy, re
import pandas as pd
from tqdm import tqdm
from evaluate import load

pattern = re.compile(r"[\.,=\(\)]")
vignette = re.compile(f"^\d+[\.,] ")
bertscore = load("bertscore", device="mps")

def token_level_entries(ref, pred):
	pred = vignette.sub("", pred)
	ref = pattern.sub("", ref).lower()
	pred= pattern.sub("", pred).lower()
	ref = Counter(word_tokenize(ref))
	pred = Counter(word_tokenize(pred))

	tp = sum(min(ref.get(w, 0), pred[w]) for w in pred)
	fp = sum(-min(0, ref.get(w, 0)-pred[w]) for w in pred)
	fn = sum(max(0, ref[w]-pred.get(w,0)) for w in ref)

	p = tp/(tp+fp) if tp+fp > 0 else 0.
	r = tp/(tp+fn) if tp+fn > 0 else 0.
	f1 = 2*(p*r/(p+r)) if p+r > 0 else 0.

	


	return p, r, f1

def process_extractions(extractions):
	if isinstance(extractions, str):
		# This is Chunwei's format
		# extractions = extractions.replace(r"\\\\", r"\\")
		ret = list()
		lines = extractions.split("\n")
		for l in lines:
			if l != "--- | --- | ---":
				tokens = l.split(" | ")
				if len(tokens) == 3:
					var = tokens[0]
					desc = tokens[1]
					val = tokens[2]
					if val.lower() not in {"none", "null"}:
						ret.append([var, val, "var val"])
					if desc.lower() not in {"none", "null"}:
						ret.append([var, desc, "var desc"])

		return ret

	else:
		return extractions

def process_block(annotations, block, eval_type):
	block = copy.deepcopy(block)

	extractions = {
		"descs": list(),
		"vals": list()
	}

	extracted_info = process_extractions(block['extracted_info'])

	for extraction in extracted_info:
		var, val, type_ = extraction

		if type_ == "var desc":
			key = 'descs'
		elif type_ == "var val":
			key = 'vals'

		if isinstance(val, str) and val.lower() in {"null", "none"}:
			val = ""

		extractions[key].append(f"{var} {val}")


	scores = list()
	used_predictions = set()
	pair_wise_candidates = list()
	for annotation in annotations:
		ref, type_ = annotation[2], annotation[3]

		if type_ == "var desc":
			preds = extractions['descs']
		elif type_ == "var val":
			preds = extractions['vals']

		max_p, max_r, max_f1 = 0., 0., 0.
		pp = None

		for pred in preds:
			pair_wise_candidates.append({"text_id": block['id'], "annotation":ref, "ann_type": type_, "prediction":pred})
			p, r, f1 = token_level_entries(ref, pred)
			if f1 > max_f1:
				max_p, max_r, max_f1 = p, r, f1
				if f1 > 0:
					pp = pred
			

		if pp:
			used_predictions.add(pp)

		scores.append({"ref":ref, "pred":pp, "p":max_p, "r":max_r, "f1":max_f1})

	# Add the over predictions
	for pred in chain.from_iterable(extractions.values()):
		if pred not in used_predictions:
			scores.append({"ref":None, "pred":pred, "p":0., "r":0., "f1":0.})
	
	block['scores'] = scores
	block['pairs'] = pair_wise_candidates


	return block


if __name__ == "__main__":
	# p, r, f = token_level_entries("susceptibles (S)", "susceptibles S")
	# print(f"P:{p}\tR:{r}\tF1:{f}")


	with open('benchmark/chunkset/askem_variable_dataset.json') as f:
		ds = json.load(f)
		all_annotations = {b['all_text']:b['annotations'] for b in ds}


	rows = []
	all_candidates = []
	eval = "bert"
	for path in tqdm(Path("eval").glob("*/noinfer/*.json"), desc="Computing scores"):
		

			with path.open() as f:
				data = json.load(f)
			method = path.stem

			for block in tqdm(data, desc="Processing blocks"):
				
				key = block["all_text"]
				annotations = all_annotations[key]
				try:
					b = process_block(annotations, block, eval_type=eval)
					b['annotations'] = annotations
					all_candidates.extend([{"method": method, **p} for p in b['pairs']])
					for s in b["scores"]:
						row = {
							'method': method,
							'text_id': block['id'],
							'doc':block['file'],
							**s
						}
						rows.append(row)
				except Exception as e:
					print(e, path.name)

	# Compute BERT scores
	references, predictions = list(), list()
	for candidate in all_candidates:
		references.append(candidate["annotation"])
		predictions.append(candidate["prediction"])
		
	# bscores = bertscore.compute(references=references, predictions=predictions, lang="en")


	with open("candidates.json", 'w') as f:
		json.dump(all_candidates, f, indent=2)

	frame = pd.DataFrame(rows)
	frame.to_csv(f"results_token.csv")
	# stats = frame.groupby("method")[['p','r','f1']].agg("mean")
	# # Percentage of completely missed 
	# frame.groupby("method").agg(lambda )
	# print(stats)
