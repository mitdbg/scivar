import pandas as pd
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", device="mps")

def sentence_similarity(refs, preds):
	r = model.encode(refs)
	p = model.encode(preds)

	similarities = model.similarity_pairwise(r, p)
	
	return similarities

if __name__ == "__main__":
	path = "results_bertscore.csv"
	frame = pd.read_csv(path)
	refs = frame.ref.fillna(value="")
	preds = frame.pred.fillna(value="")

	similarities = sentence_similarity(refs, preds)

	frame['cos_sim'] = similarities
	frame.loc[pd.isna(frame.ref) | pd.isna(frame.pred), ["cos_sim"]] = None
	frame.to_csv(path)
	print(frame)