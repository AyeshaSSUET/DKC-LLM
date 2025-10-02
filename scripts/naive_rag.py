import yaml,faiss,json,numpy as np,torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
cfg=yaml.safe_load(open("config.yaml")); device=cfg["device"] if torch.cuda.is_available() else "cpu"
embedder=SentenceTransformer(cfg["embedding_model"])
index=faiss.read_index(cfg["faiss_index_path"]); meta=json.load(open(cfg["faiss_index_path"]+".meta.json"))
tok=AutoTokenizer.from_pretrained(cfg["generator_model"])
gen=AutoModelForSeq2SeqLM.from_pretrained(cfg["generator_model"]).to(device)
def retrieve(query,k=5):
    q=embedder.encode([query],convert_to_numpy=True); faiss.normalize_L2(q); D,I=index.search(q,k)
    return [meta[i] for i in I[0]]
def rerank(query,cands,k=3):
    q=embedder.encode([query],convert_to_numpy=True); c=embedder.encode([x['question'] for x in cands],convert_to_numpy=True)
    faiss.normalize_L2(q); faiss.normalize_L2(c); sims=(c@q.T).squeeze(); idx=sims.argsort()[::-1][:k]; return [cands[i] for i in idx]
def generate(query,ctxs):
    prompt=query+"\nContext:\n"+"\n".join([c['answer'] for c in ctxs])
    inp=tok(prompt,return_tensors="pt",truncation=True,max_length=512).to(device)
    out=gen.generate(**inp,max_new_tokens=64); return tok.decode(out[0],skip_special_tokens=True)
def answer(q): return generate(q,rerank(q,retrieve(q,cfg["top_k_retriever"]),cfg["top_k_rerank"]))
