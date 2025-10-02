import yaml,faiss,json,os,time,numpy as np
from sentence_transformers import SentenceTransformer
from naive_rag import retrieve,rerank,generate
cfg=yaml.safe_load(open("config.yaml")); embedder=SentenceTransformer(cfg["embedding_model"])
p_idx,p_meta="artifacts/dkc.index","artifacts/dkc.meta.json"
if os.path.exists(p_idx): index=faiss.read_index(p_idx); meta=json.load(open(p_meta))
else: index=faiss.IndexFlatIP(embedder.get_sentence_embedding_dimension()); meta=[]
def save(): faiss.write_index(index,p_idx); json.dump(meta,open(p_meta,"w"),indent=2)
def lookup(qe,th=0.85):
    faiss.normalize_L2(qe); D,I=index.search(qe,1)
    if I[0][0]!=-1 and D[0][0]>=th: m=meta[I[0][0]]; m["u"]+=1; return m["ans"]
def add(q,a,qe):
    qn=qe.copy(); faiss.normalize_L2(qn); index.add(qn)
    meta.append({"q":q,"ans":a,"e":qn[0].tolist(),"u":1,"t":time.time()}); save()
def answer(q):
    qe=embedder.encode([q],convert_to_numpy=True)
    hit=lookup(qe,cfg["cache_threshold_dkc"])
    if hit: return hit
    ctx=rerank(q,retrieve(q,cfg["top_k_retriever"]),cfg["top_k_rerank"]); ans=generate(q,ctx)
    add(q,ans,qe); return ans
