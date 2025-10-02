import yaml,json,os,numpy as np
from sentence_transformers import SentenceTransformer
from naive_rag import retrieve,rerank,generate,embedder
cfg=yaml.safe_load(open("config.yaml")); path="artifacts/vc_cache.json"; cache=json.load(open(path)) if os.path.exists(path) else {}
def find(q):
    qe=embedder.encode([q],convert_to_numpy=True)
    for k,v in cache.items():
        ce=np.array(v["emb"]); qe/=np.linalg.norm(qe)+1e-12; ce/=np.linalg.norm(ce)+1e-12
        if float(qe@ce.T)>=cfg["cache_threshold_vc_rag"]: return v["ans"]
    return None
def answer(q):
    hit=find(q); 
    if hit: return hit
    ctx=rerank(q,retrieve(q,cfg["top_k_retriever"]),cfg["top_k_rerank"]); ans=generate(q,ctx)
    cache[q[:200]]={"q":q,"ans":ans,"emb":embedder.encode([q],convert_to_numpy=True)[0].tolist()}
    json.dump(cache,open(path,"w"),indent=2); return ans
