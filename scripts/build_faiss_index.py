import faiss,numpy as np,os,json,pandas as pd,yaml
from sentence_transformers import SentenceTransformer
cfg=yaml.safe_load(open("config.yaml"))
model=SentenceTransformer(cfg["embedding_model"]); os.makedirs("artifacts",exist_ok=True)
def build(csv_path="data/splits/bankfaqs_train.csv",index_path=cfg["faiss_index_path"]):
    df=pd.read_csv(csv_path); texts=df['Question'].astype(str).tolist()
    emb=model.encode(texts,show_progress_bar=True,convert_to_numpy=True)
    dim=emb.shape[1]; index=faiss.IndexFlatIP(dim); faiss.normalize_L2(emb); index.add(emb)
    faiss.write_index(index,index_path)
    meta=[{"question":q,"answer":a} for q,a in zip(df['Question'],df['Answer'])]
    json.dump(meta,open(index_path+".meta.json","w"),indent=2)
if __name__=="__main__": build()
