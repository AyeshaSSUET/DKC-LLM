import sys,pandas as pd,numpy as np,yaml,time
from utils import now_ms,save_json
from naive_rag import answer as naive
from vc_rag import answer as vc
from dkc_llm import answer as dkc
cfg=yaml.safe_load(open("config.yaml"))
def run(func,queries):
    lat=[]; res=[]
    for q in queries:
        t0=now_ms(); r=func(q); lat.append(now_ms()-t0); res.append({"q":q,"a":r})
    arr=np.array(lat); stats={"avg":float(arr.mean()),"p50":float(np.percentile(arr,50)),"p95":float(np.percentile(arr,95)),"p99":float(np.percentile(arr,99))}
    return stats,res
if __name__=="__main__":
    pipe=sys.argv[1] if len(sys.argv)>1 else "dkc"
    df=pd.read_csv("data/splits/bankfaqs_test.csv"); queries=df['Question'].astype(str).tolist()[:200]
    f={"naive":naive,"vc":vc,"dkc":dkc}[pipe]; stats,res=run(f,queries)
    print("Stats:",stats); save_json(f"artifacts/bench_{pipe}.json",{"cfg":cfg,"stats":stats})
