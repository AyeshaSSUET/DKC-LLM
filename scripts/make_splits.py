import pandas as pd, json, random, os
from sklearn.model_selection import train_test_split
random.seed(42)
def split_bankfaqs(path="data/bankfaqs.csv",outdir="data/splits"):
    os.makedirs(outdir,exist_ok=True)
    df=pd.read_csv(path)
    tr,te=train_test_split(df,test_size=0.2,random_state=42,shuffle=True)
    tr.to_csv(f"{outdir}/bankfaqs_train.csv",index=False)
    te.to_csv(f"{outdir}/bankfaqs_test.csv",index=False)
def split_hotpotqa(path="data/hotpotqa.jsonl",outdir="data/splits"):
    os.makedirs(outdir,exist_ok=True)
    lines=[json.loads(l) for l in open(path,'r',encoding='utf-8')]
    random.shuffle(lines); split=int(len(lines)*0.8)
    tr,te=lines[:split],lines[split:]
    open(f"{outdir}/hotpotqa_train.jsonl","w").writelines([json.dumps(r)+"\n" for r in tr])
    open(f"{outdir}/hotpotqa_test.jsonl","w").writelines([json.dumps(r)+"\n" for r in te])
if __name__=="__main__": split_bankfaqs()
