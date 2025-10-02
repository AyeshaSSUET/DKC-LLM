import time, numpy as np, torch, json

def now_ms(): return time.perf_counter()*1000

def measure_latency(func, warmup=5):
    lat = []
    for _ in range(warmup): func()
    for _ in range(100):
        t0 = now_ms(); func(); 
        if torch.cuda.is_available(): torch.cuda.synchronize()
        lat.append(now_ms()-t0)
    arr = np.array(lat)
    return {"avg":float(arr.mean()),"p50":float(np.percentile(arr,50)),"p95":float(np.percentile(arr,95)),"p99":float(np.percentile(arr,99)),"raw":lat}

def save_json(path,obj): open(path,"w",encoding="utf-8").write(json.dumps(obj,indent=2))
