# DKC-LLM Framework Reproducibility Artifacts

This repository contains reproducible code, configs, and benchmark harnesses for Naïve RAG, VC-RAG, and the proposed DKC-LLM framework.

## Datasets
- BankFAQs: [Kaggle link](https://www.kaggle.com/datasets/somanathkshirasagar/bankfaqs)
- HotpotQA: [Kaggle link](https://www.kaggle.com/datasets/jeromeblanchet/hotpotqa-question-answering-dataset)

Download and place them in the `data/` folder.

## Usage
1. Install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Generate splits:
   ```
   python scripts/make_splits.py
   ```
3. Build FAISS index:
   ```
   python scripts/build_faiss_index.py
   ```
4. Run pipeline (example: DKC):
   ```
   python scripts/benchmark.py dkc
   ```

Artifacts including configs, dataset splits, FAISS index, and benchmark logs are saved under `artifacts/`.
