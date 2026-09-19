# Demo app

Type a Bangla comment and see which of the five classes the fine-tuned BanglaBERT (and the TF-IDF baseline) assigns it to.

## Setup (once)

1. Download `cyberbullying_model.zip` from the Output tab of the Kaggle dev notebook and unzip it so this folder contains `cyberbullying_model/` (with `banglabert/`, `tfidf_logreg.joblib`, `label_map.json`, `examples.csv`, `versions.json`).
2. From this folder:

```
python -m pip install --user uv
python -m uv venv --python 3.12 .venv
python -m uv pip install --python .venv\Scripts\python.exe torch --index-url https://download.pytorch.org/whl/cpu
python -m uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

`uv` fetches its own Python 3.12 (the version the models were trained with on Kaggle), so this works whatever Python is installed system-wide. `scikit-learn` is pinned to 1.6.1 because `tfidf_logreg.joblib` was pickled with it — and 1.6.1 has no wheels for Python 3.14.

## Run

```
.venv\Scripts\python app.py            # then open http://127.0.0.1:7860
.venv\Scripts\python app.py --share    # also prints a temporary public link (valid ~72 h)
```

The first start after installing takes a couple of minutes (first-time imports); later starts are much faster. Each comment takes ~0.1 s on CPU.

Preprocessing in `preprocess.py` is identical to training (Cell 2 + Cell 5); the Kaggle notebook's Cell 11 verifies this on the full test set.
