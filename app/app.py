"""Bangla cyberbullying classifier — demo interface.

Type a Bangla comment; the fine-tuned BanglaBERT and the TF-IDF + Logistic
Regression baseline each return a probability for the five classes.

    python app.py            # http://127.0.0.1:7860
    python app.py --share    # also prints a temporary public link
"""
import argparse, json, os, warnings

import gradio as gr
import joblib
import pandas as pd
import sklearn
import torch
from normalizer import normalize as bn_normalize
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from preprocess import clean_text, prep_bert, prep_tfidf

MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                     'cyberbullying_model'))
if not os.path.isdir(MODEL_DIR):
    raise SystemExit(f'Model folder not found: {MODEL_DIR}\n'
                     f'Unzip cyberbullying_model.zip (Kaggle Output tab) into that folder.')

label2id = json.load(open(os.path.join(MODEL_DIR, 'label_map.json')))['label2id']
LABELS = [l for l, _ in sorted(label2id.items(), key=lambda kv: kv[1])]

trained = json.load(open(os.path.join(MODEL_DIR, 'versions.json')))
if sklearn.__version__ != trained['scikit-learn']:
    warnings.warn(f"scikit-learn {sklearn.__version__} installed, model saved with "
                  f"{trained['scikit-learn']} — install that version if TF-IDF predictions look off.")

tok   = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, 'banglabert'))
model = AutoModelForSequenceClassification.from_pretrained(os.path.join(MODEL_DIR, 'banglabert')).eval()
tfidf = joblib.load(os.path.join(MODEL_DIR, 'tfidf_logreg.joblib'))


@torch.no_grad()
def classify(text):
    text = clean_text(text)
    if not text:
        return None, None
    enc = tok(bn_normalize(prep_bert(text)), truncation=True, max_length=128, return_tensors='pt')
    probs = torch.softmax(model(**enc).logits, -1)[0].tolist()
    bert = {LABELS[i]: p for i, p in enumerate(probs)}

    tp = tfidf['model'].predict_proba(tfidf['vectorizer'].transform([prep_tfidf(text)]))[0]
    tf = {LABELS[int(c)]: float(p) for c, p in zip(tfidf['model'].classes_, tp)}
    return bert, tf


examples = pd.read_csv(os.path.join(MODEL_DIR, 'examples.csv'))['Text'].tolist()

with gr.Blocks(title='Bangla Cyberbullying Classifier') as demo:
    gr.Markdown('## Bangla Cyberbullying Classifier\n'
                'Classes: Neutral · Harassment · Sexual Aggression · Hate Speech · Violent Extremism.  '
                'Test macro-F1 — BanglaBERT **0.836 ± 0.003** (3 seeds), TF-IDF + LogReg **0.812**.')
    inp = gr.Textbox(label='Comment', lines=3, placeholder='এখানে একটি বাংলা মন্তব্য লিখুন…')
    btn = gr.Button('Classify', variant='primary')
    with gr.Row():
        out_bert = gr.Label(label='BanglaBERT (fine-tuned)', num_top_classes=5)
        out_tf   = gr.Label(label='TF-IDF + Logistic Regression', num_top_classes=5)
    gr.Examples(examples=examples, inputs=inp, label='Examples from the test set')
    btn.click(classify, inp, [out_bert, out_tf])
    inp.submit(classify, inp, [out_bert, out_tf])

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--share', action='store_true', help='create a temporary public link')
    demo.launch(share=ap.parse_args().share, inbrowser=True)   # opens the browser tab itself
