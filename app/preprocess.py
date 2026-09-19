"""Preprocessing identical to training (Cell 2 + Cell 5). Do not edit one without the other."""
import re, unicodedata

# --- Cell 2: Unicode NFC, zero-width chars removed, whitespace collapsed ---
def clean_text(s):
    s = unicodedata.normalize('NFC', str(s))
    s = s.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
    return re.sub(r'\s+', ' ', s).strip()

# --- Cell 5: the two model-specific views ---
URL_RE   = re.compile(r'https?://\S+|www\.\S+')
PUNCT_RE = re.compile(r'([^\u0980-\u09FFa-zA-Z0-9\s])')
DIGIT_RE = re.compile(r'[0-9\u09E6-\u09EF]+')
URL_TOK, NUM_TOK = ' urltoken ', ' numtoken '

def prep_bert(s):
    s = URL_RE.sub(URL_TOK, str(s))
    return re.sub(r'\s+', ' ', s).strip()

def prep_tfidf(s):
    s = URL_RE.sub(URL_TOK, str(s))
    s = s.replace('\ufe0f', '')
    s = PUNCT_RE.sub(r' \1 ', s)
    s = DIGIT_RE.sub(NUM_TOK, s)
    return re.sub(r'\s+', ' ', s).strip()
