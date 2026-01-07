import csv
import re
from classifier.training.config import DATASET_PATH

def _clean_token(tok: str) -> str:
    tok = tok.strip()
    # Drop trailing non-numeric chars (e.g., 'Z')
    tok = re.sub(r'[^\d\.\-eE\+]+$', '', tok)
    return tok

def load_dataset(path=DATASET_PATH):
    X, y = [], []
    expected_len = None
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader, 1):
            if not row or len(row) < 2:
                continue
            try:
                label = row[0].strip()
                feats = [float(_clean_token(v)) for v in row[1:] if v.strip() != ""]
                if expected_len is None:
                    expected_len = len(feats)  
                if len(feats) != expected_len:
                    continue
                X.append(feats)
                y.append(label)
            except Exception:
                continue
    return X, y