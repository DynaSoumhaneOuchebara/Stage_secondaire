"""
Dataset preparation script — run ONCE before the internship.

Downloads the Kaggle phishing-email dataset and produces a clean
CSV that the intern will use throughout the week.

Prerequisites
-------------
pip install kagglehub pandas scikit-learn

Usage
-----
python prepare_dataset.py
"""

import pathlib
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = pathlib.Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ── 1. Download from Kaggle ────────────────────────────────────────────
import kagglehub

dataset_path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
print(f"Dataset downloaded to: {dataset_path}")

csv_candidates = list(pathlib.Path(dataset_path).rglob("*.csv"))
print(f"CSV files found: {csv_candidates}")

# The main file is usually phishing_email.csv (the largest one)
csv_path = max(csv_candidates, key=lambda p: p.stat().st_size)
print(f"Using: {csv_path}")

# ── 2. Load and inspect ────────────────────────────────────────────────
df_raw = pd.read_csv(csv_path)
print(f"\nRaw shape: {df_raw.shape}")
print(f"Columns  : {list(df_raw.columns)}")
print(f"Label distribution:\n{df_raw.iloc[:, -1].value_counts()}\n")

# ── 3. Normalise column names ──────────────────────────────────────────
# The dataset typically has "Email Text" and "Email Type".
# We rename to simple, lowercase names.
rename_map = {}
for col in df_raw.columns:
    low = col.strip().lower()
    if "text" in low or "body" in low or "content" in low or "message" in low:
        rename_map[col] = "text"
    elif "type" in low or "label" in low or "class" in low:
        rename_map[col] = "label"

df_raw.rename(columns=rename_map, inplace=True)

if "text" not in df_raw.columns or "label" not in df_raw.columns:
    # Fallback: assume first col is text, last col is label
    cols = list(df_raw.columns)
    df_raw.rename(columns={cols[0]: "text", cols[-1]: "label"}, inplace=True)

# Keep only the two columns we need
if "Unnamed: 0" in df_raw.columns:
    df_raw.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
df = df_raw[["text", "label"]].copy()

# ── 4. Clean labels → binary 0 / 1 ────────────────────────────────────
label_str = df["label"].astype(str).str.strip().str.lower()
phishing_keywords = ["phishing", "spam", "1", "malicious"]
df["label"] = label_str.apply(
    lambda x: 1 if any(k in x for k in phishing_keywords) else 0
)
print(f"After label cleanup:\n{df['label'].value_counts().rename({0: 'safe', 1: 'phishing'})}\n")

# ── 5. Drop nulls and duplicates ──────────────────────────────────────
df.dropna(subset=["text"], inplace=True)
df.drop_duplicates(subset=["text"], inplace=True)
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"].str.len() > 20]  # drop very short junk rows

print(f"After cleaning: {len(df)} rows")

# ── 6. Shuffle ────────────────────────────────────────────────────────
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Final dataset: {len(df)} rows  "
      f"(safe={sum(df.label==0)}, phishing={sum(df.label==1)})")

# ── 7. Train / val / test split  (70 / 15 / 15) ──────────────────────
train, tmp = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
val, test = train_test_split(tmp, test_size=0.50, random_state=42, stratify=tmp["label"])

for name, split in [("train", train), ("val", val), ("test", test)]:
    path = DATA_DIR / f"{name}.csv"
    split.to_csv(path, index=False)
    print(f"  {name:5s}: {len(split):>5d} rows  →  {path}")

# Also save the full cleaned set (useful for exploration)
full_path = DATA_DIR / "emails_clean.csv"
df.to_csv(full_path, index=False)
print(f"\n  full : {len(df):>5d} rows  →  {full_path}")

print("\n✓ Dataset preparation complete. The intern is ready to go!")
