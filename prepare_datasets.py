"""
Dataset preparation script — run ONCE before the internship.

Downloads the Kaggle phishing-email dataset and produces:
  - data_exploration/  →  SpamAssasin.csv (cleaner text, for notebooks 00 & 01)
  - data_training/     →  phishing_email.csv (larger, for notebooks 02 & 03)

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

BASE_DIR = pathlib.Path(__file__).parent

# ── 1. Download from Kaggle ────────────────────────────────────────────
import kagglehub

dataset_path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
print(f"Dataset downloaded to: {dataset_path}")

csv_candidates = list(pathlib.Path(dataset_path).rglob("*.csv"))
print(f"CSV files found: {[p.name for p in csv_candidates]}")


# ═══════════════════════════════════════════════════════════════════════
#  Helper: clean any of the CSVs into a standardised (text, label) format
# ═══════════════════════════════════════════════════════════════════════

def clean_dataset(csv_path):
    """Load a CSV and return a clean DataFrame with columns [text, label]."""
    print(f"\n{'─'*60}")
    print(f"  Processing: {csv_path.name}")
    print(f"{'─'*60}")

    df_raw = pd.read_csv(csv_path)
    print(f"  Raw shape : {df_raw.shape}")
    print(f"  Columns   : {list(df_raw.columns)}")

    # ── Normalise column names ─────────────────────────────────────
    rename_map = {}
    for col in df_raw.columns:
        low = col.strip().lower()
        if "text" in low or "body" in low or "content" in low or "message" in low:
            rename_map[col] = "text"
        elif "type" in low or "label" in low or "class" in low:
            rename_map[col] = "label"

    df_raw.rename(columns=rename_map, inplace=True)

    if "text" not in df_raw.columns or "label" not in df_raw.columns:
        cols = list(df_raw.columns)
        df_raw.rename(columns={cols[0]: "text", cols[-1]: "label"}, inplace=True)

    # Drop index column if present
    if "Unnamed: 0" in df_raw.columns:
        df_raw.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
    df = df_raw[["text", "label"]].copy()

    # ── Clean labels → binary 0 / 1 ───────────────────────────────
    label_str = df["label"].astype(str).str.strip().str.lower()
    phishing_keywords = ["phishing", "spam", "1", "malicious"]
    df["label"] = label_str.apply(
        lambda x: 1 if any(k in x for k in phishing_keywords) else 0
    )

    # ── Drop nulls, duplicates, junk rows ──────────────────────────
    df.dropna(subset=["text"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 20]

    # ── Shuffle ────────────────────────────────────────────────────
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"  After cleaning: {len(df)} rows  "
          f"(safe={sum(df.label==0)}, phishing={sum(df.label==1)})")
    return df


# ═══════════════════════════════════════════════════════════════════════
#  A. SpamAssasin dataset  →  data_exploration/  (notebooks 00 & 01)
# ═══════════════════════════════════════════════════════════════════════

spamassasin_path = None
for p in csv_candidates:
    if "spamassasin" in p.name.lower() or "spamassassin" in p.name.lower():
        spamassasin_path = p
        break

if spamassasin_path is None:
    raise FileNotFoundError(
        f"Could not find SpamAssasin.csv among: {[p.name for p in csv_candidates]}"
    )

EXPLORE_DIR = BASE_DIR / "data_exploration"
EXPLORE_DIR.mkdir(exist_ok=True)

df_spamassasin = clean_dataset(spamassasin_path)

# Train / val / test split (70 / 15 / 15)
train_sa, tmp_sa = train_test_split(
    df_spamassasin, test_size=0.30, random_state=42, stratify=df_spamassasin["label"]
)
val_sa, test_sa = train_test_split(
    tmp_sa, test_size=0.50, random_state=42, stratify=tmp_sa["label"]
)

for name, split in [("train", train_sa), ("val", val_sa), ("test", test_sa)]:
    path = EXPLORE_DIR / f"{name}.csv"
    split.to_csv(path, index=False)
    print(f"  {name:5s}: {len(split):>5d} rows  →  {path}")

explore_full = EXPLORE_DIR / "emails_clean.csv"
df_spamassasin.to_csv(explore_full, index=False)
print(f"  full : {len(df_spamassasin):>5d} rows  →  {explore_full}")


# ═══════════════════════════════════════════════════════════════════════
#  B. phishing_email dataset  →  data_training/oks 02 & 03)
# ═══════════════════════════════════════════════════════════════════════

phishing_path = None
for p in csv_candidates:
    if "phishing_email" in p.name.lower():
        phishing_path = p
        break

if phishing_path is None:
    phishing_path = max(csv_candidates, key=lambda p: p.stat().st_size)
    print(f"  (Nazario excluded, falling back to largest file: {phishing_path.name})")

DATA_DIR = BASE_DIR / "data_training"
DATA_DIR.mkdir(exist_ok=True)

df_phishing = clean_dataset(phishing_path)

# Train / val / test split (70 / 15 / 15)
train, tmp = train_test_split(
    df_phishing, test_size=0.30, random_state=42, stratify=df_phishing["label"]
)
val, test = train_test_split(
    tmp, test_size=0.50, random_state=42, stratify=tmp["label"]
)

for name, split in [("train", train), ("val", val), ("test", test)]:
    path = DATA_DIR / f"{name}.csv"
    split.to_csv(path, index=False)
    print(f"  {name:5s}: {len(split):>5d} rows  →  {path}")

full_path = DATA_DIR / "emails_clean.csv"
df_phishing.to_csv(full_path, index=False)
print(f"  full : {len(df_phishing):>5d} rows  →  {full_path}")


# ═══════════════════════════════════════════════════════════════════════
print("\n✓ Dataset preparation complete!")
print(f"  data_exploration/  →  SpamAssasin  ({len(df_spamassasin):,} rows)  — for notebooks 00 & 01")
print(f"  data_training/ →  phishing_email ({len(df_phishing):,} rows)  — for notebooks 02 & 03")
