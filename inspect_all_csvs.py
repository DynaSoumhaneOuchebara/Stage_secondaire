"""
Quick inspection script: for each CSV in the Kaggle download,
show its columns, row count, and label distribution (safe vs phishing).
"""

import pathlib
import kagglehub
import pandas as pd

dataset_path = kagglehub.dataset_download("naserabdullahalam/phishing-email-dataset")
csv_candidates = sorted(pathlib.Path(dataset_path).rglob("*.csv"))

print(f"Found {len(csv_candidates)} CSV file(s) in: {dataset_path}\n")

for csv_path in csv_candidates:
    print(f"{'═'*70}")
    print(f"  File: {csv_path.name}  ({csv_path.stat().st_size / 1e6:.1f} MB)")
    print(f"{'═'*70}")

    df = pd.read_csv(csv_path, nrows=None)
    print(f"  Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}\n")

    # Try to find the label column
    label_col = None
    for col in df.columns:
        low = col.strip().lower()
        if "type" in low or "label" in low or "class" in low:
            label_col = col
            break

    if label_col is None:
        label_col = df.columns[-1]
        print(f"  (No obvious label column found, using last column: '{label_col}')")

    vc = df[label_col].value_counts()
    print(f"  Label column: '{label_col}'")
    print(f"  Unique values ({len(vc)}):")
    for val, count in vc.items():
        print(f"    {str(val):25s} → {count:>7,} rows")

    # Check if it contains both safe and phishing
    labels_lower = set(str(v).strip().lower() for v in vc.index)
    phishing_keys = {"phishing", "spam", "1", "malicious"}
    safe_keys = {"safe", "safe email", "ham", "legitimate", "0", "benign"}

    has_phishing = bool(labels_lower & phishing_keys)
    has_safe = bool(labels_lower & safe_keys)

    if has_phishing and has_safe:
        verdict = "✓ BOTH safe and phishing"
    elif has_phishing:
        verdict = "✗ Phishing ONLY"
    elif has_safe:
        verdict = "✗ Safe ONLY"
    else:
        verdict = f"? Unknown labels: {labels_lower}"

    print(f"\n  → {verdict}")

    # Show a sample of the text column to judge readability
    text_col = None
    for col in df.columns:
        low = col.strip().lower()
        if "text" in low or "body" in low or "content" in low or "message" in low:
            text_col = col
            break
    if text_col is None:
        text_col = df.columns[0]

    sample = df[text_col].dropna().iloc[0]
    preview = str(sample)[:200].replace("\n", " ")
    print(f"\n  Sample text (first 200 chars):")
    print(f"    {preview}")
    print()
