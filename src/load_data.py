import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(r'C:\Users\KIIT0001\sepsis-detection\data\raw\kaggle')
OUTPUT = Path(r'C:\Users\KIIT0001\sepsis-detection\data\processed')
OUTPUT.mkdir(parents=True, exist_ok=True)

files = list(DATA_DIR.rglob('*.psv'))
print(f"Found {len(files)} patient files")

dfs = []
for f in tqdm(files, desc="Loading patients"):
    df = pd.read_csv(f, sep='|')
    df['patient_id'] = f.stem
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
print(f"Total rows: {len(data):,}")
print(f"Sepsis cases: {data.groupby('patient_id')['SepsisLabel'].max().sum():.0f}")

data.to_parquet(OUTPUT / 'combined_raw.parquet', index=False)
print("Saved to data/processed/combined_raw.parquet")