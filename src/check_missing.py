import pandas as pd

data = pd.read_parquet(r'C:\Users\KIIT0001\sepsis-detection\data\processed\combined_raw.parquet')

missing = data.isnull().mean().sort_values(ascending=False) * 100
print("Missing data % per column:")
print(missing[missing > 0].round(1).to_string())
print(f"\nRows with ANY missing vital: {data[['HR','O2Sat','Temp','SBP','Resp']].isnull().any(axis=1).sum():,}")