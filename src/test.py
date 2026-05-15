import pickle, os
os.chdir(r'C:\Users\KIIT0001\sepsis-detection')

with open('data/processed/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print(f"Scaler expects: {scaler.n_features_in_} features")
