import pandas
import shap
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Pandas:", pandas.__version__)
print("SHAP:", shap.__version__)
print("All imports OK")
