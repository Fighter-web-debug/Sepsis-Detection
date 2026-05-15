import os, sys, pickle
import numpy as np
import torch
from flask import Flask, request, jsonify, render_template
sys.path.append(os.path.join(os.path.dirname(__file__)))
os.chdir(r'C:\Users\KIIT0001\sepsis-detection')

from model import SepsisLSTM

app = Flask(__name__,
            template_folder='../templates',
            static_folder='../static')

INPUT_SIZE  = 21
THRESHOLD   = 0.15

model = SepsisLSTM(input_size=INPUT_SIZE)
model.load_state_dict(
    torch.load('models/best_model.pt', map_location='cpu')
)
model.eval()

with open('data/processed/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

FEATURE_NAMES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp',
    'Creatinine', 'Lactate', 'WBC', 'Glucose',
    'Potassium', 'Hgb', 'Age',
    'HR_roll_mean', 'O2Sat_roll_mean',
    'SBP_roll_mean', 'Resp_roll_mean',
    'HR_roll_std', 'O2Sat_roll_std',
    'SBP_roll_std', 'Resp_roll_std',
]

@app.route('/')
def index():
    return render_template('index.html',
                           features=FEATURE_NAMES[:8])

@app.route('/predict_sequence', methods=['POST'])
def predict_sequence():
    try:
        seq = request.json['sequence']
        x = np.array(seq, dtype=np.float32)
        x = scaler.transform(x)
        x_t = torch.tensor(x[np.newaxis], dtype=torch.float32)
        with torch.no_grad():
            prob = torch.sigmoid(model(x_t)).item()
        return jsonify({'probability': round(prob * 100, 1)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model': 'SepsisLSTM',
                    'input_size': INPUT_SIZE})

if __name__ == '__main__':
    app.run(debug=True, port=5000)