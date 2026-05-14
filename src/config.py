MODEL_CONFIG = {
    'input_size':  22,
    'hidden_size': 64,
    'num_layers':  2,
    'dropout':     0.3,
}

TRAIN_CONFIG = {
    'epochs':      30,
    'batch_size':  256,
    'lr':          1e-3,
    'weight_decay':1e-4,
    'pos_weight':  46.6,
    'patience':    5,
    'grad_clip':   1.0,
}

FEATURE_NAMES = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp',
    'Creatinine', 'Lactate', 'WBC', 'Glucose',
    'Potassium', 'Hgb', 'Age', 'ICULOS',
    'HR_roll_mean', 'O2Sat_roll_mean',
    'SBP_roll_mean', 'Resp_roll_mean',
    'HR_roll_std', 'O2Sat_roll_std',
    'SBP_roll_std', 'Resp_roll_std',
]

WINDOW_SIZE = 12
PRED_GAP    = 6