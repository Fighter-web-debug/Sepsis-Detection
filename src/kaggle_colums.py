VITALS = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']

LABS = ['Creatinine', 'Lactate', 'WBC', 'Bilirubin_total',
        'Platelets', 'Glucose', 'Potassium', 'Hgb']

DEMOGRAPHICS = ['Age', 'Gender', 'ICULOS', 'HospAdmTime']

LABEL = 'SepsisLabel'

ALL_FEATURES = VITALS + LABS + DEMOGRAPHICS

MIMIC_EQUIV = {
    'HR':         'heart_rate (220045)',
    'O2Sat':      'spo2 (220277)',
    'Temp':       'temperature (223762)',
    'SBP':        'systolic bp (220179)',
    'Resp':       'resp rate (220210)',
    'Creatinine': 'creatinine (220615)',
    'Lactate':    'lactate (225668)',
    'WBC':        'wbc (220546)',
}

