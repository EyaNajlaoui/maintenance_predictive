"""PrognosIA - site de demonstration pour le diagnostic (classification
sain/critique) et la prevision RUL (LSTM) des moteurs, a partir des modeles
deja entraines dans ../models.

Lancement : python website/app.py
"""
import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, render_template, request

import config

app = Flask(__name__)

SENSOR_COLUMNS = [f'sensor{i}' for i in range(1, 22)]
OP_COLUMNS = ['op_setting1', 'op_setting2', 'op_setting3']
CLASSIFICATION_FEATURE_ORDER = SENSOR_COLUMNS + OP_COLUMNS + ['op_regime']

MODEL_LABELS = {
    'RandomForest': 'Random Forest',
    'XGBoost': 'XGBoost',
    'SVM': 'SVM',
}
CLASSIFICATION_MODEL_NAME = 'XGBoost'


def _load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, 'r') as f:
        return json.load(f)


class ModelBundle:
    """Charge une fois tous les artefacts necessaires aux predictions."""

    def __init__(self):
        self.classification_scaler = joblib.load(
            os.path.join(config.MODELS_DIR, 'classification_scaler.pkl'))
        self.classifier = joblib.load(
            os.path.join(config.MODELS_DIR, f'classifier_{CLASSIFICATION_MODEL_NAME.lower()}.pkl'))

        self.kmeans = joblib.load(os.path.join(config.MODELS_DIR, 'kmeans_model.pkl'))
        self.op_scaler = joblib.load(os.path.join(config.MODELS_DIR, 'op_scaler.pkl'))

        self.lstm_scaler = joblib.load(os.path.join(config.MODELS_DIR, 'scaler.pkl'))

        features_data = _load_json(os.path.join(config.MODELS_DIR, 'selected_features.json'))
        if not features_data:
            raise RuntimeError(
                "models/selected_features.json est introuvable. "
                "Lancer le script d'export des artefacts avant de demarrer le site."
            )
        self.lstm_features = features_data['lstm_selected_features']

        self.lstm_models = []
        i = 1
        while True:
            path = os.path.join(config.MODELS_DIR, f'{config.LSTM_MODEL_PREFIX}_{i}.h5')
            if not os.path.exists(path):
                break
            self.lstm_models.append(tf.keras.models.load_model(path, compile=False))
            i += 1
        if not self.lstm_models:
            raise RuntimeError("Aucun modele LSTM trouve dans models/.")

        self.example_values = _load_json(
            os.path.join(config.MODELS_DIR, 'example_values.json'), default={'sensors': {}, 'op_settings': {}})

        self.classification_metrics = _load_json(config.CLASSIFICATION_METRICS_FILE, default={})
        self.rul_metrics = _load_json(config.RUL_METRICS_FILE, default={})

    def predict_op_regime(self, op_values):
        scaled = self.op_scaler.transform([op_values])
        return int(self.kmeans.predict(scaled)[0])

    def predict_classification(self, sensor_values, op_values, op_regime):
        row = {**sensor_values, **op_values, 'op_regime': op_regime}
        vector = np.array([[row[col] for col in CLASSIFICATION_FEATURE_ORDER]])
        scaled = self.classification_scaler.transform(vector)
        proba_critical = float(self.classifier.predict_proba(scaled)[0, 1])
        label = 'critique' if proba_critical >= 0.5 else 'sain'
        return label, proba_critical

    def predict_rul(self, sensor_values, op_regime):
        row = {**sensor_values, 'op_regime': op_regime}
        step = np.array([row[feat] for feat in self.lstm_features], dtype=float)
        sequence = np.tile(step, (config.SEQUENCE_LENGTH, 1))[np.newaxis, :, :]

        shape = sequence.shape
        scaled = self.lstm_scaler.transform(sequence.reshape(-1, shape[-1])).reshape(shape)

        predictions = [m.predict(scaled, verbose=0).flatten()[0] for m in self.lstm_models]
        rul = float(np.mean(predictions))
        rul = max(0.0, min(rul, config.RUL_CAP))
        return rul


bundle = ModelBundle()


def _numeric_fields():
    """Construit la liste ordonnee des champs du formulaire avec leurs
    valeurs par defaut, pour affichage et pour le parsing des reponses."""
    fields = {}
    for col in OP_COLUMNS:
        stats = bundle.example_values.get('op_settings', {}).get(col, {})
        fields[col] = stats.get('mean', 0.0)
    for col in SENSOR_COLUMNS:
        stats = bundle.example_values.get('sensors', {}).get(col, {})
        fields[col] = stats.get('mean', 0.0)
    return fields


@app.route('/')
def index():
    return render_template(
        'index.html',
        classification_metrics=bundle.classification_metrics.get(CLASSIFICATION_MODEL_NAME, {}),
        rul_metrics=bundle.rul_metrics,
    )


@app.route('/diagnostic', methods=['GET', 'POST'])
def diagnostic():
    defaults = _numeric_fields()
    values = dict(defaults)
    result = None
    error = None

    if request.method == 'POST':
        try:
            for key in values:
                raw = request.form.get(key)
                values[key] = float(raw) if raw not in (None, '') else defaults[key]

            sensor_values = {col: values[col] for col in SENSOR_COLUMNS}
            op_values = {col: values[col] for col in OP_COLUMNS}
            op_values_list = [op_values[col] for col in OP_COLUMNS]

            op_regime = bundle.predict_op_regime(op_values_list)
            label, proba_critical = bundle.predict_classification(sensor_values, op_values, op_regime)
            rul = bundle.predict_rul(sensor_values, op_regime)

            result = {
                'op_regime': op_regime,
                'classification_label': label,
                'proba_critical': proba_critical,
                'rul': rul,
                'rul_cap': config.RUL_CAP,
            }
        except (TypeError, ValueError) as exc:
            error = f"Valeurs invalides dans le formulaire : {exc}"

    return render_template(
        'diagnostic.html',
        sensor_columns=SENSOR_COLUMNS,
        op_columns=OP_COLUMNS,
        lstm_features=set(bundle.lstm_features),
        values=values,
        result=result,
        error=error,
        model_label=MODEL_LABELS[CLASSIFICATION_MODEL_NAME],
    )


@app.route('/methodologie')
def methodologie():
    return render_template(
        'methodologie.html',
        classification_metrics=bundle.classification_metrics,
        rul_metrics=bundle.rul_metrics,
        model_labels=MODEL_LABELS,
        lstm_features=bundle.lstm_features,
        rul_cap=config.RUL_CAP,
        sequence_length=config.SEQUENCE_LENGTH,
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
