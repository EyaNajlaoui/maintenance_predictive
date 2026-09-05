"""PrognosIA - version Streamlit du site de demonstration pour le diagnostic
(classification sain/critique) et la prevision RUL (LSTM) des moteurs, a
partir des modeles deja entraines dans ../models.

Lancement local : streamlit run website/streamlit_app.py
Deploiement : streamlit.io (Streamlit Community Cloud), main file path =
website/streamlit_app.py
"""
import os
import sys
import json

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import joblib
import tensorflow as tf
import streamlit as st

import config

SENSOR_COLUMNS = [f'sensor{i}' for i in range(1, 22)]
OP_COLUMNS = ['op_setting1', 'op_setting2', 'op_setting3']
CLASSIFICATION_FEATURE_ORDER = SENSOR_COLUMNS + OP_COLUMNS + ['op_regime']

MODEL_LABELS = {
    'RandomForest': 'Random Forest',
    'XGBoost': 'XGBoost',
    'SVM': 'SVM',
}
CLASSIFICATION_MODEL_NAME = 'XGBoost'

st.set_page_config(page_title='PrognosIA', page_icon='🔧', layout='wide')


def _load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, 'r') as f:
        return json.load(f)


@st.cache_resource(show_spinner="Chargement des modèles...")
def load_bundle():
    class ModelBundle:
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

    return ModelBundle()


bundle = load_bundle()


def numeric_fields():
    fields = {}
    for col in OP_COLUMNS:
        stats = bundle.example_values.get('op_settings', {}).get(col, {})
        fields[col] = stats.get('mean', 0.0)
    for col in SENSOR_COLUMNS:
        stats = bundle.example_values.get('sensors', {}).get(col, {})
        fields[col] = stats.get('mean', 0.0)
    return fields


# --- Style (repris de website/static/css/style.css) ---
st.markdown("""
<style>
:root {
  --color-bg: #f5f7f8;
  --color-surface: #ffffff;
  --color-border: #e2e8ea;
  --color-text: #0f1a1c;
  --color-text-muted: #3f4d50;
  --color-primary: #0f6e6c;
  --color-primary-dark: #0b524f;
  --color-primary-soft: #e6f2f1;
  --color-accent: #c98a2e;
  --color-danger: #b3423a;
  --color-danger-soft: #fbeae8;
  --color-success: #2f7d4f;
  --color-success-soft: #eaf6ee;
  --radius: 10px;
}
html, body, [class*="css"] { font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; }
.stApp { background: var(--color-bg); color: var(--color-text); }
.stApp p, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: var(--color-text); }
#MainMenu, footer, header { visibility: hidden; }

.eyebrow {
  color: var(--color-primary); font-weight: 700; letter-spacing: .04em;
  text-transform: uppercase; font-size: .78rem; margin-bottom: 6px;
}
.hero-title { font-size: 2.3rem; line-height: 1.15; margin: 0 0 14px; color: var(--color-text); font-weight: 800; }
.hero-lead { font-size: 1.05rem; color: var(--color-text-muted); margin-bottom: 10px; }

.panel {
  background: var(--color-surface); border: 1px solid var(--color-border);
  border-radius: var(--radius); padding: 22px; box-shadow: 0 1px 2px rgba(16,40,40,.06);
}

.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 10px; }
.stat { background: var(--color-primary-soft); border-radius: 8px; padding: 14px 16px; }
.stat .value { font-size: 1.5rem; font-weight: 800; color: var(--color-primary-dark); }
.stat .label { font-size: .8rem; color: var(--color-text-muted); }

.card-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-top: 10px; }
.card {
  background: var(--color-surface); border: 1px solid var(--color-border);
  border-radius: var(--radius); padding: 20px; box-shadow: 0 1px 2px rgba(16,40,40,.06);
}
.card__icon {
  width: 38px; height: 38px; border-radius: 8px; background: var(--color-primary-soft);
  color: var(--color-primary-dark); display: flex; align-items: center; justify-content: center;
  font-weight: 700; margin-bottom: 12px;
}
.card h3 { margin: 0 0 6px; font-size: 1.02rem; }
.card p { color: var(--color-text-muted); margin: 0; font-size: .92rem; }

.badge {
  display: inline-block; font-size: .62rem; font-weight: 700; padding: 2px 7px;
  border-radius: 999px; background: var(--color-accent); color: #fff;
  text-transform: uppercase; letter-spacing: .02em; margin: 2px;
}
.badge-lstm { background: var(--color-primary); }

.result-card {
  border-radius: var(--radius); padding: 20px; margin-bottom: 14px; border: 1px solid var(--color-border);
}
.result-card.state-ok { background: var(--color-success-soft); border-color: #bfe3cb; }
.result-card.state-critical { background: var(--color-danger-soft); border-color: #f0c4c0; }
.result-card__label {
  font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .03em;
  color: var(--color-text-muted); margin-bottom: 6px;
}
.result-card__value { font-size: 1.6rem; font-weight: 800; }
.state-ok .result-card__value { color: var(--color-success); }
.state-critical .result-card__value { color: var(--color-danger); }
.result-card__meta { font-size: .85rem; color: var(--color-text-muted); margin-top: 6px; }
.progress-bar { height: 8px; border-radius: 999px; background: #e6e6e6; overflow: hidden; margin-top: 10px; }
.progress-bar__fill { height: 100%; border-radius: 999px; background: var(--color-primary); }

.placeholder-note {
  color: var(--color-text-muted); font-size: .9rem; padding: 16px; background: var(--color-bg);
  border-radius: 8px; border: 1px dashed var(--color-border);
}
.disclaimer {
  font-size: .8rem; color: var(--color-text-muted); margin-top: 14px; padding-top: 12px;
  border-top: 1px solid var(--color-border);
}

table.custom { width: 100%; border-collapse: collapse; font-size: .92rem; background: var(--color-surface); }
table.custom th, table.custom td {
  text-align: left; padding: 10px 14px; border-bottom: 1px solid var(--color-border);
  color: var(--color-text);
}
table.custom th {
  background: var(--color-bg); color: var(--color-text-muted); font-size: .76rem;
  text-transform: uppercase; letter-spacing: .03em;
}
table.custom tr:last-child td { border-bottom: none; }

.site-footer { color: var(--color-text-muted); font-size: .84rem; text-align: center; padding: 24px 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Maintenance prédictive</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">PrognosIA</div>', unsafe_allow_html=True)

tab_accueil, tab_diagnostic, tab_methodologie = st.tabs(["🏠 Accueil", "🔧 Diagnostic", "📊 Méthodologie"])

# --- Accueil ---
with tab_accueil:
    col_left, col_right = st.columns([1.1, 0.9])
    with col_left:
        st.markdown(
            '<p class="hero-lead">PrognosIA combine un modèle de classification (état sain / critique) et un '
            'modèle de prévision de durée de vie restante (RUL) pour aider à planifier la maintenance des '
            'moteurs, à partir de leurs données capteurs.</p>',
            unsafe_allow_html=True,
        )
    with col_right:
        cm = bundle.classification_metrics.get(CLASSIFICATION_MODEL_NAME, {})
        rm = bundle.rul_metrics
        st.markdown(f"""
        <div class="panel">
          <h3 style="margin-top:0;font-size:.9rem;color:var(--color-text-muted);
              text-transform:uppercase;letter-spacing:.03em;">Performance des modèles</h3>
          <div class="stat-grid">
            <div class="stat"><div class="value">{cm.get('accuracy', 0) * 100:.1f}%</div><div class="label">Précision classification</div></div>
            <div class="stat"><div class="value">{cm.get('roc_auc', 0):.2f}</div><div class="label">ROC-AUC</div></div>
            <div class="stat"><div class="value">{rm.get('r2', 0):.2f}</div><div class="label">R² prévision RUL</div></div>
            <div class="stat"><div class="value">±{rm.get('mae', 0):.0f}</div><div class="label">Cycles d'erreur moyenne (MAE)</div></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<h2 style="margin-top:36px;">Deux modèles, une seule décision</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:var(--color-text-muted);">Un diagnostic complet nécessite de savoir où en est le '
        'moteur aujourd\'hui, et combien de temps il lui reste.</p>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    <div class="card-grid">
      <div class="card">
        <div class="card__icon">1</div>
        <h3>Diagnostic d'état</h3>
        <p>Un modèle de classification (XGBoost) analyse les mesures capteurs et les réglages opérationnels
        pour déterminer si le moteur est dans un état <strong>sain</strong> ou <strong>critique</strong>.</p>
      </div>
      <div class="card">
        <div class="card__icon">2</div>
        <h3>Prévision RUL</h3>
        <p>Un réseau de neurones LSTM estime le nombre de cycles restants avant la panne (Remaining Useful
        Life), à partir de l'évolution récente des capteurs les plus significatifs.</p>
      </div>
      <div class="card">
        <div class="card__icon">3</div>
        <h3>Paramètres ajustables</h3>
        <p>Sur l'onglet Diagnostic, chaque paramètre (capteurs, réglages moteur) est modifiable librement pour
        simuler différents scénarios et observer leur impact sur la prévision.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

# --- Diagnostic ---
with tab_diagnostic:
    st.markdown('<div class="eyebrow">Diagnostic</div>', unsafe_allow_html=True)
    st.markdown('<h1 style="font-size:1.7rem;margin:0 0 6px;">Diagnostic et prévision moteur</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:var(--color-text-muted);margin-bottom:24px;">Ajustez les réglages opérationnels et '
        'les mesures capteurs ci-dessous, puis lancez le calcul pour obtenir l\'état du moteur et la durée de '
        'vie restante estimée.</p>',
        unsafe_allow_html=True,
    )

    if 'diag_values' not in st.session_state:
        st.session_state.diag_values = numeric_fields()

    col_form, col_result = st.columns([1.2, 0.8])

    with col_form:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<h2 style="margin-top:0;font-size:1.05rem;">Paramètres du moteur</h2>', unsafe_allow_html=True)

        with st.form("diagnostic_form"):
            st.markdown('<div style="font-size:.8rem;font-weight:700;text-transform:uppercase;'
                        'letter-spacing:.03em;color:var(--color-text-muted);margin-bottom:8px;">'
                        'Réglages opérationnels</div>', unsafe_allow_html=True)
            op_cols = st.columns(3)
            op_values = {}
            for i, col in enumerate(OP_COLUMNS):
                op_values[col] = op_cols[i].number_input(
                    col, value=float(st.session_state.diag_values[col]), format="%.4f", key=f"input_{col}")

            st.markdown('<div style="font-size:.8rem;font-weight:700;text-transform:uppercase;'
                        'letter-spacing:.03em;color:var(--color-text-muted);margin:18px 0 8px;">'
                        'Capteurs</div>', unsafe_allow_html=True)
            sensor_values = {}
            sensor_cols = st.columns(3)
            for i, col in enumerate(SENSOR_COLUMNS):
                label = col + (" (RUL)" if col in set(bundle.lstm_features) else "")
                sensor_values[col] = sensor_cols[i % 3].number_input(
                    label, value=float(st.session_state.diag_values[col]), format="%.4f", key=f"input_{col}")

            c1, c2 = st.columns(2)
            submitted = c1.form_submit_button("Calculer le diagnostic", type="primary", use_container_width=True)
            reset = c2.form_submit_button("Réinitialiser aux valeurs moyennes", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        if reset:
            st.session_state.diag_values = numeric_fields()
            st.session_state.diag_result = None
            st.rerun()

        if submitted:
            st.session_state.diag_values = {**op_values, **sensor_values}
            op_values_list = [op_values[c] for c in OP_COLUMNS]
            op_regime = bundle.predict_op_regime(op_values_list)
            label, proba_critical = bundle.predict_classification(sensor_values, op_values, op_regime)
            rul = bundle.predict_rul(sensor_values, op_regime)
            st.session_state.diag_result = {
                'op_regime': op_regime,
                'classification_label': label,
                'proba_critical': proba_critical,
                'rul': rul,
            }

    with col_result:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<h2 style="margin-top:0;font-size:1.05rem;">Résultat</h2>', unsafe_allow_html=True)

        result = st.session_state.get('diag_result')
        if result:
            state_class = 'state-critical' if result['classification_label'] == 'critique' else 'state-ok'
            state_label = 'Critique' if result['classification_label'] == 'critique' else 'Sain'
            st.markdown(f"""
            <div class="result-card {state_class}">
              <div class="result-card__label">État du moteur</div>
              <div class="result-card__value">{state_label}</div>
              <div class="result-card__meta">Probabilité d'état critique : {result['proba_critical'] * 100:.1f}%
                — modèle {MODEL_LABELS[CLASSIFICATION_MODEL_NAME]}</div>
              <div class="progress-bar"><div class="progress-bar__fill" style="width:{result['proba_critical'] * 100:.1f}%;"></div></div>
            </div>
            <div class="result-card">
              <div class="result-card__label">Durée de vie restante estimée (RUL)</div>
              <div class="result-card__value">{result['rul']:.0f} cycles</div>
              <div class="result-card__meta">Régime opérationnel détecté : cluster {result['op_regime']}
                — prévision plafonnée à {config.RUL_CAP} cycles</div>
            </div>
            <p class="disclaimer">La prévision RUL suppose que les valeurs saisies restent stables sur les
            {config.SEQUENCE_LENGTH} derniers cycles (le modèle LSTM analyse un historique de
            {config.SEQUENCE_LENGTH} cycles). Pour un moteur réel, cet historique proviendrait de mesures
            successives plutôt que d'une seule saisie.</p>
            """, unsafe_allow_html=True)
        else:
            st.markdown(
                '<p class="placeholder-note">Renseignez les paramètres puis cliquez sur '
                '« Calculer le diagnostic » pour afficher le résultat ici.</p>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

# --- Methodologie ---
with tab_methodologie:
    st.markdown('<div class="eyebrow">Méthodologie</div>', unsafe_allow_html=True)
    st.markdown('<h1 style="font-size:1.7rem;margin:0 0 6px;">Comment fonctionnent les modèles</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:var(--color-text-muted);margin-bottom:24px;">PrognosIA s\'appuie sur le jeu de '
        'données NASA C-MAPSS (sous-ensemble FD002), qui simule la dégradation de moteurs d\'avion sous '
        'plusieurs régimes opérationnels.</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<h2 style="font-size:1.1rem;">Classification — état sain / critique</h2>', unsafe_allow_html=True)
    st.markdown(
        '<p>Trois modèles ont été entraînés et comparés (Random Forest, XGBoost, SVM) pour prédire si un '
        'moteur est en état critique (moins de 30 cycles avant la panne) à partir des 21 capteurs, des 3 '
        'réglages opérationnels et du régime détecté.</p>',
        unsafe_allow_html=True,
    )

    rows = ""
    for key, label in MODEL_LABELS.items():
        m = bundle.classification_metrics.get(key)
        if m:
            rows += f"""<tr>
              <td>{label}</td><td>{m['accuracy']:.3f}</td><td>{m['precision']:.3f}</td>
              <td>{m['recall']:.3f}</td><td>{m['f1']:.3f}</td><td>{m['roc_auc']:.3f}</td>
            </tr>"""
    st.markdown(f"""
    <table class="custom">
      <thead><tr><th>Modèle</th><th>Accuracy</th><th>Précision</th><th>Rappel</th><th>F1-score</th><th>ROC-AUC</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<h2 style="font-size:1.1rem;margin-top:32px;">Prévision RUL — réseau LSTM</h2>', unsafe_allow_html=True)
    st.markdown(
        f'<p>La durée de vie restante (Remaining Useful Life) est prédite par un ensemble de réseaux LSTM '
        f'entraînés par validation croisée, sur une fenêtre de {config.SEQUENCE_LENGTH} cycles et les '
        f'{len(bundle.lstm_features)} capteurs les plus informatifs, sélectionnés par importance (Random Forest) :</p>',
        unsafe_allow_html=True,
    )
    badges = "".join(f'<span class="badge badge-lstm">{feat}</span>' for feat in bundle.lstm_features)
    st.markdown(f'<p>{badges}</p>', unsafe_allow_html=True)

    rm = bundle.rul_metrics
    st.markdown(f"""
    <table class="custom" style="margin-top:12px;">
      <thead><tr><th>MAE</th><th>RMSE</th><th>R²</th><th>Plafond RUL</th></tr></thead>
      <tbody><tr>
        <td>{rm.get('mae', 0):.2f} cycles</td>
        <td>{rm.get('rmse', 0):.2f} cycles</td>
        <td>{rm.get('r2', 0):.3f}</td>
        <td>{config.RUL_CAP} cycles</td>
      </tr></tbody>
    </table>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="site-footer">PrognosIA — Diagnostic et prévision de fiabilité moteur, à partir du jeu de '
    'données NASA C-MAPSS (FD002).</div>',
    unsafe_allow_html=True,
)
