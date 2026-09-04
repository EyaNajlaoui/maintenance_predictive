"""Configuration centralisee du pipeline (chemins, hyperparametres).
"""
import os


def _env(name, default, cast=str):
    value = os.environ.get(f'RUL_{name}')
    return cast(value) if value is not None else default


# --- Chemins ---
DATA_DIR = _env('DATA_DIR', 'data')
MODELS_DIR = _env('MODELS_DIR', 'models')
RESULTS_DIR = _env('RESULTS_DIR', 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

DATASET = _env('DATASET', 'FD002')
TRAIN_FILE = os.path.join(DATA_DIR, f'train_{DATASET}.txt')
TEST_FILE = os.path.join(DATA_DIR, f'test_{DATASET}.txt')
RUL_FILE = os.path.join(DATA_DIR, f'RUL_{DATASET}.txt')

CLASSIFICATION_METRICS_FILE = os.path.join(METRICS_DIR, 'classification_metrics.json')
RUL_METRICS_FILE = os.path.join(METRICS_DIR, 'rul_metrics.json')
LSTM_MODEL_PREFIX = 'lstm_model_fold'

# --- Clustering des regimes operationnels ---
N_CLUSTERS = _env('N_CLUSTERS', 3, int)

# --- Selection de features ---
N_FEATURES = _env('N_FEATURES', 15, int)

# --- Classification etat sain / critique ---
CRITICAL_THRESHOLD = _env('CRITICAL_THRESHOLD', 30, int)

# --- Sequences / LSTM ---
# Valeurs reduites pour un entrainement rapide sur CPU (quelques minutes au lieu
# de ~2h avec les valeurs d'origine : 100 epochs x 3 folds x batch_size=32).
# Pour un entrainement plus long/plus precis, augmenter via les variables
# d'environnement RUL_LSTM_EPOCHS, RUL_LSTM_N_SPLITS, etc.
SEQUENCE_LENGTH = _env('SEQUENCE_LENGTH', 30, int)
# Plafonnement du RUL (pratique standard sur C-MAPSS, cf. Heimes 2008 / Saxena et al.) :
# au-dela de ce nombre de cycles, la degradation n'est pas encore lineaire/detectable,
# donc predire une valeur exacte n'a pas de sens et perturbe l'apprentissage.
RUL_CAP = _env('RUL_CAP', 125, int)
LSTM_LEARNING_RATE = _env('LSTM_LEARNING_RATE', 0.001, float)
LSTM_N_SPLITS = _env('LSTM_N_SPLITS', 2, int)
LSTM_EPOCHS = _env('LSTM_EPOCHS', 30, int)
LSTM_BATCH_SIZE = _env('LSTM_BATCH_SIZE', 64, int)
LSTM_EARLY_STOPPING_PATIENCE = _env('LSTM_EARLY_STOPPING_PATIENCE', 15, int)
LSTM_REDUCE_LR_PATIENCE = _env('LSTM_REDUCE_LR_PATIENCE', 2, int)
LSTM_MIN_LR = _env('LSTM_MIN_LR', 0.0001, float)

RANDOM_STATE = _env('RANDOM_STATE', 42, int)
