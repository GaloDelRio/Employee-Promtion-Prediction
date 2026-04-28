from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs'

TARGET_COLUMN = 'promoted'
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE_WITHIN_TEMP = 0.50  # 10% val y 10% test a partir del 20% temporal
THRESHOLD = 0.50

COLUMNS_TO_DROP = [
    'employee_id',
    'team_size',
    'remote_work_ratio',
    'deadline_adherence_rate',
    'cross_department_projects',
    'mentoring_sessions',
    'internal_mobility_score',
    'attendance_rate',
    'training_hours_last_year',
    'certifications_count',
    'performance_two_years_ago',
]

# Hiperparámetros base del modelo Keras SAINT-like
EMB_DIM = 16
NUM_HEADS = 1
NUM_LAYERS = 1
DROPOUT = 0.10
BATCH_SIZE = 1024
EPOCHS = 15
LEARNING_RATE = 0.001
PATIENCE = 3

# Archivos de salida
PREPROCESSOR_PATH = OUTPUT_DIR / 'preprocessor.pkl'
MODEL_PATH = OUTPUT_DIR / 'model.keras'
MODEL_SUMMARY_PATH = OUTPUT_DIR / 'model_summary.txt'
HISTORY_CSV_PATH = OUTPUT_DIR / 'training_history.csv'
HISTORY_JSON_PATH = OUTPUT_DIR / 'training_history.json'
TRAINING_PLOT_PATH = OUTPUT_DIR / 'training_curve.png'
METRICS_JSON_PATH = OUTPUT_DIR / 'metrics.json'
REPORTS_JSON_PATH = OUTPUT_DIR / 'classification_reports.json'
DATASET_SUMMARY_PATH = OUTPUT_DIR / 'dataset_summary.json'
VALIDATION_CM_PATH = OUTPUT_DIR / 'confusion_matrix_validation.png'
TEST_CM_PATH = OUTPUT_DIR / 'confusion_matrix_test.png'
BATCH_PREDICTIONS_PATH = OUTPUT_DIR / 'batch_predictions.csv'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)