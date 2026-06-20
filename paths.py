from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "synthetic_bank_data"
DATASET_PATH = DATA_DIR / "combined_transactions.csv"
DEMO_DATASET_PATH = DATA_DIR / "demo_transactions.csv"
MODEL_PATH = PROJECT_ROOT / "spending_classifier.pkl"
DATABASE_PATH = PROJECT_ROOT / "spending_analytics.db"
