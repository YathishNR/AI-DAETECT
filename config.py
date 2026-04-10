import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "leak_detector.joblib")
