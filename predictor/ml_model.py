import os
from joblib import load
from django.conf import settings

# Path to the pickle file
MODEL_PATH = os.path.join(settings.BASE_DIR, "model", "train_profit_loss_model_compressed.pkl")

# Load the pipeline and maps once at startup
try:
    pipeline = load(MODEL_PATH)
    model = pipeline["model"]
    customer_type_map = pipeline["customer_type_map"]
    product_name_map = pipeline["product_name_map"]
    print("ML model loaded successfully.")
except Exception as e:
    # If loading fails, raise an error with details (Render logs will show it)
    print(f"Error loading ML model: {e}")
    model = None
    customer_type_map = {}
    product_name_map = {}
