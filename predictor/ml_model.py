import os
from joblib import load
from django.conf import settings

# Path to the pickle file
MODEL_PATH = os.path.join(settings.BASE_DIR, "model", "train_profit_loss_model_compressed.pkl")

pipeline = None
model = None
customer_type_map = {}
product_name_map = {}

def get_pipeline():
    global pipeline, model, customer_type_map, product_name_map
    if pipeline is None:
        try:
            pipeline = load(MODEL_PATH)
            model = pipeline["model"]
            customer_type_map = pipeline["customer_type_map"]
            product_name_map = pipeline["product_name_map"]
            print("ML model loaded successfully.")
        except Exception as e:
            print(f"Error loading ML model: {e}")
            pipeline = None
            model = None
            customer_type_map = {}
            product_name_map = {}
    return model, customer_type_map, product_name_map
