import os
import numpy as np
import tensorflow as tf
from models.advanced_predictor import build_health_ann

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'ann_model.keras')



_ann_model = None

def get_ann_model():
    global _ann_model
    if _ann_model is None:
        if os.path.exists(ANN_MODEL_PATH):
            _ann_model = tf.keras.models.load_model(ANN_MODEL_PATH)
        else:
            print(f"WARNING: Model not found at {ANN_MODEL_PATH}. Using dummy predictions.")
            
            _ann_model = build_health_ann(input_size=4) 
    return _ann_model

def generate_health_scores(patient_data):
    """
    Takes patient data, runs it through Keras models, returns a dictionary of scores.
    """
    model = get_ann_model()
    

    
    input_array = np.array([patient_data], dtype=np.float32)
    
    # 2. Get real predictions from Keras model
    try:
        
        predictions = model.predict(input_array, verbose=0)[0]
        
        # Map the array outputs to your specific patterns based on training order
        scores = {
            "bmi_pattern": float(predictions[0]),
            "heart_rate_pattern": float(predictions[1]),
            "cholesterol_pattern": float(predictions[2]),
            "sleep_pattern": float(predictions[3])
        }
    except Exception as e:
        print(f"Model prediction failed: {e}. Using mock data.")
        
        scores = {
            "bmi_pattern": 0.828,
            "heart_rate_pattern": 0.408,
            "cholesterol_pattern": 0.304,
            "sleep_pattern": 0.291
        }
    
    
    results = {}
    for key, score in scores.items():
        results[key] = {
            "model_score": round(score * 100, 1),
            "remaining": round((1 - score) * 100, 1)
        }
        
    return results