import os

os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from transformers import AutoTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Setup Paths
MODEL_DIR = "artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)
TRANSFORMER_NAME = "dmis-lab/biobert-base-cased-v1.2" 

# 2. Create Dummy Data
print("Generating dummy data...")
np.random.seed(42)
n_samples = 200
df = pd.DataFrame({
    "age": np.random.randint(20, 80, n_samples),
    "bmi": np.random.uniform(15, 40, n_samples),
    "glucose": np.random.uniform(70, 200, n_samples),
    "symptoms_text": ["I feel dizzy and tired." if i % 2 == 0 else "I feel fine." for i in range(n_samples)],
    "target": np.random.randint(0, 2, n_samples) 
})

# 3. Tokenize Text
print("Loading Transformer Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)


encodings = tokenizer(
    df["symptoms_text"].tolist(), 
    padding="max_length", 
    truncation=True, 
    max_length=64, 
    return_tensors="tf"
)

# Convert TF tensors to NumPy arrays
X_ids_np = encodings['input_ids'].numpy()
X_mask_np = encodings['attention_mask'].numpy()

# Verify shapes
print(f"Numeric Shape: {df[['age', 'bmi', 'glucose']].values.shape}")
print(f"Text Shape: {X_ids_np.shape}") 

# 4. Scale Numeric Data
numeric_features = ["age", "bmi", "glucose"]
X_numeric = df[numeric_features].values
y = df["target"].values

scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)
joblib.dump(scaler, os.path.join(MODEL_DIR, "numeric_scaler.joblib"))

# Train/Test Split
X_train_num, X_test_num, X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
    X_numeric_scaled, X_ids_np, X_mask_np, y, test_size=0.2, random_state=42
)

# 5. Build Keras Multi-Modal Model
print("Building Keras Model...")
num_input = layers.Input(shape=(len(numeric_features),), name="numeric_input")
x = layers.Dense(64, activation='relu')(num_input)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation='relu')(x)

text_input_ids = layers.Input(shape=(64,), dtype=tf.int32, name="text_ids")
text_input_mask = layers.Input(shape=(64,), dtype=tf.int32, name="text_mask")


base_transformer = TFBertModel.from_pretrained(TRANSFORMER_NAME, from_pt=True)
base_transformer.trainable = False 
transformer_output = base_transformer(text_input_ids, attention_mask=text_input_mask).pooler_output
t = layers.Dense(32, activation='relu')(transformer_output)


combined = layers.Concatenate()([x, t])
combined = layers.Dense(64, activation='relu')(combined)
combined = layers.Dropout(0.4)(combined)
output = layers.Dense(1, activation='sigmoid', name="output")(combined)

model = models.Model(inputs=[num_input, text_input_ids, text_input_mask], outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# 6. Train and Save
print("Training model...")
model.fit(
    [X_train_num, X_train_ids, X_train_mask], y_train, 
    validation_data=([X_test_num, X_test_ids, X_test_mask], y_test), 
    epochs=2, batch_size=32
)

model_path = os.path.join(MODEL_DIR, "keras_multimodal.keras")
model.save(model_path)
print(f" Success! Model saved to {model_path}")