import tensorflow as tf
from tensorflow.keras import layers, models, Input

# 1. The ANN Model 
def build_health_ann(input_size, output_size=4):
    model = models.Sequential([
        Input(shape=(input_size,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2), 
        layers.Dense(32, activation='relu'),
        # Sigmoid is used because we want probabilities (0 to 1) for each pattern
        layers.Dense(output_size, activation='sigmoid') 
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 2. The Transformer Model 
def build_health_transformer(input_dim, num_heads=4, ff_dim=128):
    inputs = Input(shape=(None, input_dim))
    
    # Multi-Head Attention Layer
    attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_dim)(inputs, inputs)
    attention_output = layers.Dropout(0.1)(attention_output)
    
    
    x = layers.Add()([inputs, attention_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    # Feed Forward Network
    ff_output = layers.Dense(ff_dim, activation="relu")(x)
    ff_output = layers.Dense(input_dim)(ff_output)
    ff_output = layers.Dropout(0.1)(ff_output)
    
    
    x = layers.Add()([x, ff_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    
    
    x = layers.GlobalAveragePooling1D()(x)
    
    # Final classification layer
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model