import pandas as pd
import numpy as np
from keras.models import load_model
import tensorflow_model_optimization as tfmot
import tensorflow as tf

# === Load and preprocess data ===
print("Loading model data...")
cnn = load_model("trained_cnn_model.keras")

# === Convert to quantized TFLite model ===
print("Converting to quantized TFLite model...")
converter = tf.lite.TFLiteConverter.from_keras_model(cnn)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("trained_cnn_model_quantized.tflite", "wb") as f:
    f.write(tflite_model)


print("Quantized model saved")

