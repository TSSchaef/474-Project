import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tensorflow.keras.models import load_model
from sklearn.utils import resample
import tensorflow as tf
import os

# === Config ===
USE_TFLITE = True  # Set to False to test the Keras model
TFLITE_MODEL_PATH = 'trained_cnn_model_quantized.tflite'

print("Loading test data...")
test_data = pd.read_csv('data/archive/emnist-balanced-test.csv')
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

print("Preprocessing test data...")
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

print("Loading scaler...")
scaler = joblib.load('cnn_scaler.pkl')
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(-1, 28, 28, 1)

# === Define prediction function ===
def predict_keras(model, X):
    preds = model.predict(X, verbose=0)
    return np.argmax(preds, axis=1)

def predict_tflite(interpreter, X):
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    input_shape = interpreter.get_input_details()[0]['shape']
    y_preds = []

    for i in range(X.shape[0]):
        input_data = np.expand_dims(X[i], axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_index)
        y_pred = np.argmax(output_data[0])
        y_preds.append(y_pred)
    return np.array(y_preds)

# === Load model ===
if USE_TFLITE:
    print("Loading TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
else:
    print("Loading Keras model...")
    cnn = load_model('trained_cnn_model.keras')

# === Bootstrap Evaluation ===
print("Evaluating model with bootstrapping...")
n_iterations = 10
bootstrap_accuracies = []

import time
# Start measuring CPU time
start_cpu = time.process_time()


for i in range(n_iterations):
    print(f"Bootstrap iteration {i + 1}/{n_iterations}...")
    X_bootstrap, y_bootstrap = resample(X_test, y_test, replace=True, random_state=53 + i)
    
    if USE_TFLITE:
        y_pred_classes = predict_tflite(interpreter, X_bootstrap)
    else:
        y_pred_classes = predict_keras(cnn, X_bootstrap)
    
    accuracy = accuracy_score(y_bootstrap, y_pred_classes)
    bootstrap_accuracies.append(accuracy)

average_accuracy = np.mean(bootstrap_accuracies)
print(f"Average Test Accuracy (Bootstrap): {average_accuracy:.2f}")

# End measuring CPU time
end_cpu = time.process_time()

print(f"CPU time used: {end_cpu - start_cpu:.4f} seconds")
