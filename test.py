import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import joblib
import time
from sklearn.utils import resample
import tensorflow as tf

print("Loading test data...")
test_data = pd.read_csv('data/archive/emnist-balanced-test.csv')
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

print("Preprocessing test data...")
X_test = X_test / 255.0
X_test = X_test.reshape(X_test.shape[0], -1)

print("Loading scaler and quantized model...")
scaler = joblib.load('MLP_scaler.pkl')
X_test = scaler.transform(X_test)

interpreter = tf.lite.Interpreter(model_path="trained_model_quantized.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Evaluating the quantized model on test data with bootstrapping...")
n_iterations = 10
bootstrap_accuracies = []

start_cpu = time.process_time()

for i in range(n_iterations):
    print(f"Bootstrap iteration {i + 1}/{n_iterations}...")
    X_bootstrap, y_bootstrap = resample(X_test, y_test, replace=True, random_state=53 + i)
    y_pred = []
    for sample in X_bootstrap:
        input_data = np.expand_dims(sample.astype(np.float32), axis=0)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        y_pred.append(np.argmax(output))
    accuracy = accuracy_score(y_bootstrap, y_pred)
    bootstrap_accuracies.append(accuracy)

average_accuracy = np.mean(bootstrap_accuracies)
print(f"Average Test Accuracy (Bootstrap): {average_accuracy:.2f}")

end_cpu = time.process_time()
print(f"CPU time used: {end_cpu - start_cpu:.4f} seconds")