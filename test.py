import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tensorflow.keras.models import load_model
from sklearn.utils import resample

print("Loading test data...")
test_data = pd.read_csv('data/archive/emnist-balanced-test.csv')
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

print("Preprocessing test data...")
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)  # Reshape for CNN input

print("Loading the trained CNN model and scaler...")
cnn = load_model('trained_cnn_model.keras')  # Updated to load the .keras file
scaler = joblib.load('cnn_scaler.pkl')
X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(-1, 28, 28, 1)

print("Evaluating the model on test data with bootstrapping...")
n_iterations = 10  # Number of bootstrap samples
bootstrap_accuracies = []

for i in range(n_iterations):
    print(f"Bootstrap iteration {i + 1}/{n_iterations}...")
    X_bootstrap, y_bootstrap = resample(X_test, y_test, replace=True, random_state=53 + i)
    y_pred = cnn.predict(X_bootstrap)
    y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = accuracy_score(y_bootstrap, y_pred_classes)
    bootstrap_accuracies.append(accuracy)

average_accuracy = np.mean(bootstrap_accuracies)
print(f"Average Test Accuracy (Bootstrap): {average_accuracy:.2f}")