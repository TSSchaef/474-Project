import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn.utils import resample

print("Loading test data...")
test_data = pd.read_csv('data/archive/emnist-balanced-test.csv')
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

print("Preprocessing test data...")
X_test = X_test / 255.0
X_test = X_test.reshape(X_test.shape[0], -1)

print("Loading the trained model and scaler...")
mlp = joblib.load('trained_model.pkl')
scaler = joblib.load('scaler.pkl')
X_test = scaler.transform(X_test)

print("Evaluating the model on test data with bootstrapping...")
n_iterations = 10  # Number of bootstrap samples
bootstrap_accuracies = []

for i in range(n_iterations):
    print(f"Bootstrap iteration {i + 1}/{n_iterations}...")
    X_bootstrap, y_bootstrap = resample(X_test, y_test, replace=True, random_state=42 + i)
    y_pred = mlp.predict(X_bootstrap)
    accuracy = accuracy_score(y_bootstrap, y_pred)
    bootstrap_accuracies.append(accuracy)

average_accuracy = np.mean(bootstrap_accuracies)
print(f"Average Test Accuracy (Bootstrap): {average_accuracy:.2f}")

print("Saving predictions of the last bootstrap sample to 'test_predictions.csv'...")
np.savetxt('test_predictions.csv', y_pred, delimiter=',', fmt='%d')
