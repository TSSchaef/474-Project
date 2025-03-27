import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from model import create_and_tune_model

print("Loading training data...")
train_data = pd.read_csv('data/archive/emnist-balanced-train.csv')
X = train_data.iloc[:, 1:].values
y = train_data.iloc[:, 0].values

print("Preprocessing training data...")
X = X / 255.0
X = X.reshape(X.shape[0], -1)

print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Standardizing features...")
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print("Creating and tuning the model...")
mlp = create_and_tune_model(X_train, y_train)

print("Training the MLPClassifier with best parameters...")
mlp.fit(X_train, y_train)

print("Evaluating the model on validation data...")
y_pred = mlp.predict(X_val)
print(classification_report(y_val, y_pred))

print("Saving the trained model and scaler...")
import joblib
joblib.dump(mlp, 'trained_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
