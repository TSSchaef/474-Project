import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from cnn_model import create_cnn_model  # Ensure this matches the function in cnn_model.py
from training_plot import TrainingPlot  # Import the custom callback

print("Loading training data...")
train_data = pd.read_csv('data/archive/emnist-balanced-train.csv')
X = train_data.iloc[:, 1:].values
y = train_data.iloc[:, 0].values

print("Preprocessing training data...")
X = X / 255.0
X = X.reshape(-1, 28, 28, 1)  # Reshape for CNN input

print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print("Standardizing features...")
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(-1, 28, 28, 1)
X_val = scaler.transform(X_val.reshape(X_val.shape[0], -1)).reshape(-1, 28, 28, 1)

print("Converting labels to categorical format...")
y_train = to_categorical(y_train, num_classes=47)
y_val = to_categorical(y_val, num_classes=47)

print("Creating the CNN model...")
cnn = create_cnn_model(input_shape=(28, 28, 1), num_classes=47)

print("Setting up callbacks...")
training_plot = TrainingPlot(output_path="training_plot.png")
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Training the CNN model...")
cnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=6,
    batch_size=128,
    callbacks=[training_plot, early_stopping]  # Add the TrainingPlot callback
)

print("Evaluating the model on validation data...")
val_loss, val_accuracy = cnn.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.2f}")

print("Saving the trained CNN model and scaler...")
import joblib
cnn.save('trained_cnn_model.keras')  # Updated to use the .keras format
joblib.dump(scaler, 'cnn_scaler.pkl')