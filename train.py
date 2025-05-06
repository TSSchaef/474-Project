import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
import joblib

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
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Adding EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=256,
    verbose=1,
    callbacks=[early_stopping]  # Pass the callback here
)

accuracy_history = history.history['accuracy']
val_accuracy_history = history.history['val_accuracy']
np.save('accuracy_history.npy', accuracy_history)
np.save('val_accuracy_history.npy', val_accuracy_history)

print("Evaluating the model on validation data...")
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_accuracy:.4f}")

print("Plotting training and validation accuracy...")
plt.figure(figsize=(10, 6))
plt.plot(accuracy_history, label='Training Accuracy')
plt.plot(val_accuracy_history, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')
plt.show()

print("Quantizing and saving the trained model...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("trained_model_quantized.tflite", "wb") as f:
    f.write(tflite_model)

joblib.dump(scaler, 'MLP_scaler.pkl')
print("Quantized model and scaler saved.")
