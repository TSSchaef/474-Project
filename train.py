import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

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

hyperparameters = {
    'hidden_layer_sizes': (128, 64),  # Reduced complexity
    'activation': 'logistic',
    'solver': 'adam',
    'learning_rate_init': 0.01,
    'max_iter': 1,  # Set to 1 for manual epoch control
    'random_state': 53,
    'verbose': True
}

mlp = MLPClassifier(**hyperparameters)
epochs = 50  # Define the number of epochs
early_stopping_threshold = 5  # Stop if validation accuracy doesn't improve for 5 epochs

print("Creating and tuning the model...")

accuracy_history = []
val_accuracy_history = []
no_improvement_epochs = 0
best_val_accuracy = 0

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    mlp.partial_fit(X_train, y_train, classes=np.unique(y_train))  # Use partial_fit for incremental learning
    
    # Training accuracy
    y_train_pred = mlp.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    accuracy_history.append(train_accuracy)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    
    # Validation accuracy
    y_val_pred = mlp.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_accuracy_history.append(val_accuracy)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Early stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        no_improvement_epochs = 0
    else:
        no_improvement_epochs += 1
        if no_improvement_epochs >= early_stopping_threshold:
            print("Early stopping triggered.")
            break

# Save the accuracy history for plotting
np.save('accuracy_history.npy', accuracy_history)
np.save('val_accuracy_history.npy', val_accuracy_history)

print("Evaluating the model on validation data...")
y_pred = mlp.predict(X_val)
print(classification_report(y_val, y_pred))

print("Plotting training and validation accuracy...")
plt.figure(figsize=(10, 6))
plt.plot(accuracy_history, label='Training Accuracy')
plt.plot(val_accuracy_history, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_plot.png')  # Save the plot as an image
plt.show()

print("Saving the trained model and scaler...")
import joblib
joblib.dump(mlp, 'trained_model.pkl')
joblib.dump(scaler, 'MLP_scaler.pkl')