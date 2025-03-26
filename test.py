import numpy as np
from model import CNN
from emnist import extract_test_samples

def compute_accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == labels)

def test(model, X_test, y_test):
    predictions = model.forward(X_test)
    accuracy = compute_accuracy(predictions, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    X_test, y_test = extract_test_samples()
    X_test = X_test / 255.0  # Normalize pixel values
    X_test = X_test.reshape(-1, 1, 28, 28)  # Reshape for CNN input
    
    conv_config = [(8, 3), (16, 3)]  # Same architecture as training
    fc_sizes = [128]
    model = CNN(input_shape=(1, 28, 28), conv_config=conv_config, fc_sizes=fc_sizes, num_classes=47)
    
    test(model, X_test, y_test)

