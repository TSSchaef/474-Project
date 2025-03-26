import argparse
import numpy as np
from model import CNN
from emnist import extract_training_samples

def cross_entropy_loss(predictions, labels):
    batch_size = predictions.shape[0]
    epsilon = 1e-8 # to avoid log(0) 
    return -np.sum(np.log(predictions[np.arange(batch_size), labels] + epsilon)) / batch_size

def compute_accuracy(predictions, labels):
    return np.mean(np.argmax(predictions, axis=1) == labels)

def train(model, X_train, y_train, epochs=10, learning_rate=0.001, batch_size=32):
    num_samples = X_train.shape[0]
    for epoch in range(epochs):
        indices = np.random.permutation(num_samples)
        X_train, y_train = X_train[indices], y_train[indices]
        
        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            
            predictions = model.forward(X_batch)
            loss = cross_entropy_loss(predictions, y_batch)
            acc = compute_accuracy(predictions, y_batch)
            
            # Backpropagation
            model.backward(X_batch, y_batch)
            
            model.update_weights(learning_rate)


            print(f"Epoch {epoch+1}, Batch {i//batch_size+1}: Loss={loss:.4f}, Accuracy={acc:.4f}")
            

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Train a CNN on the EMNIST dataset.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for training')
    
    args = parser.parse_args()

    # Load training data
    X_train, y_train = extract_training_samples()
    X_train = X_train / 255.0  # Normalize pixel values
    X_train = X_train.reshape(-1, 1, 28, 28)  # Reshape for CNN input

    # Define the model configuration
    conv_config = [(8, 3), (16, 3)]  # Two convolutional layers
    fc_sizes = [128]  # One hidden FC layer
    model = CNN(input_shape=(1, 28, 28), conv_config=conv_config, fc_sizes=fc_sizes, num_classes=47)

    # Train the model with parameters from the command line
    train(model, X_train, y_train, epochs=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size)

