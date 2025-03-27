import numpy as np
from scipy.signal import correlate2d

class ConvLayer:
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize filters (kernels) and biases
        # He Init
        limit = np.sqrt(2. / (in_channels * kernel_size * kernel_size))
        self.filters = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * limit
        self.biases = np.zeros(out_channels)

        # For backpropagation
        self.input = None
        self.output = None

    def forward(self, X):
        # Save input for backpropagation
        self.input = X
        batch_size, in_channels, height, width= X.shape

        # Apply padding to the input if necessary
        if self.padding > 0:
            X = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        # Output dimensions after convolution
        self.output_height = (X.shape[2] - self.kernel_size) // self.stride + 1
        self.output_width = (X.shape[3] - self.kernel_size) // self.stride + 1

        # Initialize the output of the convolutional layer
        self.output = np.zeros((X.shape[0], self.out_channels, self.output_height, self.output_width))

        # Perform convolution for each image in the batch
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                for b in range(batch_size):
                    self.output[b, i, :, :] += correlate2d(X[b, j, :, :], self.filters[i, j, :, :], mode='valid')

        # Adding biases
        self.output += self.biases.reshape(1, -1, 1, 1)
        print("Filter Norm: ", np.linalg.norm(self.filters))

        return self.output

    def backward(self, d_out):
        batch_size, out_channels, height, width= d_out.shape
        _, _, input_height, input_width = self.input.shape

        # Calculate gradients for filters and biases
        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input = np.zeros_like(self.input)

        if self.padding > 0:
            padded_input = np.pad(self.input, ((0,0), (0,0),
                                               (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
            d_padded_input = np.zeros_like(padded_input)
        else:
            padded_input = self.input
            d_padded_input = d_input


        # Calculate gradients
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                for b in range(batch_size):
                    d_filters[i, j, :, :] += correlate2d(padded_input[b, j, :, :], d_out[b, i, :, :], mode='valid')
                    d_padded_input[b, j, :, :] += correlate2d(d_out[b, i, :, :], np.flip(self.filters[i, j, :, :]), mode='full')

            d_biases[i] += np.sum(d_out[:, i, :, :])

        if self.padding > 0:
            d_input = d_padded_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_input = d_padded_input
        
        self.d_filters = d_filters
        self.d_biases = d_biases
        return d_input


    def update_weights(self, learning_rate):
        clip_value = 1.0
        self.d_filters = np.clip(self.d_filters, -clip_value, clip_value)
        self.d_biases = np.clip(self.d_biases, -clip_value, clip_value)

        # Update weights and biases using gradients
        self.filters -= learning_rate * self.d_filters
        self.biases -= learning_rate * self.d_biases


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, X):
        # Save input for backpropagation
        self.input = X
        return np.maximum(0, X)

    def backward(self, d_out):
        return d_out * (self.input > 0)

class Flatten:
    def __init__(self):
        self.input_shape = None  # Store original shape

    def forward(self, X):
        self.input_shape = X.shape  # Remember shape for backward pass
        return X.reshape(X.shape[0], -1)  # Flatten except batch dimension

    def backward(self, d_out):
        return d_out.reshape(self.input_shape)  # Restore original shape

class FullyConnected:
    def __init__(self, in_size, out_size):
        # He initialization
        limit = np.sqrt(2.0 / in_size)
        self.weights = np.random.randn(out_size, in_size) * limit 
        self.biases = np.zeros(out_size)
        self.input = None

    def forward(self, X):
        self.input = X
        output = X @ self.weights.T + self.biases
        return output

    def backward(self, d_out):
        # Compute the gradient of the loss w.r.t weights, biases, and input
        d_weights = d_out.T @ self.input
        d_biases = np.sum(d_out, axis=0)
        d_input = d_out @ self.weights

        # Store gradients for weight updates
        self.d_weights = d_weights
        self.d_biases = d_biases

        return d_input

    def update_weights(self, learning_rate):
        clip_value = 1.0
        self.d_weights = np.clip(self.d_weights, -clip_value, clip_value)
        self.d_biases = np.clip(self.d_biases, -clip_value, clip_value)

        # Update the weights and biases using the gradients
        self.weights -= learning_rate * self.d_weights
        self.biases -= learning_rate * self.d_biases


class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, X):
        # Compute softmax
        exp_values = np.exp(X - np.max(X, axis=1, keepdims=True))  # For numerical stability
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities  # Store the output
        return probabilities

    def backward(self, d_out):
        grad = self.output - d_out
        return grad

class CNN:
    def __init__(self, input_shape, conv_config, fc_sizes, num_classes):
        self.layers = []
        in_channels = input_shape[0]

        # Add convolutional layers
        for out_channels, kernel_size in conv_config:
            self.layers.append(ConvLayer(in_channels, out_channels, kernel_size))
            self.layers.append(ReLU())
            in_channels = out_channels

        # **Add a Flatten layer to correct dimensions when passing from Conv to Fully Connected**
        self.layers.append(Flatten())

        # Compute the flattened size dynamically
        dummy_input = np.zeros((1, *input_shape))
        for layer in self.layers:
            dummy_input = layer.forward(dummy_input)
        fc_input_size = dummy_input.size

        # Add fully connected layers
        for size in fc_sizes:
            self.layers.append(FullyConnected(fc_input_size, size))
            self.layers.append(ReLU())
            fc_input_size = size

        # Output layer
        self.layers.append(FullyConnected(fc_input_size, num_classes))
        self.layers.append(Softmax())

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)

        return X

    def backward(self, X_batch, y_batch):
        # Perform backward pass through each layer (from last to first)
        d_out = self.compute_loss_gradients(X_batch, y_batch)
        for layer in reversed(self.layers):
            print(f"Gradient Norm Before {layer.__class__.__name__}:", np.linalg.norm(d_out))
            d_out = layer.backward(d_out)

    def compute_loss_gradients(self, X_batch, y_batch):
        # Compute gradient of loss w.r.t softmax output
        batch_size = X_batch.shape[0]
        d_out = np.copy(self.layers[-1].output)
        d_out[np.arange(batch_size), y_batch] -= 1  # dL/dz for softmax (cross-entropy loss)
        d_out /= batch_size  # Average gradients across batch
        return d_out

    def update_weights(self, learning_rate):
        # Update weights for each layer in the CNN (Conv and FC layers)
        for layer in self.layers:
            if isinstance(layer, ConvLayer):
                layer.update_weights(learning_rate)
            elif isinstance(layer, FullyConnected):
                layer.update_weights(learning_rate)

