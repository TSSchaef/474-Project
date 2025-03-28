from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization

def create_cnn_model(input_shape=(28, 28, 1), num_classes=47):
    """
    Creates and returns an improved CNN model.
    
    Args:
        input_shape (tuple): Shape of the input data (default is (28, 28, 1)).
        num_classes (int): Number of output classes (default is 47).
    
    Returns:
        model (Sequential): Compiled CNN model.
    """
    print("Creating a CNN model...")
    model = Sequential([
        Conv2D(128, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.07),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.1),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.1),

        GlobalAveragePooling2D(),
        Dense(256, activation='relu'),
        Dropout(0.1),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Improved CNN model created and compiled.")
    return model
