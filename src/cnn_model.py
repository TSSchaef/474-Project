from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, SeparableConv2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization, Add
)

def create_cnn_model(input_shape=(28, 28, 1), num_classes=47):
    """
    Creates an optimized CNN model for inference using depthwise separable convolutions and dense connections.
    
    Args:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of output classes.
    
    Returns:
        model (Model): Compiled CNN model.
    """
    print("Creating an optimized CNN model...")
    inputs = Input(shape=input_shape)

    # Block 1
    x1 = SeparableConv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2))(x1)

    # Block 2
    x2 = SeparableConv2D(128, (3, 3), activation='relu', padding='same')(x1)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D((2, 2))(x2)

    # Skip connection from x1 (after downsampling and projection)
    skip = MaxPooling2D((2, 2))(x1)
    skip = Conv2D(128, (1, 1), padding='same')(skip)
    x2 = Add()([x2, skip])

    # Block 3
    x3 = SeparableConv2D(256, (3, 3), activation='relu', padding='same')(x2)
    x3 = BatchNormalization()(x3)
    x3 = MaxPooling2D((2, 2))(x3)

    # Global Average Pooling & Dense Layers
    x = GlobalAveragePooling2D()(x3)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.05)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("Optimized CNN model created and compiled.")
    return model
