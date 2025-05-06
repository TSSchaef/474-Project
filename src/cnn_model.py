from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import (
    Input, SeparableConv2D, Conv2D, MaxPooling2D, Concatenate, GlobalAveragePooling2D,
    Dense, Dropout, BatchNormalization, Add, RandomRotation, RandomZoom
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
    x = RandomRotation(0.02)(inputs)
    x = RandomZoom(0.02)(x)

    # Block 1
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x)
    x1 = BatchNormalization()(x1)
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=l2(1e-4))(x1)
    x1 = BatchNormalization()(x1)
    x1 = MaxPooling2D((2, 2))(x1)

    # Block 2 with skip connection
    x2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same', depthwise_regularizer=l2(1e-4), pointwise_regularizer=l2(1e-4))(x1)
    x2 = BatchNormalization()(x2)
    x2 = SeparableConv2D(64, (3, 3), activation='relu', padding='same', depthwise_regularizer=l2(1e-4), pointwise_regularizer=l2(1e-4))(x2)
    x2 = BatchNormalization()(x2)
    x2 = MaxPooling2D((2, 2))(x2)

    skip = Conv2D(64, (1, 1), padding='same')(x1)
    skip = MaxPooling2D((2, 2))(skip)
    x2 = Add()([x2, skip])

    # Block 3
    x3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same', depthwise_regularizer=l2(1e-4), pointwise_regularizer=l2(1e-4))(x2)
    x3 = BatchNormalization()(x3)
    x3 = SeparableConv2D(128, (3, 3), activation='relu', padding='same', depthwise_regularizer=l2(1e-4), pointwise_regularizer=l2(1e-4))(x3)
    x3 = BatchNormalization()(x3)

    x3 = SeparableConv2D(128, (1, 1), activation='relu', padding='same', depthwise_regularizer=l2(1e-4), pointwise_regularizer=l2(1e-4))(x3)
    x3 = BatchNormalization()(x3)
    x3 = SeparableConv2D(128, (1, 1), activation='relu', padding='same', depthwise_regularizer=l2(1e-4), pointwise_regularizer=l2(1e-4))(x3)
    x3 = BatchNormalization()(x3)

    # Global Pooling & Dense
    x = GlobalAveragePooling2D()(x3)
    x = Dense(64, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate =0.001), loss=CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])
    print("Optimized CNN model created and compiled.")
    return model
