from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential

from pneumonia_classifier import config


def build_model(num_classes: int) -> Sequential:
    """Builds and returns the CNN model."""
    model = Sequential(
        [
            Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 1),
            ),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(512, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation="softmax", dtype="float32"),
        ]
    )
    return model
