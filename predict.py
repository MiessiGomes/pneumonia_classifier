import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from pneumonia_classifier import config


def predict_single_image(model_path: Path, img_path: Path) -> None:
    """Predicts the class of a single image."""
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the image
    img = image.load_img(
        img_path,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        color_mode="grayscale",
    )
    img_array = image.img_to_array(img)
    img_array /= 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_batch)[0]
    predicted_class_index = np.argmax(prediction)

    # Get class names from a dummy generator
    dummy_ds = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
    )
    class_names = dummy_ds.class_names
    class_labels = {i: name for i, name in enumerate(class_names)}

    predicted_class_label = class_labels[predicted_class_index]
    confidence = 100 * np.max(prediction)

    # Visualize
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="gray")
    plt.title(f"Prediction: {predicted_class_label}\nConfidence: {confidence:.2f}%")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a single image.")
    parser.add_argument("image_path", type=str, help="Path to the image to predict.")
    args = parser.parse_args()
    predict_single_image(config.MODEL_PATH, Path(args.image_path))
