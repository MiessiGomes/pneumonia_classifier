from typing import List, Tuple

import numpy as np
import tensorflow as tf

from pneumonia_classifier import config

AUTOTUNE: int = tf.data.AUTOTUNE


def create_data_pipelines() -> (
    Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str], np.ndarray]
):
    """Creates the training, validation, and test data pipelines."""
    train_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        config.TRAIN_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        interpolation="bilinear",
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        color_mode="grayscale",
    )

    val_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        config.VAL_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        interpolation="bilinear",
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        color_mode="grayscale",
    )

    test_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
        config.TEST_DIR,
        labels="inferred",
        label_mode="categorical",
        image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        interpolation="bilinear",
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        color_mode="grayscale",
    )

    class_names: List[str] = train_ds.class_names
    test_labels_one_hot: np.ndarray = np.concatenate([y for x, y in test_ds], axis=0)

    return train_ds, val_ds, test_ds, class_names, test_labels_one_hot


def preprocess_data(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Rescales pixel values to [0, 1]."""
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def get_data_augmentation_layer() -> tf.keras.Sequential:
    """Returns a Sequential model with data augmentation layers."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
        ]
    )


def prepare_datasets(
    train_ds: tf.data.Dataset, val_ds: tf.data.Dataset, test_ds: tf.data.Dataset
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Applies preprocessing and augmentation to the datasets."""
    data_augmentation: tf.keras.Sequential = get_data_augmentation_layer()

    train_ds = train_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE,
    )
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    val_ds = val_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = test_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE).prefetch(
        buffer_size=AUTOTUNE
    )

    return train_ds, val_ds, test_ds
