import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from pneumonia_classifier import config
from pneumonia_classifier.data_pipeline import create_data_pipelines, prepare_datasets
from pneumonia_classifier.model import build_model
from pneumonia_classifier.utils import calculate_class_weights, save_plot


def train() -> None:
    """Trains the model and saves it."""
    (train_ds, val_ds, test_ds, class_names, _) = create_data_pipelines()
    (train_ds, val_ds, _) = prepare_datasets(train_ds, val_ds, test_ds)

    class_weights = calculate_class_weights(train_ds)

    # Build and compile the model
    model = build_model(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Define callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001
    )

    history = model.fit(
        train_ds,
        epochs=config.EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=[early_stopping, reduce_lr],
    )

    model.save(config.MODEL_PATH)
    print(f"Model saved to {config.MODEL_PATH}")

    save_plot(history, "accuracy", config.REPORTS_DIR / "training_accuracy.png")
    save_plot(history, "loss", config.REPORTS_DIR / "training_loss.png")
    print(f"Training plots saved to {config.REPORTS_DIR}")


if __name__ == "__main__":
    train()
