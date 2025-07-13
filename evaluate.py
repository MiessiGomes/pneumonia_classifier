import numpy as np
import tensorflow as tf

from pneumonia_classifier import config
from pneumonia_classifier.data_pipeline import create_data_pipelines, preprocess_data
from pneumonia_classifier.utils import (
    save_classification_report,
    save_confusion_matrix,
    save_roc_curve,
)


def evaluate() -> None:
    """Evaluates the trained model and saves the reports."""
    model = tf.keras.models.load_model(config.MODEL_PATH)

    (_, _, test_ds, class_names, test_labels_one_hot) = create_data_pipelines()

    test_ds = test_ds.map(
        preprocess_data, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(buffer_size=tf.data.AUTOTUNE)

    y_pred_proba = model.predict(test_ds)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(test_labels_one_hot, axis=1)

    save_classification_report(
        y_true, y_pred, class_names, config.REPORTS_DIR / "classification_report.txt"
    )
    save_confusion_matrix(
        y_true, y_pred, class_names, config.REPORTS_DIR / "confusion_matrix.png"
    )
    save_roc_curve(
        y_true, y_pred_proba, class_names, config.REPORTS_DIR / "roc_curve.png"
    )
    print(f"Evaluation reports saved to {config.REPORTS_DIR}")


if __name__ == "__main__":
    evaluate()
