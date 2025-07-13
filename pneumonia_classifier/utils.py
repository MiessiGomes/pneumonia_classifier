from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.utils.class_weight import compute_class_weight

from pneumonia_classifier import config


def calculate_class_weights(train_ds: tf.data.Dataset) -> Dict[int, float]:
    """Calculates class weights for handling data imbalance."""
    train_labels = np.concatenate([y for x, y in train_ds], axis=0)
    train_labels_indices = np.argmax(train_labels, axis=1)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels_indices),
        y=train_labels_indices,
    )
    return dict(enumerate(class_weights))


def save_plot(
    history: tf.keras.callbacks.History, plot_type: str, file_path: str
) -> None:
    """Saves plots of training history."""
    plt.figure(figsize=(12, 8))
    if plot_type == "accuracy":
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend(loc="lower right")
    elif plot_type == "loss":
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.legend(loc="upper right")
    plt.savefig(file_path)
    plt.close()


def save_classification_report(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], file_path: str
) -> None:
    """Saves the classification report to a text file."""
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(file_path, "w") as f:
        f.write(report)


def save_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], file_path: str
) -> None:
    """Saves the confusion matrix as an image."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig(file_path)
    plt.close()


def save_roc_curve(
    y_true: np.ndarray, y_pred_proba: np.ndarray, class_names: List[str], file_path: str
) -> None:
    """Saves the ROC curve as an image."""
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_pred_proba[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        plt.plot(
            fpr[i], tpr[i], label=f"{class_name} vs Rest (area = {roc_auc[i]:.2f})"
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.savefig(file_path)
    plt.close()
