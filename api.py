import io

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from pneumonia_classifier import config

app = FastAPI(title="Pneumonia Classifier API")

model = tf.keras.models.load_model(config.MODEL_PATH)

dummy_ds = tf.keras.utils.image_dataset_from_directory(
    config.TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    image_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
)
class_names = dummy_ds.class_names
class_labels = {i: name for i, name in enumerate(class_names)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file and returns a prediction.
    """
    # Read the image
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("L")
    img = img.resize((config.IMG_HEIGHT, config.IMG_WIDTH))

    # Preprocess the image
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    img_batch = np.expand_dims(img_batch, axis=-1)

    # Predict
    prediction = model.predict(img_batch)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = 100 * np.max(prediction)

    return {
        "prediction": predicted_class_label,
        "confidence": float(confidence),
    }


# uvicorn api:app --reload
