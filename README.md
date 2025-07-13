# Pneumonia Detection from Chest X-Rays

This is a computer vision project to build and train a deep learning model that classifies chest X-ray images as either "Normal" or "Pneumonia".

## Dataset

The dataset used is from the following link:
**https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia**

## Key Libraries & Frameworks
- **TensorFlow (Keras):** The primary deep learning framework.
- **scikit-learn:** For evaluating model performance metrics (`classification_report`, `confusion_matrix`).
- **NumPy:** For numerical operations.
- **Matplotlib & Seaborn:** For plotting results and data visualization.
- **pathlib:** For OS-agnostic path management.
- **FastAPI:** For serving the model as a REST API.

## Project Structure
- `main.py`: Main script to run the full training and evaluation pipeline.
- `train.py`: Script to train the model.
- `evaluate.py`: Script to evaluate the trained model.
- `predict.py`: Script for predicting a single image.
- `api.py`: Exposes the trained model as a FastAPI endpoint.
- `requirements.txt`: Project dependencies.
- `data/`: Contains the dataset (ignored by git).
- `output/`: Contains trained models and evaluation reports.
- `pneumonia_classifier/`: Source code.
  - `config.py`: Centralized configuration for paths and model parameters.
  - `data_pipeline.py`: Defines the `tf.data.Dataset` pipelines for training, validation, and testing.
  - `model.py`: Contains the CNN model architecture.
  - `utils.py`: Utility functions for tasks like calculating class weights and saving evaluation plots/reports.

## CNN Model Architecture

The model is a Convolutional Neural Network (CNN) built using the Keras Sequential API. It is designed to process 128x128 grayscale images. The architecture is as follows:

1.  **Convolutional Block 1:**
    *   `Conv2D` with 32 filters, a (3, 3) kernel, and 'relu' activation.
    *   `BatchNormalization` to stabilize and accelerate training.
    *   `MaxPooling2D` with a (2, 2) pool size to downsample the feature maps.

2.  **Convolutional Block 2:**
    *   `Conv2D` with 64 filters, a (3, 3) kernel, and 'relu' activation.
    *   `BatchNormalization`.
    *   `MaxPooling2D` with a (2, 2) pool size.

3.  **Convolutional Block 3:**
    *   `Conv2D` with 128 filters, a (3, 3) kernel, and 'relu' activation.
    *   `BatchNormalization`.
    *   `MaxPooling2D` with a (2, 2) pool size.

4.  **Convolutional Block 4:**
    *   `Conv2D` with 256 filters, a (3, 3) kernel, and 'relu' activation.
    *   `BatchNormalization`.
    *   `MaxPooling2D` with a (2, 2) pool size.

5.  **Classification Head:**
    *   `Flatten` layer to convert the 2D feature maps into a 1D vector.
    *   `Dense` layer with 512 units and 'relu' activation.
    *   `BatchNormalization`.
    *   `Dropout` with a rate of 0.3 to prevent overfitting.
    *   `Dense` output layer with a 'softmax' activation function to produce class probabilities.

## How to Use

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone <repository_url>
cd pneumonia_classifier
pip install -r requirements.txt
```

### 2. Training
To train the model, run the `train.py` script:
```bash
python train.py
```

### 3. Evaluation
To evaluate the trained model, run the `evaluate.py` script:
```bash
python evaluate.py
```

### 4. Prediction
To predict a single image, use the `predict.py` script:
```bash
python predict.py <path_to_image>
```

## API Endpoint

This project includes a FastAPI endpoint to serve the trained model.

### Running the API
To start the API, run the following command:
```bash
uvicorn api:app --reload
```

### Making a Prediction
You can send a POST request to the `/predict` endpoint with an image file to get a prediction.

**Example using `curl`:**
```bash
curl -X POST -F "file=@/path/to/your/image.jpeg" http://127.0.0.1:8000/predict
```

**Expected Response:**
```json
{
  "prediction": "PNEUMONIA",
  "confidence": 99.87
}
```

## Important Commands
- **Install dependencies:** `pip install -r requirements.txt`
- **Run the full pipeline:** `python main.py`
- **Train the model:** `python train.py`
- **Evaluate the model:** `python evaluate.py`
- **Predict a single image:** `python predict.py <path_to_image>`
- **Start the API:** `uvicorn api:app --reload`

## The Model
**The generated model is not in the repository because it has exceeded the file size limit. It will be necessary to train a model and generate it to use the project.**
