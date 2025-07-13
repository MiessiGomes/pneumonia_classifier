# Pneumonia Detection from Chest X-Rays

This project is a deep learning model to detect pneumonia from chest X-ray images.

## Project Structure

- `data/`: Contains the dataset.
- `output/`: Contains the trained models and evaluation reports.
- `pneumonia_classifier/`: Contains the source code for the project.
  - `config.py`: Configuration settings.
  - `data_pipeline.py`: Data loading and preprocessing.
  - `model.py`: CNN model architecture.
  - `utils.py`: Utility functions.
- `train.py`: Script to train the model.
- `evaluate.py`: Script to evaluate the model.
- `predict.py`: Script to predict a single image.
- `requirements.txt`: Project dependencies.

## Usage

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the full pipeline (train and evaluate):**
   ```bash
   python main.py
   ```

3. **Alternatively, you can run the steps individually:**
   - **Train the model:**
     ```bash
     python train.py
     ```
   - **Evaluate the model:**
     ```bash
     python evaluate.py
     ```
   - **Predict a single image:**
     ```bash
     python predict.py <path_to_image>
     ```
