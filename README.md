# Plan-Leaf-Image-Classification
Image Classification repo for Plant Leaf Detection.
Dataset Link: https://www.kaggle.com/datasets/mahmoudshaheen1134/plant-leaf-image-dataset

# Image Classification with PyTorch

This project demonstrates how to train, evaluate, and deploy an image classification model using PyTorch. The implementation includes custom model training, validation, checkpointing, early stopping, and inference pipelines, complete with image preprocessing and result visualization.

## Features

- **Model Training**: Supports training from scratch and fine-tuning pretrained models.
- **Validation and Early Stopping**: Includes validation tracking and early stopping to optimize training.
- **Model Checkpointing**: Automatically saves the best-performing model during training.
- **Inference Pipeline**: Provides an end-to-end inference pipeline to predict image classes from input image files.
- **Visualization**: Displays input images with predicted class labels using Matplotlib.

---

## Project Structure

```plaintext
├── checkpoints/          # Directory to save model checkpoints
├── data/                 # Directory to store input images
├── logs/                 # Directory for TensorBoard logs
├── main.py               # Main script to train, evaluate, and infer
├── model.py              # Defines the model architecture
├── utils.py              # Utility functions (e.g., preprocessing, checkpointing)
└── README.md             # Project documentation
```

---

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/timothyafolami/Plan-Leaf-Image-Classification.git
   cd Plan-Leaf-Image-Classification
   ```

2. **Install Dependencies**:
   Make sure Python 3.7+ and `pip` are installed. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:
   ```plaintext
    torch
    torchvision
    torchaudio
    opencv-python==4.10.0.84
    tqdm==4.67.1
    scikit-learn==1.6.1
    matplotlib==3.10.0
    tensorboard==2.18.0
   ```
   
3. **Prepare Data**:
   Place your input images in the `data/` directory.

4. **Train the Model**:
   Run the `main.py` script to train the model:
   ```bash
   python main.py --train
   ```

5. **Perform Inference**:
   Use the trained model to make predictions on new images:
   ```bash
   python main.py --infer --image_path data/example_image.jpg
   ```

---

## Example Usage

### Training

- The `train_model` function handles training, validation, checkpointing, and early stopping. Logs are generated for monitoring progress using TensorBoard.

### Inference

- Use the `predict_image` function to classify new images. Example:
  ```python
  input_image, predicted_class = predict_image(model, 'data/example_image.jpg', class_names, device)
  display_prediction(input_image, predicted_class)
  ```

### Visualization

- Results are displayed using Matplotlib:
  - The input image is shown alongside the predicted class label.

---

## Customization

- **Model Architecture**: Modify `model.py` to define a custom model or use pretrained models from `torchvision.models`.
- **Training Parameters**: Adjust hyperparameters like learning rate, batch size, and number of epochs in `main.py`.
- **Data Augmentation**: Customize preprocessing pipelines in `utils.py`.

---

## Results

Here’s an example of the model's prediction on a sample image:

![Example Result](assets/example_result.png)

---

## Acknowledgments

- [PyTorch Documentation](https://pytorch.org/docs/)
- [TorchVision Models](https://pytorch.org/vision/stable/models.html)
- [Pillow (PIL)](https://python-pillow.org/)
- [Matplotlib](https://matplotlib.org/)

---

