# Facial Emotion Recognition using ResNet-50

This project implements an emotion recognition system using facial images. It extracts facial features and classifies emotions using a fine-tuned ResNet-50 model trained on six categories: anger, disgust, fear, happiness, pain, and sadness.

## Table of Contents
1. [Installation](#installation)
2. [Dependencies](#dependencies)
3. [How to Use](#how-to-use)
4. [File Structure](#file-structure)
5. [Model Architecture](#model-architecture)
6. [Results](#results)

## Installation

To run this project, clone the repository and install the dependencies listed below.

```bash
git clone https://github.com/your-username/facial-emotion-recognition.git
cd facial-emotion-recognition

Here you go — starting **from the Dependencies section**, written in the **exact same format** as your LSTM project README:

---

### Dependencies

This project requires the following Python libraries:

- `tensorflow` (for ResNet-50 model)
- `opencv-python` (for image processing)
- `numpy` (for numerical operations)
- `flask` (for creating the backend API)
- `Pillow` (for image handling)

You can install these dependencies by running:

pip install -r requirements.txt


Alternatively, you can install them individually using:

pip install tensorflow opencv-python numpy flask Pillow


## How to Use

### 1. Prepare your facial image

The system accepts `.jpg` or `.png` images of facial expressions. Place your image in the `input_images` folder or upload it using the web interface.

### 2. Run the Web App

Start the Flask backend server:

```bash
python app.py
```

Then, open your browser and go to `http://localhost:5000` to access the web interface. Upload an image and click “Predict” to view the predicted emotion.

### 3. Use the API directly (optional)

You can also send requests directly to the API using `curl`:

```bash
curl -X POST -F "image=@your_image.jpg" http://localhost:5000/predict
```

## File Structure

The repository is organized as follows:

```
facial-emotion-recognition/
│
├── input_images/              # Folder for input test images
├── model/                     # Folder for storing the trained model
├── static/                    # Static files for frontend (CSS/JS)
├── templates/                 # HTML templates for the web interface
├── app.py                     # Flask backend logic
├── preprocess.py              # Preprocessing utilities
├── predict.py                 # Emotion prediction functions
├── requirements.txt           # List of required packages
├── README.md                  # Project documentation
```

## Model Architecture

The model is based on a modified ResNet-50 architecture:

1. **Base Model**: Pretrained ResNet-50 with ImageNet weights  
2. **Feature Extraction**: GlobalAveragePooling applied to extracted features  
3. **Dropout Layer**: 50% dropout for regularization  
4. **Dense Output Layer**: Final dense layer with 6 units and softmax activation  

The model is trained using categorical cross-entropy loss and the Adam optimizer.

## Results

The model achieves the following performance:

- **Validation Accuracy**: 72%
- **Test Accuracy**: 74%

### Supported Emotions

- Anger  
- Disgust  
- Fear  
- Happiness  
- Pain  
- Sadness

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)  
- [ResNet-50 Basics](https://wandb.ai/mostafaibrahim17/ml-articles/reports/The-Basics-of-ResNet50---Vmlldzo2NDkwNDE2)  
- [Kaggle Dataset](https://www.kaggle.com/datasets/yousefmohamed20/sentiment-images-classifier)
```
