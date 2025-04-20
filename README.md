# Facial Emotion Recognition using ResNet-50

This project implements a deep learning-based facial emotion recognition system using a fine-tuned ResNet-50 model. It classifies facial expressions into six emotion categories: **anger**, **disgust**, **fear**, **happiness**, **pain**, and **sadness**. Designed with real-time applications in mind, the system includes both a Flask backend and a simple HTML/JS frontend for end-to-end deployment.

## 🔍 Overview

Facial emotion detection enhances human-computer interaction by recognizing emotions from facial cues. This system uses **transfer learning** with pre-trained ImageNet weights on a modified ResNet-50, retrained specifically for emotion classification.

## 🧠 Model Architecture

- **Base Model:** ResNet-50 with ImageNet pre-trained weights
- **Modified Output:** Final dense layer changed to output 6 emotion classes
- **Dropout Layer:** Added to reduce overfitting
- **Activation Function:** Softmax for multi-class classification

## 🗂 Dataset

- Images categorized into six emotions
- Source: [Kaggle - Sentiment Images Classifier](https://www.kaggle.com/datasets/yousefmohamed20/sentiment-images-classifier)
- Preprocessing: Resizing to 224×224, normalization (ImageNet stats), data augmentation (cropping, rotation, color jitter)

## 🏋️‍♀️ Training Details

- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Scheduler:** ReduceLROnPlateau
- **Validation Accuracy:** 72%
- **Test Accuracy:** 74%

## 🖥 Application Flow

1. **User uploads an image** via the web interface.
2. **Image is preprocessed** (resized, normalized).
3. **Model predicts emotion** from the facial expression.
4. **Output is displayed** on the frontend.

## 🧪 Backend (Flask)

- Endpoint: `/predict`
- Accepts image uploads via POST
- Returns predicted emotion as JSON

## 🌐 Frontend (HTML/JS)

- Simple form to upload image
- Button to trigger prediction
- Displays predicted emotion

## 📈 Results

- Consistent classification across all six emotions
- Visual outputs show real-time predictions
- Ideal for applications in healthcare, surveillance, education, and AI-based user interfaces

## 💡 Key Applications

- **Healthcare:** Monitor emotional well-being
- **Education:** Gauge student engagement
- **Customer Service:** Enhance chatbot interaction
- **Gaming:** Enable emotion-adaptive gameplay

## ⚠️ Challenges

- Class imbalance in dataset
- Generalization to varied lighting and faces
- Limited dataset diversity affects accuracy

## 🚀 Future Scope

- Multimodal emotion recognition (combine voice/text)
- Edge deployment for low-resource environments
- Expanded dataset for better generalization

## 📚 References

- [ResNet Overview](https://wandb.ai/mostafaibrahim17/ml-articles/reports/The-Basics-of-ResNet50---Vmlldzo2NDkwNDE2)
- [Dataset Source](https://www.kaggle.com/datasets/yousefmohamed20/sentiment-images-classifier)

---

Feel free to contribute or adapt this project for your own use cases.
