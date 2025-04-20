# 🎧 Audio Speech Sentiment Classification

This project focuses on building deep learning models to classify spoken audio clips into three sentiment categories: **Positive**, **Neutral**, and **Negative**. It uses both **Artificial Neural Network (ANN)** and **Convolutional Neural Network (CNN)** architectures to analyze audio features and determine the emotional tone of speech.

---

## 📁 Dataset

**Dataset Name**: [Audio Speech Sentiment Dataset](https://www.kaggle.com/datasets/imsparsh/audio-speech-sentiment) (Available on Kaggle)

**Structure**:
- `TRAIN.csv`: Metadata file with two columns:
  - `Filename`: Audio file name
  - `Class`: Sentiment label (`Positive`, `Neutral`, `Negative`)
- `TRAIN/`: Directory containing labeled audio training files (`.wav`)
- `TEST/`: Directory containing unlabeled audio files for inference/testing

**Classes**:
- `Positive`
- `Neutral`
- `Negative`

**Features Used**:
- **MFCC (Mel-Frequency Cepstral Coefficients)**: 40 coefficients per audio clip
- Averaged over time to obtain a fixed 40-dimensional vector per sample

---

## 🧠 Model Architectures

### 1. 🔹 Artificial Neural Network (ANN)
- Input: 40 MFCC features
- Layers:
  - Dense(128) + ReLU + Dropout(0.3)
  - Dense(64) + ReLU + Dropout(0.3)
  - Output: Dense(3) + Softmax
- Optimizer: Adam
- Loss: Categorical Crossentropy

### 2. 🔹 Convolutional Neural Network (CNN)
- Input: MFCC features reshaped to `(40, 1, 1)`
- Layers:
  - Conv2D(32, kernel_size=(3,1)) + ReLU
  - MaxPooling2D(pool_size=(2,1))
  - Flatten
  - Dense(64) + ReLU + Dropout(0.5)
  - Output: Dense(3) + Softmax
- Optimizer: Adam
- Loss: Categorical Crossentropy

---

## 📈 Training and Evaluation

- Dataset split: 80% Training, 20% Validation
- Callbacks used:
  - `EarlyStopping`
  - `ReduceLROnPlateau`

### Evaluation Metrics:
- Accuracy & Loss curves
- Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix

---

## 📊 Results

| Model | Validation Accuracy | Validation Loss | Highlights |
|-------|---------------------|-----------------|------------|
| ANN   | 78.00%              | 0.6255          | Lightweight, Fast |
| CNN   | 90.00%              | 0.2371          | Better at learning spatial audio features |

> 🧠 **Observation**: The CNN model significantly outperforms the ANN by capturing spatial relationships in MFCC features, leading to a 12% improvement in accuracy and a lower validation loss.

---

## 🖼️ Visualizations

- **Waveform Plot** of audio signals
- **Spectrogram Plot** (using STFT)
- **Training Curves** for Accuracy and Loss
- **Confusion Matrix Heatmap**

---

## 🔧 Setup Instructions

### 📦 Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn librosa scikit-learn tensorflow
