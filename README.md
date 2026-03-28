# 🧠 Brain Tumor Classification using CNN

A deep learning project that classifies brain MRI scans into **4 tumor categories** using a Convolutional Neural Network (CNN) built with TensorFlow/Keras. Features an interactive **Streamlit web app** for real-time predictions.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?logo=streamlit)
![Accuracy](https://img.shields.io/badge/Test%20Accuracy-~89%25-green)

---

## 🎯 Problem Statement

Brain tumors are life-threatening conditions where early detection is critical. This project automates MRI scan analysis using deep learning, classifying tumors into:

| Class | Description |
|-------|-------------|
| 🔴 Glioma | Tumor originating in glial cells |
| 🟠 Meningioma | Usually benign, arises from meninges |
| 🟡 Pituitary | Located in the pituitary gland |
| 🟢 No Tumor | Healthy MRI scan |

---

## 🏗️ Project Structure

```
brain-tumor-classifier/
│
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
│
├── model/
│   └── train.py            # CNN model definition + training script
│
├── utils/
│   └── predict.py          # Inference utilities
│
├── assets/                 # Saved plots (confusion matrix, training curves)
├── data/                   # Dataset (download from Kaggle — see below)
│   ├── Training/
│   └── Testing/
└── README.md
```

---

## 🧬 Model Architecture

```
Input (224×224×3)
    ↓
Conv Block 1  →  Conv2D(32) + BN + Conv2D(32) + MaxPool + Dropout(0.25)
    ↓
Conv Block 2  →  Conv2D(64) + BN + Conv2D(64) + MaxPool + Dropout(0.25)
    ↓
Conv Block 3  →  Conv2D(128) + BN + Conv2D(128) + MaxPool + Dropout(0.40)
    ↓
Global Average Pooling
    ↓
Dense(256) + BN + Dropout(0.5)
    ↓
Dense(4, softmax)  →  Output
```

**Training Techniques:**
- Data Augmentation (rotation, flip, zoom, shift)
- Batch Normalization for stable training
- Early Stopping + ReduceLROnPlateau callbacks
- Adam optimizer (lr = 1e-4)

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~89% |
| Loss Function | Categorical Cross-entropy |
| Optimizer | Adam |
| Epochs trained | ~18 (early stopping) |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/brain-tumor-classifier.git
cd brain-tumor-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from Kaggle: [Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)

Extract into a `data/` folder:
```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

### 4. Train the model
```bash
python model/train.py
```
The trained model is saved to `model/brain_tumor_cnn.h5`.

### 5. Launch the web app
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🖥️ Web App Demo

Upload any brain MRI image and get:
- Predicted tumor class
- Confidence score
- Probability distribution chart

---

## 🧪 Key Concepts Demonstrated

- Convolutional Neural Networks (CNN)
- Image Preprocessing & Augmentation
- Transfer Learning concepts (BatchNorm, Dropout)
- Model Evaluation (Confusion Matrix, Classification Report)
- Deployment with Streamlit

---

## 📚 Dataset

**Source:** [Kaggle — Brain Tumor Classification MRI](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)  
**Size:** ~3,000 MRI images across 4 classes  
**Format:** JPEG images, pre-split into Training/Testing folders

---

## ⚠️ Disclaimer

This project is built for **educational purposes only**. It is not a substitute for professional medical diagnosis. Always consult a licensed medical professional for health concerns.

---

## 👤 Author

Khushi V.R Yewale  
B.Tech [AI]
[LinkedIn](https://www.linkedin.com/in/khushi-yewale-44b898375/) · [GitHub](https://github.com/khushiy05)
