# 🛍️ Shoplifting Detection System

A comprehensive video classification system for detecting shoplifting activities from surveillance footage using multiple deep learning architectures, trained and evaluated on a custom dataset of 855 videos.

---

## 📋 Project Overview

The system classifies surveillance video clips into two categories — **shoplifting** and **non-shoplifting** — by learning spatio-temporal patterns across video frames. Three families of deep learning models were implemented and benchmarked, achieving **100% test accuracy** on all approaches.

The project is structured across three phases:
1. **From-Scratch Models** — Custom-designed 3D CNN, CNN+LSTM, and Video Transformer architectures
2. **Pretrained Transfer Learning** — R3D-18 backbone pretrained on Kinetics-400, fine-tuned end-to-end
3. **Deployment** — Django-based real-time inference web application

---

## 📁 Repository Structure

```
Shoplifting_Detection_Project/
├── FromScratch_Model/
│   ├── src/
│   │   ├── data_load.py       # Video loading, uniform frame sampling, augmentation pipeline
│   │   ├── model.py           # 3D CNN, CNN+LSTM, Video Transformer architectures
│   │   └── train.py           # Training loop, early stopping, TensorBoard logging
│   └── main.py                # Training entry point & hyperparameter configuration
├── PreTrained_Model/
│   ├── src/
│   │   ├── data_load.py       # Same data pipeline as from-scratch
│   │   ├── model.py           # R3D-18 pretrained model with custom classifier head
│   │   └── train.py           # Same training loop
│   └── main.py                # Training entry point
├── Deployment/                # Django web application for real-time inference
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Shoplifting videos** | 324 |
| **Non-shoplifting videos** | 531 |
| **Total videos** | 855 |
| **Video duration** | 15–20 seconds (average) |
| **Format** | MP4, fixed-camera surveillance footage |
| **Frame resolution** | Resized to 224×224 |
| **Frame sampling** | Uniform sampling of 16–20 frames per video |
| **Train / Test split** | 80% / 20% (stratified, `random_state=42`) |

---

## 🧠 Model Architectures

### From-Scratch Models

| Model | Architecture Details | Parameters |
|---|---|---|
| **3D CNN** | 4× Conv3D blocks (64→128→256→512 channels) + BatchNorm3D + MaxPool3D + AdaptiveAvgPool3D + 2 FC layers (512→256→2) + Dropout(0.5) | ~15M |
| **CNN+LSTM** | 4× Conv2D blocks (64→128→256→512) + AdaptiveAvgPool + 2-layer Bi-directional LSTM (hidden=128, dropout=0.3) + FC + Dropout(0.5) | ~12M |
| **Video Transformer** | 4× Conv2D feature extractor + Positional Embeddings + 4-layer TransformerEncoder (d_model=512, nhead=8, feedforward=2048) + GlobalAvgPool + FC | ~40M |

### Pretrained Transfer Learning Model

| Model | Backbone | Pretrained Dataset | Fine-tuning Strategy |
|---|---|---|---|
| **R3D-18** | ResNet-3D (18 layers) | Kinetics-400 | Full end-to-end fine-tuning (no frozen layers) |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| **Optimizer** | Adam |
| **Learning Rate** | 0.0001 |
| **Batch Size** | 8 |
| **Loss Function** | CrossEntropyLoss |
| **LR Scheduler** | StepLR (step=10, γ=0.1) |
| **Early Stopping Patience** | 10 epochs |
| **Max Epochs** | 20–30 |
| **Number of Frames** | 16–20 (uniform sampling) |
| **Experiment Tracking** | TensorBoard |

---

## 🔄 Data Augmentation Pipeline

Applied per-frame during training only:

| Augmentation | Probability | Parameters |
|---|---|---|
| Random Horizontal Flip | 50% | — |
| Random Brightness | 30% | α ∈ [0.7, 1.3] |
| Random Rotation | 30% | angle ∈ [−10°, +10°] |
| Random Contrast | 30% | factor ∈ [0.7, 1.3] |

---

## 📈 Results

### Performance Comparison

| Model | Test Accuracy | Test Loss | Best Train Loss | Epochs | Time/Epoch |
|---|---|---|---|---|---|
| **3D CNN** | **100%** | 0.0027 | 0.0045 | 30 | ~5m 24s |
| **CNN+LSTM** | **100%** | 0.0039 | 0.0061 | 30 | ~7m 45s |
| **CNN+LSTM (Augmented)** | **100%** | 0.0038 | 0.0104 | 20 | ~5m 30s |
| **Pretrained R3D-18** | **100%** | 0.0257 | **0.0019** | 20 | ~8m 30s |

### Evaluation Metrics (Per Model)
- Accuracy, Precision, Recall, F1-Score tracked per epoch
- Confusion matrix computed on held-out test set
- Best model checkpoint saved based on lowest validation loss

### Key Findings
- All 4 architectures achieved **100% test accuracy**, confirming that shoplifting actions have clear spatio-temporal discriminative features
- **Pretrained R3D-18** achieved the lowest training loss (0.0019), indicating the highest prediction confidence
- **3D CNN from scratch** was the fastest to train (~5m 24s/epoch), making it the best candidate for real-time deployment
- Data augmentation during CNN+LSTM training improved generalization robustness

---

## 🚀 Deployment — Django Web Application

A production-ready Django web app was developed for real-time video inference.

| Component | Details |
|---|---|
| **Framework** | Django |
| **Inference Model** | Best-performing checkpoint (Pretrained R3D-18) |
| **Input** | Uploaded MP4 surveillance video |
| **Output** | Binary classification: Shoplifting / Not Shoplifting |
| **Media Storage** | Django `media/` directory |
| **API** | REST endpoint via `shoplifting_api` Django app |

### Running the App

```bash
cd Deployment
pip install -r requirements_django.txt
python manage.py runserver
```

---

## 🛠️ Technologies & Libraries

| Category | Tools |
|---|---|
| **Deep Learning** | PyTorch, torchvision |
| **Video Processing** | OpenCV (cv2) |
| **Training Utilities** | tqdm, TensorBoard (SummaryWriter) |
| **Evaluation** | scikit-learn (accuracy, precision, recall, F1, confusion matrix) |
| **Deployment** | Django, SQLite |
| **Environment** | Kaggle (GPU training), Python 3.x |

---

## 💻 Installation & Training

```bash
# Install dependencies
pip install -r requirements.txt

# Train from-scratch models (3D CNN / CNN+LSTM / Transformer)
cd FromScratch_Model
python main.py

# Train pretrained R3D-18
cd PreTrained_Model
python main.py
```
