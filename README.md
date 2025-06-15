# 🧠 Cardiomegaly Detection from Chest X-ray Images

A deep learning project for detecting **Cardiomegaly (enlarged heart)** from chest X-ray images using transfer learning with **CheXNet** and **DenseNet121**.

---

## 🩺 Project Objective

The goal of this project is to develop an intelligent model capable of detecting **cardiomegaly** in chest X-ray images with high accuracy. Early detection of this condition is vital for preventing serious cardiovascular complications.

We focus on **binary classification**:
- 🔴 **Cardiomegaly** (class 1)
- 🟢 **Normal** (class 0)

This model aims to assist radiologists by offering a second opinion and supporting clinical decisions.

---

## 📊 Dataset Summary

We use a custom chest X-ray dataset, carefully balanced across the training, validation, and test sets. All images are grayscale and categorized as either “Normal” or “Cardiomegaly”.

| Dataset Split | Total Images | Cardiomegaly | Normal |
|---------------|--------------|--------------|--------|
| Training       | 3,865         | 1,934 (50%)   | 1,931 (50%) |
| Validation     | 1,103         | 560 (51%)     | 543 (49%)   |
| Test           | 551           | 253 (46%)     | 298 (54%)   |

All images are preprocessed to size **224×224** and normalized to match the input expectations of the model.

---

## 🧠 Model Architecture

We combine **CheXNet’s pretrained weights** (via `torchxrayvision`) with a custom **DenseNet121** model from `torchvision`. Here's the core idea:

- Input images are grayscale (1 channel), so we **adapt the first convolutional layer**.
- We **load pretrained weights** from CheXNet (except the final classification layer).
- We **replace the classifier** with a new fully connected layer for **binary classification** (1 neuron + sigmoid).
- Training is done using **binary cross-entropy loss**.

This hybrid approach takes advantage of domain-specific pretrained features while giving us flexibility to fine-tune the classification for our specific task.

---

## 🛠️ Key Implementation Steps

- 🧪 **Transforms**: Resize → Tensor → Normalize (mean=0.5, std=0.5)
- 🏷️ **Loss**: `nn.BCEWithLogitsLoss()`
- 📈 **Evaluation**: Accuracy, Loss, Classification Report, Confusion Matrix
- 🖼️ **Inference**: Load single image, preprocess, predict using sigmoid threshold (> 0.5 = Cardiomegaly)

---


