# 🗑️ Garbage Classification System using MobileNetV2

This project is a deep learning-based garbage classification model that uses transfer learning with EfficientnetV2B2 & MobileNetV2 to classify images into six waste categories:

📦 Cardboard | 🧪 Glass | 🛠️ Metal | 📄 Paper | 🧴 Plastic | 🚮 Trash

---

## 🎯 Goal
To develop an **accurate and efficient image classification model** for automated garbage sorting using transfer learning and data augmentation techniques.

---

## 📁 Dataset
- Total Images: **3600**
- Classes: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`
- Split: **80% training**, **20% validation**

---

## 🛠️ Tools & Technologies

| Tool           | Purpose                          |
|----------------|----------------------------------|
| Python         | Core programming language        |
| TensorFlow/Keras | Deep learning framework         |
| MobileNetV2    | Pretrained model architecture    |
| Matplotlib     | Accuracy/Loss plots              |
| Seaborn        | Confusion matrix heatmap         |
| scikit-learn   | Classification metrics           |

---

## 🔍 Methodology

1. ✅ Preprocess & augment image dataset  
2. 🧠 Use MobileNetV2 pretrained on ImageNet  
3. 🧪 Freeze base model → Train classification head  
4. 🔁 Unfreeze + fine-tune all layers  
5. ⚖️ Apply **class weights** to balance classes  
6. 📊 Evaluate using confusion matrix, accuracy, loss  

---

## 📈 Accuracy & Loss Plots

![Accuracy & Loss](screenshots/accuracy_loss_graph.png)

---

## 🧩 Confusion Matrix

![Confusion Matrix](screenshots/confusion_matrix.png)

---

## 🧪 Final Results

| Metric         | Value           |
|----------------|-----------------|
| Final Accuracy | **71.25%**      |
| Epochs         | 30 (incl. fine-tuning) |
| Best Classes   | Plastic, Glass, Paper |
| Challenging    | Trash, Metal, Cardboard |

---

## 🚀 Next Steps

- Add more data and balance class distribution  
- Try other lightweight models (e.g., MobileNetV3, ResNet50)  
- Improve class labels and augmentation techniques  
- Deploy using Flask, FastAPI, or Gradio (future scope)  

---

## 💾 Model Checkpoint

Model saved as: `garbage_classification_model.h5`

---
