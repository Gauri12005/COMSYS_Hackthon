# COMSYS_Hackthon
This is For COMSYS Hackthon-2025 .
# 🧠 FACECOM Challenge: Face Recognition & Gender Classification in Challenging Environments

## 📌 Problem Statement

Face recognition algorithms often struggle with images captured in degraded environments such as blur, fog, rain, or low-light. This project addresses these challenges using the *FACECOM* dataset, aiming to build two robust models:

- *Task A:* Gender Classification (Binary)
- *Task B:* Face Recognition (Multi-class)

---

## 📂 Dataset: FACECOM

FACECOM is a purpose-built dataset with 5,000+ images captured or synthesized under conditions like:

- Motion Blur
- Overexposure / Sunny
- Fog
- Rain
- Low Light
- Glare / Uneven Lighting

### 🏷 Annotations

- *Gender:* Male / Female
- *Identity (ID):* Person ID for face recognition

### 📊 Splits

- Train: 70%
- Validation: 15%
- Test: 15% (hidden for final evaluation)

## 🎯 Objectives

- *Task A – Gender Classification*
  - Predict gender (Binary Classification)
  - Metrics: Accuracy, Precision, Recall, F1-score

- *Task B – Face Recognition*
  - Predict identity (Multi-class Classification)
  - Metrics: Top-1 Accuracy, Macro F1-score

*💡 Final Score = 30% (Task A) + 70% (Task B)*

## 🔧 Project Workflow
### 1️⃣ Dataset Setup

- Download dataset (provided separately or via link)
- Unzip and structure:
- facecom-challenge/
├── facecom_pipeline.ipynb         # Google Colab training notebook
├── gender_model.pkl               # Trained binary classifier
├── face_recognition_model.pkl     # Trained multi-class model
├── streamlit_app.py               # Streamlit app
├── requirements.txt               # Project dependencies
└── README.md                      # Documentation
