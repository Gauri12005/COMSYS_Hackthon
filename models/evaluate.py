import os
import cv2
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import FaceNetEmbedder

# -------------------------------
# ✳️ PATHS
task_a_dir = "FACECOM/Task_A_Test"
task_b_dir = "FACECOM/Task_B/test"

# -------------------------------
# ✳️ Load models
with open("saved_models/gender_classifier.pkl", "rb") as f:
    gender_model, gender_le = pickle.load(f)

with open("saved_models/face_recognizer.pkl", "rb") as f:
    face_model, face_le = pickle.load(f)

embedder = FaceNetEmbedder()

# -------------------------------
# ✅ TASK A: Gender Classification
print("\n🔎 Evaluating Gender Classification (Task A)...")

X_a, y_true_a, y_pred_a = [], [], []

for gender in os.listdir(task_a_dir):
    folder = os.path.join(task_a_dir, gender)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        if img is None: continue

        emb = embedder.get_embedding(img)
        pred = gender_model.predict([emb])[0]

        X_a.append(emb)
        y_true_a.append(gender)
        y_pred_a.append(gender_le.inverse_transform([pred])[0])

# ✅ Compute metrics
print("Accuracy:", accuracy_score(y_true_a, y_pred_a))
print("Precision:", precision_score(y_true_a, y_pred_a, pos_label='male', average='binary'))
print("Recall:", recall_score(y_true_a, y_pred_a, pos_label='male', average='binary'))
print("F1-Score:", f1_score(y_true_a, y_pred_a, pos_label='male', average='binary'))

# -------------------------------
# ✅ TASK B: Face Recognition
print("\n🔎 Evaluating Face Recognition (Task B)...")

X_b, y_true_b, y_pred_b = [], [], []

for person in os.listdir(task_b_dir):
    folder = os.path.join(task_b_dir, person)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        if img is None: continue

        emb = embedder.get_embedding(img)
        pred = face_model.predict([emb])[0]

        X_b.append(emb)
        y_true_b.append(person)
        y_pred_b.append(face_le.inverse_transform([pred])[0])

# ✅ Compute metrics
print("Top-1 Accuracy:", accuracy_score(y_true_b, y_pred_b))
print("Macro-averaged F1:", f1_score(y_true_b, y_pred_b, average='macro'))
