import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from utils import get_embedding
import cv2

# TASK B - Face Recognition
def train_face_recognizer():
    X, y = [], []
    base_path = "FACECOM/Task_B/train"
    for person in os.listdir(base_path):
        person_dir = os.path.join(base_path, person)
        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠ Skipping unreadable: {img_path}")
                continue
            emb = get_embedding(image)
            if emb is not None:
                X.append(emb)
                y.append(person)

    X = np.array(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, y_enc)

    with open("models/saved_models/face_svm.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("models/saved_models/face_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

# TASK A - Gender Classification
def train_gender_classifier():
    X, y = [], []
    base_path = "FACECOM/Task_A/train"
    for gender in os.listdir(base_path):
        gender_dir = os.path.join(base_path, gender)
        for img_name in os.listdir(gender_dir):
            img_path = os.path.join(gender_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠ Skipping unreadable: {img_path}")
                continue
            emb = get_embedding(image)
            if emb is not None:
                X.append(emb)
                y.append(gender)

    X = np.array(X)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X, y_enc)

    with open("models/saved_models/gender_svm.pkl", "wb") as f:
        pickle.dump(clf, f)
    with open("models/saved_models/gender_label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

if __name__ == "_main_":
    train_face_recognizer()
    train_gender_classifier()
    print("✅ Models trained and saved.")