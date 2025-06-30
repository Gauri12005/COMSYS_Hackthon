import streamlit as st
import pickle
import cv2
import numpy as np
from PIL import Image
from utils import get_embedding

# Load models
with open("models/face_svm.pkl", "rb") as f:
    face_model = pickle.load(f)
with open("models/face_label_encoder.pkl", "rb") as f:
    face_le = pickle.load(f)
with open("models/gender_svm.pkl", "rb") as f:
    gender_model = pickle.load(f)
with open("models/gender_label_encoder.pkl", "rb") as f:
    gender_le = pickle.load(f)

st.title("🧠 Face Recognition + Gender Classification")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    embedding = get_embedding(img_np)

    if embedding is None:
        st.error("❌ Could not extract features from image.")
    else:
        face_pred = face_model.predict([embedding])[0]
        name = face_le.inverse_transform([face_pred])[0]

        gender_pred = gender_model.predict([embedding])[0]
        gender = gender_le.inverse_transform([gender_pred])[0]

        st.image(image, caption=f"Prediction: {name} ({gender})", width=300)