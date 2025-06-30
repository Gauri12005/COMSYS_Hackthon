import os
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from utils import FaceNetEmbedder

def train_gender_classifier():
    embedder = FaceNetEmbedder()

    X = []
    y = []

    task_a_path = "FACECOM/Task_A/train"
    total_images = 0

    for gender in ["male", "female"]:
        folder_path = os.path.join(task_a_path, gender)
        if not os.path.exists(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            continue

        print(f"🔍 Scanning '{gender}' folder...")
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"⚠️ Skipping invalid image: {img_path}")
                continue

            # Get embedding
            embedding = embedder.get_embedding(img)
            X.append(embedding.flatten())  # ✅ Flatten to 1D vector
            y.append(gender)
            total_images += 1

    if len(X) == 0:
        print("❌ No valid images found. Aborting.")
        return

    print(f"\n📦 Total images processed: {total_images}")
    print("🧠 Encoding labels and training model...")

    X = np.array(X)
    y = np.array(y)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train SVM
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y_encoded)

    # Save model and label encoder
    os.makedirs("models/saved_models", exist_ok=True)
    joblib.dump(model, "models/saved_models/gender_svm.pkl")
    joblib.dump(label_encoder, "models/saved_models/gender_label_encoder.pkl")

    print("✅ Gender classifier trained and saved to 'models/saved_models/'")

if __name__ == "__main__":
    train_gender_classifier()