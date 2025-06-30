import os
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from utils import get_embedding, load_images_from_folder

def train_face_recognizer():
    data_dir = "FACECOM/Task_B/train"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found.")
        return

    persons = os.listdir(data_dir)
    if not persons:
        print("❌ No subfolders found in the training directory.")
        return

    X = []
    y = []
    total_images = 0

    print("📁 Scanning Task_B folders...")
    for person in persons:
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue
        images = load_images_from_folder(person_dir)
        print(f"📂 Found {len(images)} images for '{person}'")
        if not images:
            continue

        for img_path in images:
            embedding = get_embedding(img_path)
            if embedding is None:
                print(f"⚠ Skipping (embedding failed): {img_path}")
                continue
            X.append(embedding.flatten().astype(np.float32))
            y.append(person)
            total_images += 1

    print(f"\n📦 Total images processed: {total_images}")

    if total_images == 0:
        print("❌ No embeddings created. Cannot train model.")
        return

    X = np.array(X)
    y = np.array(y)

    print(f"📐 Embeddings shape: {X.shape}")

    # Apply PCA
    print("🔄 Applying PCA to reduce dimensions...")
    pca = PCA(n_components=256, whiten=True, random_state=42)
    X_reduced = pca.fit_transform(X)

    # Encode labels
    print("🧠 Encoding labels and training SVM classifier...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = SVC(kernel="linear", probability=True)
    model.fit(X_reduced, y_encoded)

    # Save models
    os.makedirs("models/saved_models", exist_ok=True)
    joblib.dump(pca, "models/saved_models/face_pca.pkl")
    joblib.dump(model, "models/saved_models/face_svm.pkl")
    joblib.dump(label_encoder, "models/saved_models/face_label_encoder.pkl")

    print("✅ Face recognizer trained and saved successfully!")

if __name__ == "__main__":
    train_face_recognizer()
