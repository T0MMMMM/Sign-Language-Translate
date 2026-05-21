import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import csv
import pickle
import numpy as np
import pandas as pd
import kagglehub
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from hand_utils import extract_landmarks

mp_hands = mp.solutions.hands

DATA_PATH = "data/landmarks.csv"
MODEL_PATH = "models/asl_model.pkl"
SAMPLES_PER_LETTER = 300


def download_dataset():
    print("Téléchargement du dataset ASL Kaggle...")
    path = kagglehub.dataset_download("grassknoted/asl-alphabet")
    print(f"Dataset disponible : {path}")
    return path


def extract_from_dataset(dataset_path):
    # static_image_mode=True est indispensable pour les images fixes du dataset
    detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    )
    os.makedirs("data", exist_ok=True)

    # Le dataset a la structure : asl_alphabet_train/asl_alphabet_train/<LETTRE>/img.jpg
    train_root = os.path.join(dataset_path, "asl_alphabet_train", "asl_alphabet_train")
    if not os.path.isdir(train_root):
        train_root = dataset_path
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"Dataset structure not found at: {dataset_path}")

    rows = []
    letters = sorted([d for d in os.listdir(train_root)
                      if os.path.isdir(os.path.join(train_root, d))
                      and len(d) == 1 and d.isalpha()])

    print(f"Lettres trouvées : {letters}")

    for letter in letters:
        letter_dir = os.path.join(train_root, letter)
        images = [f for f in os.listdir(letter_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:SAMPLES_PER_LETTER]
        count = 0
        for img_name in images:
            img = cv2.imread(os.path.join(letter_dir, img_name))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)
            if result.multi_hand_landmarks:
                landmarks = extract_landmarks(result.multi_hand_landmarks[0])
                rows.append([letter.upper()] + landmarks.tolist())
                count += 1
        print(f"  {letter.upper()} : {count} exemples extraits")

    with open(DATA_PATH, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"\nTotal : {len(rows)} exemples sauvegardés dans {DATA_PATH}")
    return rows


def train_model():
    df = pd.read_csv(DATA_PATH, header=None)
    X = df.iloc[:, 1:].values.astype(float)
    y = df.iloc[:, 0].values

    print(f"Entraînement sur {len(X)} exemples, {len(set(y))} lettres...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Modèle sauvegardé : {MODEL_PATH}")
    print(f"Classes : {model.classes_}")
    return model


if __name__ == "__main__":
    dataset_path = download_dataset()
    extract_from_dataset(dataset_path)
    train_model()
    print("\nEntraînement terminé. Lance export_model.py pour générer le fichier ONNX.")
