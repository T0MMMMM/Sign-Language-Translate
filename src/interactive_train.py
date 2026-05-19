import cv2
import csv
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from hand_utils import create_hand_detector, extract_landmarks, draw_landmarks

DATA_PATH = "data/landmarks.csv"
MODEL_PATH = "models/asl_model.pkl"
WIN_W, WIN_H = 1080, 720

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)


def load_model():
    """Charge le modele s'il existe, sinon renvoie None."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


def train_model():
    """Reentraine le modele a partir du CSV. Renvoie (modele, message)."""
    if not os.path.exists(DATA_PATH):
        return None, "Aucune donnee a entrainer"

    df = pd.read_csv(DATA_PATH, header=None)
    if len(df) < 10:
        return None, f"Pas assez de donnees ({len(df)} lignes, min 10)"

    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]
    if y.nunique() < 2:
        return None, "Il faut au moins 2 lettres differentes"

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model, f"Modele entraine ({len(df)} exemples, {y.nunique()} lettres)"


def save_sample(label, features):
    """Ajoute une ligne au CSV : label + 63 coordonnees."""
    with open(DATA_PATH, "a", newline="") as f:
        csv.writer(f).writerow([label] + features.tolist())


# --- Initialisation ---
model = load_model()
detector = create_hand_detector()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIN_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)

cv2.namedWindow("ASL", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ASL", WIN_W, WIN_H)

status = "Appuie sur une lettre pour enregistrer, R pour entrainer, Q pour quitter"

print("=== Apprentissage interactif ASL ===")
print("- Fais un signe, l'IA predit")
print("- Touche lettre (A-Z) : enregistre le signe actuel comme cette lettre")
print("- Touche R : reentraine le modele")
print("- Touche Q : quitter\n")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (WIN_W, WIN_H))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.process(rgb)

    current_features = None
    prediction = "Aucune main"

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        draw_landmarks(frame, hand)
        current_features = extract_landmarks(hand)

        if model is not None:
            proba = model.predict_proba(current_features.reshape(1, -1))[0]
            best = proba.argmax()
            prediction = f"{model.classes_[best]}  ({proba[best]:.0%})"
        else:
            prediction = "Pas encore de modele - enregistre des exemples"

    # --- Affichage du texte sur l'image ---
    cv2.rectangle(frame, (0, 0), (WIN_W, 90), (0, 0, 0), -1)
    cv2.putText(frame, f"Prediction : {prediction}", (20, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

    cv2.rectangle(frame, (0, WIN_H - 50), (WIN_W, WIN_H), (0, 0, 0), -1)
    cv2.putText(frame, status, (20, WIN_H - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("ASL", frame)

    # --- Gestion des touches ---
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    elif key == ord("r"):
        model, msg = train_model()
        status = msg
        print(msg)

    # Touche lettre A-Z : enregistrer un exemple
    elif ord("a") <= key <= ord("z"):
        letter = chr(key).upper()
        if current_features is not None:
            save_sample(letter, current_features)
            status = f"Exemple enregistre pour '{letter}' - touche R pour reentrainer"
            print(status)
        else:
            status = "Aucune main detectee - impossible d'enregistrer"

cap.release()
cv2.destroyAllWindows()