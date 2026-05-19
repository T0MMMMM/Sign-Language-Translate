import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Noms des 21 points pour un affichage lisible dans le terminal
LANDMARK_NAMES = [
    "POIGNET",
    "POUCE_base", "POUCE_1", "POUCE_2", "POUCE_bout",
    "INDEX_base", "INDEX_1", "INDEX_2", "INDEX_bout",
    "MAJEUR_base", "MAJEUR_1", "MAJEUR_2", "MAJEUR_bout",
    "ANNULAIRE_base", "ANNULAIRE_1", "ANNULAIRE_2", "ANNULAIRE_bout",
    "AURICULAIRE_base", "AURICULAIRE_1", "AURICULAIRE_2", "AURICULAIRE_bout",
]

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

cap = cv2.VideoCapture(0)
print("Fenetre ouverte. Appuie sur 'q' pour quitter.\n")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Dessiner les points et les connexions sur l'image
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Savoir si c'est la main gauche ou droite
            label = results.multi_handedness[i].classification[0].label

            # Afficher les positions dans le terminal
            print(f"--- Main {label} ---")
            h, w, _ = frame.shape
            for idx, lm in enumerate(hand_landmarks.landmark):
                # lm.x et lm.y sont entre 0 et 1 -> on convertit en pixels
                px, py = int(lm.x * w), int(lm.y * h)
                print(f"  {LANDMARK_NAMES[idx]:<18} "
                      f"x={px:4d}  y={py:4d}  z={lm.z:+.3f}")
            print()

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()