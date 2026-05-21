import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def create_hand_detector():
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )


def extract_landmarks(hand_landmarks):
    """Transforme les 21 points MediaPipe en 63 nombres normalisés."""
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
    )
    coords -= coords[0]
    max_dist = np.max(np.linalg.norm(coords, axis=1))
    if max_dist > 0:
        coords /= max_dist
    return coords.flatten()


def draw_landmarks(image, hand_landmarks):
    mp_draw.draw_landmarks(
        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
    )
