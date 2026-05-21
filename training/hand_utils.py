import numpy as np


def _get_mp():
    import mediapipe as mp
    return mp.solutions.hands, mp.solutions.drawing_utils


def create_hand_detector():
    mp_hands, _ = _get_mp()
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
    import cv2  # noqa: F401
    mp_hands, mp_draw = _get_mp()
    mp_draw.draw_landmarks(
        image, hand_landmarks, mp_hands.HAND_CONNECTIONS
    )
