"""
config.py
---------
Central place for all settings, paths, colors and constants used
by the Facial Emotion Recognition app. Change values here instead
of digging through the main logic.
"""

import cv2

# ======================================
# PATHS
# ======================================

MODEL_PATH = "model/emotion_model.keras"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

# ======================================
# CAMERA
# ======================================

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ======================================
# MODEL
# ======================================

EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]

FACE_INPUT_SIZE = (48, 48)

# ======================================
# FACE DETECTION
# ======================================

SCALE_FACTOR = 1.3
MIN_NEIGHBORS = 5
MIN_FACE_SIZE = (30, 30)

# ======================================
# FONT
# ======================================

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ======================================
# THEME COLORS (BGR)
# ======================================

BG = (24, 24, 24)
PANEL = (34, 34, 34)
PANEL_BORDER = (55, 55, 55)
HEADER_BG = (18, 18, 18)
ACCENT = (255, 178, 60)          # soft blue-orange accent
WHITE = (240, 240, 240)
MUTED = (150, 150, 150)
GREEN = (110, 231, 130)
RED = (80, 80, 240)
BLUE = (255, 150, 90)
YELLOW = (90, 220, 240)
GRAY = (80, 80, 80)

# Per-emotion accent colors (BGR)
EMOTION_COLORS = {
    "Angry":    (70, 70, 235),
    "Disgust":  (80, 170, 90),
    "Fear":     (200, 120, 240),
    "Happy":    (100, 220, 100),
    "Neutral":  (210, 210, 210),
    "Sad":      (235, 150, 70),
    "Surprise": (60, 220, 240),
    "No Face":  (110, 110, 110),
}

# ======================================
# LAYOUT
# ======================================

HEADER_HEIGHT = 70
PANEL_X = 10
PANEL_Y = 90
PANEL_W = 300
PANEL_H = 400