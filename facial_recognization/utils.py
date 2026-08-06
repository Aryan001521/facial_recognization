"""
utils.py
--------
All drawing / UI helper functions used by main.py.
Keeping these separate keeps the main loop short and readable.
"""

import cv2
import numpy as np
from datetime import datetime

import config as cfg


# ======================================
# ROUNDED PANEL BACKGROUND
# ======================================

def draw_panel(img, x, y, w, h, radius=14, color=cfg.PANEL,
               border_color=cfg.PANEL_BORDER, alpha=0.85):
    """Draws a semi-transparent rounded rectangle panel."""

    overlay = img.copy()

    cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1)
    cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)

    for cx, cy in [
        (x + radius, y + radius),
        (x + w - radius, y + radius),
        (x + radius, y + h - radius),
        (x + w - radius, y + h - radius),
    ]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 1)


# ======================================
# PROGRESS BAR
# ======================================

def progress_bar(img, x, y, w, h, percent, color=cfg.GREEN):
    """Draws a rounded confidence bar filled according to percent (0-100)."""

    percent = max(0, min(100, percent))

    cv2.rectangle(img, (x, y), (x + w, y + h), cfg.GRAY, 1)

    fill = int((percent / 100) * w)
    if fill > 0:
        cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)


# ======================================
# HEADER BAR
# ======================================

def draw_header(img, frame_w, fps):
    """Top header bar with app title and live FPS/time readout."""

    cv2.rectangle(img, (0, 0), (frame_w, cfg.HEADER_HEIGHT), cfg.HEADER_BG, -1)
    cv2.line(img, (0, cfg.HEADER_HEIGHT), (frame_w, cfg.HEADER_HEIGHT), cfg.ACCENT, 2)

    cv2.putText(
        img, "AI Facial Emotion Recognition",
        (20, 45), cfg.FONT, 1.0, cfg.WHITE, 2, cv2.LINE_AA
    )

    right_text = datetime.now().strftime("%H:%M:%S")
    (tw, _), _ = cv2.getTextSize(right_text, cfg.FONT, 0.7, 2)
    cv2.putText(
        img, right_text,
        (frame_w - tw - 20, 45), cfg.FONT, 0.7, cfg.MUTED, 2, cv2.LINE_AA
    )


# ======================================
# LEFT INFO PANEL
# ======================================

def draw_left_panel(img, emotion, confidence, fps, faces):
    """Left side status panel: camera status, face count, emotion, confidence, fps."""

    x, y, w, h = cfg.PANEL_X, cfg.PANEL_Y, cfg.PANEL_W, cfg.PANEL_H
    draw_panel(img, x, y, w, h)

    pad = 24
    row = y + 40

    cv2.putText(img, "STATUS", (x + pad, row), cfg.FONT, 0.75, cfg.WHITE, 2, cv2.LINE_AA)
    row += 40

    cv2.circle(img, (x + pad + 6, row - 6), 5, cfg.GREEN, -1)
    cv2.putText(img, "Camera Connected", (x + pad + 22, row), cfg.FONT, 0.6, cfg.GREEN, 1, cv2.LINE_AA)
    row += 40

    cv2.putText(img, f"Faces Detected : {faces}", (x + pad, row), cfg.FONT, 0.6, cfg.WHITE, 1, cv2.LINE_AA)
    row += 45

    emotion_color = cfg.EMOTION_COLORS.get(emotion, cfg.YELLOW)
    cv2.putText(img, "Emotion", (x + pad, row), cfg.FONT, 0.6, cfg.MUTED, 1, cv2.LINE_AA)
    row += 32
    cv2.putText(img, emotion, (x + pad, row), cfg.FONT, 0.9, emotion_color, 2, cv2.LINE_AA)
    row += 40

    cv2.putText(img, "Confidence", (x + pad, row), cfg.FONT, 0.6, cfg.MUTED, 1, cv2.LINE_AA)
    row += 15
    progress_bar(img, x + pad, row, w - pad * 2, 18, confidence, color=emotion_color)
    row += 40
    cv2.putText(img, f"{confidence:.1f} %", (x + pad, row), cfg.FONT, 0.6, cfg.WHITE, 1, cv2.LINE_AA)
    row += 45

    cv2.putText(img, f"FPS : {fps:.1f}", (x + pad, row), cfg.FONT, 0.6, cfg.BLUE, 1, cv2.LINE_AA)


# ======================================
# MODERN CORNER-STYLE FACE BOX
# ======================================

def draw_face_box(img, x, y, w, h, color, label, confidence):
    """Draws corner-bracket style bounding box with a label tag above it."""

    l = 22
    t = 2

    corners = [
        ((x, y), (x + l, y), (x, y + l)),
        ((x + w, y), (x + w - l, y), (x + w, y + l)),
        ((x, y + h), (x + l, y + h), (x, y + h - l)),
        ((x + w, y + h), (x + w - l, y + h), (x + w, y + h - l)),
    ]

    for corner, h_end, v_end in corners:
        cv2.line(img, corner, h_end, color, t + 2, cv2.LINE_AA)
        cv2.line(img, corner, v_end, color, t + 2, cv2.LINE_AA)

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 1, cv2.LINE_AA)

    label_text = f"{label}  {confidence:.0f}%"
    (tw, th), _ = cv2.getTextSize(label_text, cfg.FONT, 0.6, 2)

    tag_y1 = max(0, y - th - 18)
    tag_y2 = max(0, y - 4)

    overlay = img.copy()
    cv2.rectangle(overlay, (x, tag_y1), (x + tw + 16, tag_y2), color, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    cv2.putText(
        img, label_text, (x + 8, tag_y2 - 8),
        cfg.FONT, 0.6, (0, 0, 0), 2, cv2.LINE_AA
    )


# ======================================
# PREPROCESS FACE FOR MODEL
# ======================================

def preprocess_face(gray_frame, x, y, w, h):
    """Crops, resizes and normalizes a face region for model prediction."""

    face = gray_frame[y:y + h, x:x + w]
    face = cv2.resize(face, cfg.FACE_INPUT_SIZE)
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, -1)
    face = np.expand_dims(face, 0)
    return face