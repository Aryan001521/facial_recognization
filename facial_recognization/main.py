"""
main.py
-------
Entry point for the AI Facial Emotion Recognition app.
Run this file: python main.py
"""

import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model

import config as cfg
import utils


def main():

    # ======================================
    # LOAD MODEL + FACE DETECTOR
    # ======================================

    print("Loading model...")
    model = load_model(cfg.MODEL_PATH)

    face_detector = cv2.CascadeClassifier(cfg.CASCADE_PATH)

    if face_detector.empty():
        print("Error: Cascade file missing or invalid path.")
        return

    # ======================================
    # CAMERA
    # ======================================

    camera = cv2.VideoCapture(cfg.CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.FRAME_HEIGHT)

    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    window_name = "AI Facial Emotion Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    start_time = time.time()
    frame_count = 0

    print("Running... press 'q' to quit.")

    # ======================================
    # MAIN LOOP
    # ======================================

    while True:

        ret, frame = camera.read()
        if not ret:
            print("Camera read failed.")
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=cfg.SCALE_FACTOR,
            minNeighbors=cfg.MIN_NEIGHBORS,
            minSize=cfg.MIN_FACE_SIZE,
        )

        current_emotion = "No Face"
        confidence = 0.0

        for (x, y, w, h) in faces:

            face_input = utils.preprocess_face(gray, x, y, w, h)

            prediction = model.predict(face_input, verbose=0)
            emotion_index = int(np.argmax(prediction))

            current_emotion = cfg.EMOTION_LABELS[emotion_index]
            confidence = float(np.max(prediction) * 100)

            color = cfg.EMOTION_COLORS.get(current_emotion, cfg.YELLOW)

            utils.draw_face_box(frame, x, y, w, h, color, current_emotion, confidence)

        # UI overlay drawn last so it always sits on top
        utils.draw_header(frame, frame_w, fps)
        utils.draw_left_panel(frame, current_emotion, confidence, fps, len(faces))

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()