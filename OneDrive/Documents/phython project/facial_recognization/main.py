import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# ==========================
# Load Trained Model
# ==========================
model = load_model("model/emotion_model.keras")

# ==========================
# Emotion Labels
# ==========================
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise"
]

# ==========================
# Load Haar Cascade
# ==========================
face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_detector.empty():
    print("Error: haarcascade_frontalface_default.xml not found!")
    exit()

# ==========================
# Open Webcam
# ==========================
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Unable to open webcam")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX

start = time.time()
frame_count = 0

print("===================================")
print("Webcam Started Successfully...")
print("Press 'Q' to Quit")
print("===================================")

# ==========================
# Webcam Loop
# ==========================
while True:

    ret, frame = camera.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        # Face Crop
        face = gray[y:y+h, x:x+w]

        face = cv2.resize(face, (48, 48))

        face = face.astype("float32") / 255.0

        face = np.expand_dims(face, axis=-1)

        face = np.expand_dims(face, axis=0)

        # Prediction
        prediction = model.predict(face, verbose=0)

        emotion_index = np.argmax(prediction)

        emotion = emotion_labels[emotion_index]

        confidence = np.max(prediction) * 100

        # Face Box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Emotion Text
        cv2.putText(
            frame,
            f"{emotion} ({confidence:.1f}%)",
            (x, y - 10),
            font,
            0.7,
            (0, 255, 0),
            2
        )

    # FPS
    frame_count += 1

    elapsed = time.time() - start

    fps = frame_count / elapsed

    cv2.putText(
        frame,
        f"FPS : {fps:.1f}",
        (10, 30),
        font,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.imshow("Real Time Emotion Detection", frame)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

camera.release()

cv2.destroyAllWindows()