# Real-Time Facial Emotion Recognition

A real-time webcam application that detects human faces and classifies their
emotional expression using a CNN trained with TensorFlow, with face detection
powered by OpenCV's Haar Cascade classifier.

## Features

- Real-time webcam emotion detection
- CNN model trained using TensorFlow
- Face detection using OpenCV Haar Cascade
- Clean, modular codebase (`config.py`, `utils.py`, `main.py`)
- Custom UI overlay — status panel, live FPS/clock, glassy rounded panels,
  color-coded corner-bracket face boxes with confidence tags
- Predicts:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Neutral
  - Sad
  - Surprise

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- OpenCV
- NumPy

## Project Structure

```
facial_recognization/
│
├── main.py                 # Entry point — camera loop, prediction, calls UI
├── utils.py                 # All drawing/UI helper functions
├── config.py                 # Colors, paths, labels, layout constants
├── requirements.txt
├── haarcascade_frontalface_default.xml   # Face detector (download separately)
└── model/
    └── emotion_model.keras   # Trained CNN model (you provide/train this)
```

## Setup

### 1. Create a virtual environment (Python 3.11 recommended)

TensorFlow does not yet fully support the newest Python releases, so 3.11 is
the safest choice.

```bash
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Get the Haar Cascade file

Download `haarcascade_frontalface_default.xml` and place it in the project
root:

https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml

### 4. Add your trained model

Place your trained Keras model at `model/emotion_model.keras`. The model
must accept a `(1, 48, 48, 1)` grayscale input and output a 7-class softmax
matching the label order in `config.py`.

### 5. Run

```bash
python main.py
```

Press `q` in the window to quit.

## Configuration

All tunable values live in `config.py` — camera index, frame size, Haar
Cascade detection sensitivity, emotion colors, and UI layout. Change values
there instead of editing the main loop.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Import "cv2" could not be resolved` in VS Code | Select the venv interpreter: `Ctrl+Shift+P` → *Python: Select Interpreter* → choose `venv\Scripts\python.exe` |
| `No matching distribution found for tensorflow` | Your Python version is too new/old for TensorFlow. Use Python 3.11 via `py -3.11 -m venv venv` |
| `Cascade File Missing` on run | Download the Haar Cascade XML (see Setup step 3) into the project root |
| `Camera Error` on run | Check `CAMERA_INDEX` in `config.py`, or another app is using the webcam |
| Low FPS | Lower `FRAME_WIDTH`/`FRAME_HEIGHT` in `config.py`, or reduce `scaleFactor` sensitivity |

## Roadmap Ideas

- Multi-face emotion history graph over time
- Export session logs (CSV) of detected emotions with timestamps
- Swap Haar Cascade for a DNN-based face detector for better accuracy
- Package as a standalone `.exe` with PyInstaller
