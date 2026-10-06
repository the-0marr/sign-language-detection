# Sign Language Detection System

An end-to-end sign language recognition system using **MediaPipe Holistic**, **OpenCV**, and an **LSTM neural network**. The system converts webcam frames into landmark features, groups them into fixed-length sequences, classifies the sequence, and provides real-time predictions through a webcam interface.

## Overview

```text
Webcam
   ↓
MediaPipe Holistic
   ↓
1,662 landmark features / frame
   ↓
30-frame sequence
   ↓
LSTM classifier
   ↓
Sign prediction
```

The project also includes a graphical interface for managing action classes, recording new training examples, retraining the classifier, and starting real-time recognition.

## Features

- MediaPipe Holistic landmark extraction
- 1,662-feature frame representation
- 30-frame temporal sequences
- LSTM-based sequence classification
- Stratified train/validation/test splitting
- Class weighting for imbalanced datasets
- Early stopping and learning-rate scheduling
- Model checkpointing
- Accuracy, precision, recall, Macro F1, and confusion-matrix evaluation
- Confidence thresholding and temporal prediction smoothing
- Webcam-based real-time recognition
- GUI-based action management
- Add and record new actions without editing the source code
- Remove existing actions and their recorded data
- Retrain the model from the current action folders
- Optional pose-and-hands-only feature extraction for future experiments

## Repository Structure

```text
sign-language-detection/
├── Sign_Language_Detection.ipynb
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
└── models/
    └── README.md
```

`MP_Data/`, trained model files, label maps, logs, and generated artifacts are kept out of Git by default.

## Requirements

- Python 3
- A working webcam for data collection and real-time recognition
- A local desktop environment for the Tkinter GUI
- Internet access during the initial dependency installation

Google Colab can be used for notebook-based experimentation and training, but webcam/Tkinter behavior may differ from a local desktop environment.

## Installation

### Google Colab

Run this in a notebook cell:

```python
!pip install -U numpy pandas tensorflow scikit-learn opencv-python mediapipe matplotlib seaborn
```

### Jupyter Notebook / JupyterLab

Run this in a notebook cell:

```python
%pip install -U numpy pandas tensorflow scikit-learn opencv-python mediapipe matplotlib seaborn
```

### VS Code `.ipynb`

Run this in a notebook cell:

```python
%pip install -U numpy pandas tensorflow scikit-learn opencv-python mediapipe matplotlib seaborn
```

### VS Code Terminal

```bash
python -m pip install -U numpy pandas tensorflow scikit-learn opencv-python mediapipe matplotlib seaborn
```

Alternatively, from the repository root:

```bash
python -m pip install -r requirements.txt
```

After installing dependencies, restart the notebook kernel/runtime when required and run the notebook from the beginning.

## Dataset

The default dataset directory is:

```text
MP_Data/
```

Each action is stored as a separate folder. Each numbered sequence contains 30 `.npy` frame files, with 1,662 values per frame.

Example:

```text
MP_Data/
├── hello/
│   ├── 0/
│   │   ├── 0.npy
│   │   ├── 1.npy
│   │   └── ...
│   └── 1/
└── thank_you/
    └── ...
```

For reliable training, aim for **80–100 or more sequences per action** when possible. Variation in signer position, distance, lighting, background, signing speed, and hand position can improve generalization.

A `none` or idle class can also be useful when the system needs to distinguish signing from non-signing frames.

## Running the Project

Open `Sign_Language_Detection.ipynb` and run the cells from top to bottom.

The notebook provides these main stages:

1. Environment and configuration
2. MediaPipe landmark extraction
3. Dataset discovery and integrity checks
4. Train/validation/test split
5. LSTM model training
6. Test-set evaluation
7. Model and label-map saving
8. Real-time webcam recognition
9. Data collection
10. GUI-based action management and retraining

## GUI Workflow

Run:

```python
start_gui()
```

The GUI provides:

- **Add & Record** — create a new action and collect webcam sequences
- **Delete Selected** — remove an action and its recorded data
- **Retrain Model** — train a new classifier using the current dataset
- **Start Recognition** — launch real-time sign recognition
- **Refresh Actions** — reload the available action folders

After adding or deleting an action, retrain the model before recognition.

## Generated Files

Training creates:

```text
action_model.keras
labels.json
```

`labels.json` stores the action order and model configuration required for consistent inference.

## Evaluation

The notebook evaluates the final model on a held-out test set and reports:

- Test accuracy
- Macro F1
- Precision
- Recall
- Classification report
- Confusion matrix

For a stronger evaluation, test with signers and recording conditions that were not represented in the training data.

## Feature Representation

The default representation contains:

- Pose landmarks: 33 × 4
- Face landmarks: 468 × 3
- Left hand landmarks: 21 × 3
- Right hand landmarks: 21 × 3

Total: **1,662 values per frame**.

An optional pose-and-hands-only representation is included for experimentation. It contains **258 values per frame**, but it requires a separately collected dataset because the stored `.npy` files must match the feature representation used by the model.

## Webcam Controls

During real-time recognition:

- `Q` — quit the webcam window
- `C` — clear the displayed sentence

## Notes

The project is intended for research, learning, prototyping, and demonstration. Recognition quality depends on the number and quality of training sequences, class definitions, camera conditions, and signer variation.

## License

This repository is provided under the MIT License. See `LICENSE`.

## Quick Run

After a trained model has been created, recognition can be started directly without rerunning the notebook:

```bash
python run.py
```

For a different camera:

```bash
python run.py --camera 1
```

See [RUN.md](RUN.md) for the complete run workflow.
