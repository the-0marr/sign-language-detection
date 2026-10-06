"""Run the trained Sign Language Detection model without opening the notebook."""
from collections import Counter, deque
import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "action_model.keras"
LABELS_PATH = ROOT / "models" / "labels.json"
SEQUENCE_LENGTH = 30
FEATURES_PER_FRAME = 1662
DEFAULT_THRESHOLD = 0.70
SMOOTHING_WINDOW = 8
STABLE_VOTES = 6

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(results):
    pose = (np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark], dtype=np.float32).flatten()
            if results.pose_landmarks else np.zeros(33 * 4, dtype=np.float32))
    face = (np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark], dtype=np.float32).flatten()
            if results.face_landmarks else np.zeros(468 * 3, dtype=np.float32))
    left = (np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
            if results.left_hand_landmarks else np.zeros(21 * 3, dtype=np.float32))
    right = (np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
             if results.right_hand_landmarks else np.zeros(21 * 3, dtype=np.float32))
    return np.concatenate([pose, face, left, right]).astype(np.float32)


def draw_landmarks(image, results):
    specs = [
        (results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 1),
        (results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 2),
        (results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 2),
        (results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 2),
    ]
    for landmarks, connections, thickness in specs:
        mp_drawing.draw_landmarks(image, landmarks, connections,
                                  mp_drawing.DrawingSpec(thickness=thickness, circle_radius=2),
                                  mp_drawing.DrawingSpec(thickness=thickness, circle_radius=1))


def stable_prediction(history, probabilities, threshold):
    if not history:
        return None, 0.0
    label, votes = Counter(history).most_common(1)[0]
    confidence = float(probabilities[label])
    if votes / len(history) >= STABLE_VOTES / SMOOTHING_WINDOW and confidence >= threshold:
        return int(label), confidence
    return None, confidence


def load_model_and_labels():
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        raise FileNotFoundError(
            "No trained model was found. Train the model from the notebook first, "
            "then run this file again.\n\nExpected:\n"
            f"  {MODEL_PATH}\n  {LABELS_PATH}"
        )
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    actions = metadata["actions"]
    if metadata.get("features_per_frame", FEATURES_PER_FRAME) != FEATURES_PER_FRAME:
        raise ValueError("The saved model uses a different feature format than this runner.")
    return model, actions, float(metadata.get("confidence_threshold", DEFAULT_THRESHOLD))


def run(camera_index=0, threshold=None):
    model, actions, saved_threshold = load_model_and_labels()
    threshold = saved_threshold if threshold is None else threshold

    sequence = deque(maxlen=SEQUENCE_LENGTH)
    history = deque(maxlen=SMOOTHING_WINDOW)
    sentence = []
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}. Check camera permissions or try --camera 1.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    print("Sign Language Detection is running.")
    print("Press Q to quit | C to clear the sentence")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)
            rgb.flags.writeable = True
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            draw_landmarks(image, results)
            sequence.append(extract_keypoints(results))

            label, confidence = "Waiting...", 0.0
            if len(sequence) == SEQUENCE_LENGTH:
                probabilities = model.predict(np.expand_dims(np.asarray(sequence), 0), verbose=0)[0]
                idx = int(np.argmax(probabilities))
                confidence = float(probabilities[idx])
                if confidence >= threshold:
                    history.append(idx)
                stable_idx, stable_conf = stable_prediction(history, probabilities, threshold)
                if stable_idx is not None:
                    label = actions[stable_idx]
                    if not sentence or sentence[-1] != label:
                        sentence.append(label)
                    sentence = sentence[-6:]
                else:
                    label = actions[idx]

            cv2.rectangle(image, (0, 0), (960, 90), (30, 30, 30), -1)
            cv2.putText(image, f"Sign: {label}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
            cv2.putText(image, f"Confidence: {confidence:.2f}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            cv2.rectangle(image, (0, 450), (960, 540), (30, 30, 30), -1)
            cv2.putText(image, "Sentence: " + " ".join(sentence), (15, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)
            cv2.imshow("Sign Language Detection", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                sentence.clear()
                history.clear()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the trained Sign Language Detection model.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--threshold", type=float, default=None, help="Override confidence threshold")
    args = parser.parse_args()
    run(args.camera, args.threshold)
