# Running the Trained Model

After the model has been trained once, you do not need to execute the notebook from the beginning for normal recognition.

## 1. Install dependencies

Use the installation command in the README or `requirements.txt`.

## 2. Train once

Open `Sign_Language_Detection.ipynb` and complete the data/training workflow. This creates:

```text
models/action_model.keras
models/labels.json
```

## 3. Run recognition directly

From the project root:

```bash
python run.py
```

The webcam window will open. Use:

- `Q` — quit
- `C` — clear the current sentence

If the default camera is unavailable:

```bash
python run.py --camera 1
```

To override the confidence threshold:

```bash
python run.py --threshold 0.80
```

## Retraining

When actions are added, removed, or new training data is collected, use the notebook GUI to retrain the model. After retraining, `python run.py` can again be used for recognition without rerunning the notebook.
