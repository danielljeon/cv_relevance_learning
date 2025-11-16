import os
import time
from collections import deque

import cv2
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from notify import call_notify, init_notify
from yolo_wrapper import YOLO

# Constants.
BUFFER_SECONDS = 2.0
FPS = 10
BUFFER_SIZE = int(BUFFER_SECONDS * FPS)

POSITIVE_LABEL = 0
NEGATIVE_LABEL = 1

# Path for saving training data.
MODEL_PATH = "relevance_training_data.npz"

# Memory.
buffer = deque(maxlen=BUFFER_SIZE)  # (timestamp, feature_vector).

X_train = []
y_train = []

nb_model = None
knn_model = None

LAST_NOTIFIED_FEATS = []


def extract_features(det):
    """Simple 3-feature representation:

    Returns:
        [class_id, bbox_area, center_y]
    """
    class_id = det.class_id
    area = det.w * det.h
    center_y = det.y + det.h / 2.0
    return np.array([class_id, area, center_y], dtype=float)


def update_models(n_new: int | None = None):
    """Update online relevance prediction model.

    If n_new is provided and a Naive Bayes model already exists, only the last
    n_new samples are used to incrementally update NB via partial_fit. kNN is
    still re-fit on the full dataset.
    """
    global nb_model, knn_model
    if len(X_train) < 5:
        return

    X = np.array(X_train)
    y = np.array(y_train)

    # Naive Bayes: online / incremental.
    if nb_model is None or n_new is None or n_new <= 0:
        # Cold start or full rebuild: fit on all data.
        nb_model_local = GaussianNB()
        nb_model_local.partial_fit(
            X, y, classes=np.array([POSITIVE_LABEL, NEGATIVE_LABEL])
        )
        nb_model = nb_model_local
    else:
        # Incremental update on just the new samples.
        X_new = X[-n_new:]
        y_new = y[-n_new:]
        nb_model.partial_fit(X_new, y_new)

    # kNN: still trained on full dataset (lazy, but dataset is small).
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    knn_model = knn

    print(f"[INFO] Models updated. Samples: {len(X_train)}")


def save_training_data():
    """Save X_train and y_train to disk."""
    if not X_train:
        print("[SAVE] No training data to save.")
        return

    X = np.array(X_train)
    y = np.array(y_train)
    np.savez(MODEL_PATH, X=X, y=y)
    print(f"[SAVE] Training data saved to {MODEL_PATH}")


def load_training_data():
    """Load X_train and y_train from disk if available, then rebuild models."""
    global X_train, y_train
    if not os.path.exists(MODEL_PATH):
        print("[LOAD] No previous training data found. Starting fresh.")
        return

    try:
        data = np.load(MODEL_PATH, allow_pickle=True)
        X = data["X"]
        y = data["y"]
        X_train = [x for x in X]
        y_train = [int(label) for label in y]
        print(
            f"[LOAD] Loaded model from {MODEL_PATH} (samples: {len(X_train)})"
        )
        update_models()
    except Exception as e:
        print(f"[LOAD] Failed to load training data: {e}")


def handle_positive_press():
    """Add positive samples to training set."""
    added = 0
    for _, feat in buffer:
        X_train.append(feat)
        y_train.append(POSITIVE_LABEL)
        added += 1

    # Pass number of new samples for online NB update.
    update_models(added)

    print(f"[TRAIN] + Positive: added {added} samples")


def handle_negative_press():
    """Add negative samples to training set."""
    global LAST_NOTIFIED_FEATS
    if not LAST_NOTIFIED_FEATS:
        print("[WARN] No notifications to tag as negative.")
        return

    added = 0
    for feat in LAST_NOTIFIED_FEATS:
        X_train.append(feat)
        y_train.append(NEGATIVE_LABEL)
        added += 1

    # Pass number of new samples for online NB update.
    update_models(added)
    LAST_NOTIFIED_FEATS = []

    print(f"[TRAIN] - Negative: added {added} samples")


def predict_notifications(detections):
    """Only triggers for POSITIVE_LABEL (0).

    Returns:
        List of (score, det, feature)
    """
    results = []
    if nb_model is None or knn_model is None:
        return results

    for det in detections:
        feat = extract_features(det).reshape(1, -1)

        nb_probs = nb_model.predict_proba(feat)[0]
        knn_probs = knn_model.predict_proba(feat)[0]

        avg_probs = (nb_probs + knn_probs) / 2.0
        pos_score = float(avg_probs[POSITIVE_LABEL])

        if pos_score > 0.6:
            results.append((pos_score, det, feat.squeeze()))

    return results


def main():
    global LAST_NOTIFIED_FEATS

    # Initialize notification service/properties.
    init_notify()

    # Try to load previous training data and rebuild models.
    load_training_data()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    yolo = YOLO("yolov5n.onnx")

    print("System started.")
    print("Press '1' to train *positive* examples.")
    print("Press 'a' to train *negative* examples.")
    print("Press 's' to save training data.")
    print("Press 'q' to quit (auto-save).")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.1)
                continue

            detections = yolo.detect(frame)
            now = time.time()

            # Keep only most salient detection in buffer.
            if detections:
                det = max(detections, key=lambda d: d.w * d.h)
                feat = extract_features(det)
                buffer.append((now, feat))

            # Draw detections.
            for det in detections:
                h, w, _ = frame.shape
                x1 = int((det.x - det.w / 2) * w)
                y1 = int((det.y - det.h / 2) * h)
                x2 = int((det.x + det.w / 2) * w)
                y2 = int((det.y + det.h / 2) * h)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"id:{det.class_id}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            cv2.imshow("Vision Relevance Trainer", frame)

            # Key handling.
            key = cv2.waitKey(1) & 0xFF

            if key == ord("1"):
                handle_positive_press()

            elif key == ord("a"):
                handle_negative_press()

            elif key == ord("s"):
                save_training_data()

            elif key == ord("q"):
                print("Quit. Saving training data...")
                save_training_data()
                break

            # Notification.
            notify_items = predict_notifications(detections)
            LAST_NOTIFIED_FEATS = []

            for score, det, feat in notify_items:
                LAST_NOTIFIED_FEATS.append(feat)

                # Pass score and rough location into notification.
                call_notify(score, det.x)

            time.sleep(max(0.0, 1.0 / FPS - 0.001))

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
