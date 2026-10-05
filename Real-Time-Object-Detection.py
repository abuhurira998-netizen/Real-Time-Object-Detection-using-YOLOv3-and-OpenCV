import os
import urllib.request
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
weights_path = BASE_DIR / "yolov3.weights"
config_path = BASE_DIR / "yolov3.cfg"
names_path = BASE_DIR / "coco.names"


def ensure_model_files():
    if not config_path.exists():
        raise FileNotFoundError(
            f"YOLO config file not found: {config_path}. "
            "Download yolov3.cfg and place it in the project root."
        )

    if not names_path.exists():
        raise FileNotFoundError(
            f"COCO class names file not found: {names_path}. "
            "Download coco.names and place it in the project root."
        )

    if not weights_path.exists():
        weights_url = "https://pjreddie.com/media/files/yolov3.weights"
        print(f"Downloading YOLOv3 weights to {weights_path}...")
        urllib.request.urlretrieve(weights_url, weights_path)


ensure_model_files()

# Load class names
with open(names_path, "r", encoding="utf-8") as f:
    class_names = [line.strip() for line in f.readlines() if line.strip()]

# Load YOLOv3 model
net = cv2.dnn_DetectionModel(str(weights_path), str(config_path))
net.setInputSize(608, 608)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)

# Start webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Object detection
        classes, scores, boxes = net.detect(
            frame,
            confThreshold=0.5,
            nmsThreshold=0.4,
        )

        # Draw detections
        if len(classes) > 0:
            for class_id, score, box in zip(
                classes.flatten(),
                scores.flatten(),
                boxes,
            ):
                class_index = int(class_id)
                if 0 <= class_index < len(class_names):
                    label = class_names[class_index]
                else:
                    label = "unknown"

                confidence = float(score)
                x, y, w, h = box.astype(int)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (255, 0, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("Real-Time Object Detection", frame)

        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
