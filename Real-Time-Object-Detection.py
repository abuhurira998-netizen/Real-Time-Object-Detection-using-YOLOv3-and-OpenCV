import cv2

# Model paths
weights_path = "models/yolov3.weights"
config_path = "models/yolov3.cfg"
names_path = "data/coco.names"

# Load class names
with open(names_path, "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Load YOLOv3 model
net = cv2.dnn_DetectionModel(weights_path, config_path)

net.setInputSize(608, 608)
net.setInputScale(1.0 / 255)
net.setInputSwapRB(True)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Object detection
    classes, scores, boxes = net.detect(
        frame,
        confThreshold=0.5,
        nmsThreshold=0.4
    )

    # Draw detections
    if len(classes) > 0:
        for class_id, score, box in zip(
            classes.flatten(),
            scores.flatten(),
            boxes
        ):
            label = class_names[class_id - 1]
            confidence = float(score)

            x, y, w, h = box

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

    cv2.imshow("Real-Time Object Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
