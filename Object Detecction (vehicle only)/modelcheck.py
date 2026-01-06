from ultralytics import YOLO
import cv2

# Pretrained YOLOv8 nano on COCO
vehicle_model = YOLO("yolov8n.pt")

# COCO class names
COCO_CLASSES = [
     "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier"
]

cap = cv2.VideoCapture("56310-479197605_small.mp4")  # video path

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = vehicle_model(frame, conf=0.5)

    # Loop through detected boxes
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in [2, 3, 5, 7]:  # Only vehicle classes
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                label = COCO_CLASSES[cls_id]

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label + confidence
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Vehicle Detection", frame)
    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
