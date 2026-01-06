from ultralytics import YOLO

model = YOLO("best.pt")

# Predict on video
results = model("56310-479197605_small.mp4", conf=0.65, show=True, save=True)
