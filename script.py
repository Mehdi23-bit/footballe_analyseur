import torch
from ultralytics import YOLO

model = YOLO("models/best.pt")
results = model.predict('videos/test14.mp4')

# Save detection results to a text file
with open("detection_results.txt", "w") as f:
    for result in results:
        for box in result.boxes:
            f.write(f"Class: {int(box.cls)}, Confidence: {box.conf:.2f}, BBox: {box.xyxy.tolist()}\n")
