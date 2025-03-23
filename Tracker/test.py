import supervision as sv
from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO("yolov5s.pt")

# Perform object detection on an image (result is a list)
detection_results = model("photos/image.jpg")

# Get the first result from the list (assuming there's only one image in the list)
detection_result = detection_results[0]

# Convert the YOLO detection result to the Supervision format
detections = sv.Detections.from_ultralytics(detection_result)

# Visualize the detections on the image
print(detections)
# detections.plot("image_with_detections.jpg")
with open("file.txt", "w") as file:
    file.write(str(detections))
    
