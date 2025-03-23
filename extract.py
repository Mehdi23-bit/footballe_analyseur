import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread("aimage.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Harris Corner Detector
gray = np.float32(gray)  # Convert to float32
harris_corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# Threshold for marking corners (lower value = more corners)
threshold = 0.01 * harris_corners.max()

# Mark corners in red
image[harris_corners > threshold] = [0, 0, 255]

# Show the result
cv2.imshow("All Corners", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
