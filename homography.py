import cv2
import numpy as np

# Load the image
image = cv2.imread("photos\download.jpeg")

# Define four points in the original image (clockwise from top-left)
pts_src = np.array([[100, 200], [400, 150], [450, 450], [120, 500]], dtype=np.float32)

# Define corresponding points in the output (desired rectangular shape)
width, height = 300, 400  # Dimensions of the output rectangle
pts_dst = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

# Compute homography matrix
H, _ = cv2.findHomography(pts_src, pts_dst)

# Apply the perspective warp
warped_image = cv2.warpPerspective(image, H, (width, height))

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Warped Image", warped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
