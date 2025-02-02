# import cv2
# import numpy as np

# for i in range(1,21):
#     print(i)
#     if i in {8,15}:
#         pass
#     else:
        
#         image = cv2.imread(f"photos/myimage{i}.jpg")

#         # Convert to HSV color space for better color segmentation
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#         # Define a range for the background color (e.g., green)
#         lower_green = np.array([35, 50, 50])  # Lower bound of green in HSV
#         upper_green = np.array([85, 255, 255])  # Upper bound of green in HSV

#         # Create a mask for the background
#         mask = cv2.inRange(hsv, lower_green, upper_green)

#         # Invert the mask to get the foreground
#         foreground_mask = cv2.bitwise_not(mask)

#         # Extract the foreground
#         foreground = cv2.bitwise_and(image, image, mask=foreground_mask)

#         # Save or display the result
#         cv2.imwrite(f"result/foreground{i}.jpg", foreground)
# cv2.imshow("Foreground", foreground)
import cv2
import numpy as np
from sklearn.cluster import KMeans

def get_dominant_color(image_path, k=1):
    # Load image and convert to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize for faster processing (optional)
    image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)
    
    # Reshape to a 2D array of pixels
    pixels = image.reshape(-1, 3)
    
    # Apply K-Means to find dominant color(s)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)
    
    # Get the RGB values of the cluster center
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    return tuple(dominant_color)

# Usage
for i in range(1,21):
    print(i)
    if i in {8,15}:
        pass
    else:
        dominant_rgb = get_dominant_color(f"result/foreground{i}.jpg")
        print(f"Dominant color (RGB): {dominant_rgb}")
        print(list(dominant_rgb))

        # Define image dimensions (width, height)
        width, height = 640, 480

        # Define the color in BGR format (e.g., blue: [255, 0, 0], green: [0, 255, 0], red: [0, 0, 255])
        color = [0, 0, 255]  # Red color

        # Create a blank image with the specified color
        image = np.zeros((height, width, 3), dtype=np.uint8)  # 3 channels for BGR
        image[:] = dominant_rgb  # Fill the image with the color

        # Save the image (optional)
        cv2.imwrite(f"data/solid_color_image{i}.jpg", image)

# # Display the image
# cv2.imshow("Solid Color Image", image)
# cv2.waitKey(0)  # Wait for a key press to close the window
# cv2.destroyAllWindows()