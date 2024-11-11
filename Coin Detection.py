import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = '"D:\Assignment 2\coins_da1f4078-f2b1-414b-9db3-5a2a71f2dedf.jpg"'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Apply Hough Circle Transform to detect circles (coins)
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                           param1=50, param2=30, minRadius=20, maxRadius=60)

# Create a binary mask with the same dimensions as the original image
binary_mask = np.zeros_like(gray)

# if circle detect, draw on the binary mask
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw filled circles on the mask
        cv2.circle(binary_mask, (i[0], i[1]), i[2], (255), thickness=-1)

# Display output
plt.figure(figsize=(10, 5))
plt.imshow(binary_mask, cmap='gray')
plt.axis("off")
plt.show()
