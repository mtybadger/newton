import cv2
import numpy as np

# Read image and isolate red
image = cv2.imread("./cubes.jpg")  # replace with your actual path

def isolate_red(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0,   100, 100])
    upper_red1 = np.array([10,  255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Clean up the mask
    kernel = np.ones((5,5), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    return red_mask

red_mask = isolate_red(image)

# Find contours in the mask
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assuming it's the red cube)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Draw the bounding box
    vis_image = image.copy()
    cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
    
    # Print the bounding box coordinates
    print(f"Bounding box coordinates:")
    print(f"xmin: {x}, ymin: {y}, xmax: {x + w}, ymax: {y + h}")
    
    # Display result
    cv2.imshow("Red cube bounding box", vis_image)
    cv2.waitKey(5000)
else:
    print("No red cube detected")
