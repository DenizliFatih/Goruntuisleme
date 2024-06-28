import cv2
import numpy as np
import skimage.segmentation as seg
import skimage.future.graph as graph
from skimage import segmentation as seg
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import graph
import matplotlib.pyplot as plt

# read the input RGB image as BGR format
bgr_img = cv2.imread('C:/Users/Tibet/Desktop/icon2.jpg')
cv2.imshow('Original', bgr_img)
# Convert the BGR image to HSV Image
hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
cv2.imwrite('hsv_image.jpg', hsv_img)
cv2.waitKey(0)
# Display the HSV image
cv2.imshow('HSV image', hsv_img)
cv2.waitKey(0)

img_gray = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2GRAY)
# Blur the image for better edge detection
img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
 
# Sobel Edge Detection
sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
# Canny Edge Detection
edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
# Display Canny Edge Detection Image
cv2.imshow('Canny Edge Detection', edges)
cv2.waitKey(0)

image = img_as_float(bgr_img)
segments = slic(image, n_segments=250, compactness=10, sigma=1 )
fig=plt.figure("superpixels-- %d segments"%(250))
ax = fig.add_subplot(1,1,1)
ax.imshow(mark_boundaries(image,segments,color=(1,1,1),outline_color=(1,1,1)))

plt.axis('off')
plt.show()
rag = graph.ra
cv2.destroyAllWindows()