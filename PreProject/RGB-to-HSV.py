import cv2
# read the input RGB image as BGR format
bgr_img = cv2.imread('C:/Users/Tibet/Desktop/icon2.jpg')

# Convert the BGR image to HSV Image
hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
cv2.imwrite('hsv_image.jpg', hsv_img)

# Display the HSV image
cv2.imshow('HSV image', hsv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()