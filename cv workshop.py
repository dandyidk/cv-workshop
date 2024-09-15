import cv2
import cv2 as cv
import numpy as np
file = r"C:\Users\DELL\Desktop\cv_workshop_data\slot 1\1.jpg"

img = cv.imread(file)
img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)

img_original = cv.imread(file)
img_original = cv.resize(img_original, (0, 0), fx=0.5, fy=0.5)

img_gray = cv.imread(file, cv.IMREAD_GRAYSCALE)
img_gray = cv.resize(img_gray, (0, 0), fx=0.5, fy=0.5)

img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

circles = cv.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,1.2,100)

canny = cv.Canny(img_gray,100,200)

contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
cv.drawContours(img, contours, -1, (255, 0, 0), 2)

cv.imshow('Original', img_gray)
cv.imshow('Detected_Shapes', img)
cv.imwrite('new_shape_color_detection.png', img)

cv.waitKey(0)
cv.destroyAllWindows()

