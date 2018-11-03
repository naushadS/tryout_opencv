import cv2
import numpy as np

BOOKPAGE = cv2.imread('bookpage.jpg', cv2.IMREAD_COLOR)
GRAYSCALED = cv2.cvtColor(BOOKPAGE, cv2.COLOR_BGR2GRAY)
# _, THRESH = cv2.threshold(GRAYSCALED, 15, 255, cv2.THRESH_BINARY)
GAUS = cv2.adaptiveThreshold(GRAYSCALED, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 155, 1)
cv2.imshow('original',BOOKPAGE)
# cv2.imshow('thresh',THRESH)

# kernel = np.ones((5,5),np.uint8)
# GAUS = cv2.bitwise_not(GAUS)
# ERODED = cv2.erode(GAUS, kernel, iterations = 1)
# DILATE = cv2.dilate(GAUS, kernel, iterations=1)

# cv2.imshow('erode', ERODED)
# cv2.imshow('dilate', DILATE)
cv2.imshow('gaus', GAUS)

cv2.waitKey(0)
cv2.destroyAllWindows()
