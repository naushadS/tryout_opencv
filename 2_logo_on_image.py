import cv2
import numpy as np

BASE = cv2.imread('2_base.jpg', cv2.IMREAD_COLOR)

# size - 400x400
LOGO = cv2.imread('2_logo.jpg', cv2.IMREAD_COLOR)

# resize it to 200x200
LOGO = cv2.resize(LOGO, (200,200))

# Whitening ROI(Region of Image)
# ImgSrc[ rowStart:rowEnd, columnStart:columnEnd ]
# BASE[50:100, 700:800] = [255, 255, 255]

rows, cols, _ = LOGO.shape
ROI = BASE[0:rows, 0:cols]

LOGO2GRAY = cv2.cvtColor(LOGO, cv2.COLOR_BGR2GRAY)
_, MASK = cv2.threshold(LOGO2GRAY, 220, 255, cv2.THRESH_BINARY_INV)

MASK_INV = cv2.bitwise_not(MASK)
BASE_BG = cv2.bitwise_and(ROI, ROI, mask=MASK_INV)
BASE_FG = cv2.bitwise_and(LOGO, LOGO, mask=MASK)

DST = cv2.add(BASE_BG, BASE_FG)
BASE[0:rows, 0:cols] = DST

cv2.imshow('mask',MASK)
cv2.imshow('mask_inv',MASK_INV)
cv2.imshow('BASE_FG',BASE_FG)
cv2.imshow('BASE_BG',BASE_BG)
cv2.imshow('DST', DST)
cv2.imshow('base', BASE)
# cv2.imshow('logo', LOGO)

cv2.waitKey(0)
cv2.destroyAllWindows()
