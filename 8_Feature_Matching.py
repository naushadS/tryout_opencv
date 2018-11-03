import cv2
import numpy as np
# import matplotilb.pyplot as plt

template = cv2.imread('8_template_for_matching.jpg',cv2.IMREAD_GRAYSCALE)
img = cv2.imread('8_feature_matching_image.jpg',cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(template, None)
kp2, des2 = orb.detectAndCompute(img, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(template, kp1, img, kp2, matches[:10], None, flags=2)
# cv2.imshow('img3',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
