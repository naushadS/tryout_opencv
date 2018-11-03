import cv2
import numpy as np

img  = cv2.imread('6_for_template_matching.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.imread('6_template.jpg',cv2.IMREAD_GRAYSCALE)
h, w = template.shape[::]
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.80
loc = np.where(res >= threshold) # tell me where in a entry is >= threshold

for pt in zip(*loc[::-1]):
    print(pt)
    cv2.rectangle(img, pt, (pt[0]+w, pt[1]+h), (0,255,255), 1)

cv2.imshow('img',img)
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()
