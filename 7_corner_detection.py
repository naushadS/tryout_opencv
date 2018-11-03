import cv2
import numpy as np

# Webcam Video

CAP = cv2.VideoCapture(0)

while CAP.isOpened():
    _, FRAME = CAP.read()
    gray = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, 3000, 0.01, 5) # args - img , np of max corner to detect, quality, minimum distance between two detected points
    corners = np.int0(corners) # np.int0 is same as np.int32 or np.int64 depending upon the system's architecture

    for corner in corners:
        # contiguous flattened array(1D array with all the input-array elements and with the same type as it
        x, y = corner.ravel()
        cv2.circle(FRAME, (x, y), 2, (0, 255, 0), -1)

    cv2.imshow('Corner', FRAME)

    if cv2.waitKey(1) & 0xFF == 27:
        break

CAP.release()
cv2.destroyAllWindows()

#IMAGE

# img = cv2.imread('7_corner-detection.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = np.float32(gray)

# corners = cv2.goodFeaturesToTrack(gray, 3000, 0.01, 8)
# corners = np.int0(corners)

# for corner in corners:
#     x, y = corner.ravel()
#     cv2.circle(img, (x,y), 2, (0,255,0), -1)

# cv2.imshow('Corner',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
