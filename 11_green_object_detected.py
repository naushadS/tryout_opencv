import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

while True:
    # Read each frame
    _, frame = cap.read()

    # Convert each frame from BGR color format to HSV
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    green_min = np.array([65, 60, 60], np.uint8)
    green_max = np.array([80, 255, 255], np.uint8)
    mask = cv.inRange(hsv, green_min, green_max)

    kernel = np.ones((5, 5), int)
    dilated = cv.dilate(mask, kernel)

    res = cv.bitwise_and(frame, frame, mask=mask)

    ret, thrshed = cv.threshold(cv.cvtColor(
        res, cv.COLOR_BGR2GRAY), 3, 255, cv.THRESH_BINARY)
    _, contours, _ = cv.findContours(
        thrshed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv.contourArea(cnt)

    if area > 1000:
        cv.putText(frame, 'Green Object Detected',
                   (10, 80), cv.FONT_HERSHEY_SIMPLEX, 1, 1)
        cv.rectangle(frame, (5, 40), (400, 100), (0, 255, 255), 2)
        
    # show images
    cv.imshow('Original', frame)
    cv.imshow('mask', mask)
    cv.imshow('Final Res', res)

    k = cv.waitKey(1) & 0xFF
    if k == 27:  # break using escape key
        break

cap.release()
cv.destroyAllWindows()
