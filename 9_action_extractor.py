import cv2
import numpy as np

video = cv2.VideoCapture(0)
action_extractor = cv2.createBackgroundSubtractorMOG2()

while video.isOpened():
    RET, FRAME = video.read()
    if RET== 0:
        break

    MASK = action_extractor.apply(FRAME)

    MASK = cv2.erode(MASK, np.ones((3, 3),np.uint8),iterations=1)
    # cv2.imshow('MASK', MASK)
    #MORPH_OPEN = Dilate then Erode
    # OPEN = cv2.morphologyEx(MASK, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    _, thres = cv2.threshold(MASK,1,255,cv2.THRESH_BINARY)
    res = cv2.bitwise_and(FRAME, FRAME, mask=thres)
    cv2.imshow('ori', FRAME)
    cv2.imshow('res', res)

    if cv2.waitKey(1) & 0xFF == 27:
        break
video.release()
cv2.destroyAllWindows()
# import cv2
# import numpy as np

# video = cv2.VideoCapture('9_people_walking.mp4')
# action_extractor = cv2.createBackgroundSubtractorMOG2()

# while video.isOpened():
#     RET, FRAME = video.read()
#     if RET== 0:
#         break

#     MASK = action_extractor.apply(FRAME)

#     MASK = cv2.erode(MASK, np.ones((3, 3),np.uint8),iterations=1)
#     # cv2.imshow('MASK', MASK)
#     #MORPH_OPEN = Dilate then Erode
#     # OPEN = cv2.morphologyEx(MASK, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
#     # res = cv2.bitwise_and(FRAME, FRAME, mask=MASK)
#     # gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
#     _, thres = cv2.threshold(MASK,1,255,cv2.THRESH_BINARY_INV)
#     cv2.imshow('ori', FRAME)
#     cv2.imshow('thres', thres)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break
# video.release()
# cv2.destroyAllWindows()


