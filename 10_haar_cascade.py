import cv2
import numpy as np

cascade_face = cv2.CascadeClassifier('haarcascade_frontalface.xml')
cascade_eye = cv2.CascadeClassifier('haarcascade_eye.xml')
# cascade_lefteye = cv2.CascadeClassifier('haarcascade_lefteye.xml')
# cascade_righteye = cv2.CascadeClassifier('haarcascade_righteye.xml')
cascade_smile = cv2.CascadeClassifier('haarcascade_smile.xml')

CAP = cv2.VideoCapture(0)

while True:
    RET, FRAME = CAP.read()
    if RET == 0:
        break
    
    GRAY = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)
    FACES = cascade_face.detectMultiScale(GRAY, 1.5, 5)
    for (x, y, w, h) in FACES:
        cv2.rectangle(FRAME, (x,y), (x+w,y+h), (255,0,0), 2)
        roi_gray = GRAY[y:y+h, x:x+w]
        roi_color = FRAME[y:y+h, x:x+w]
        EYES = cascade_eye.detectMultiScale(roi_gray, 1.5, 5)
        for (ex, ey, ew, eh) in EYES:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        SMILE = cascade_smile.detectMultiScale(roi_gray, 1.5, 5)
        for (sx, sy, sw, sh) in SMILE:
            cv2.rectangle(roi_color, (sx, sy),(sx+sw, sy+sh), (0, 0, 255), 2)

    # SMILE = cascade_face.detectMultiScale(GRAY)
    # for (sx, sy, sw, sh) in FACES:
    #     cv2.rectangle(FRAME, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
    
    cv2.imshow('faces', FRAME)
    if cv2.waitKey(10) & 0xFF == 27:
        break

CAP.release()
cv2.destroyAllWindows()
