import cv2

CAP = cv2.VideoCapture(0)

while CAP.isOpened():
    _, FRAME = CAP.read()

    LAPLACIAN = cv2.Laplacian(FRAME, cv2.CV_64F)
    sobelx = cv2.Sobel(FRAME, cv2.CV_64F, 0, 1, ksize=5)
    sobely = cv2.Sobel(FRAME, cv2.CV_64F, 1, 0, ksize=5)
    canny = cv2.Canny(FRAME, 80, 140)

    cv2.imshow('Original',FRAME)
    cv2.imshow('Laplacian',LAPLACIAN)
    cv2.imshow('sobelx',sobelx)
    cv2.imshow('sobely',sobely)
    cv2.imshow('canny',canny)

    if cv2.waitKey(0) & 0xFF == 27:
        break
CAP.release()        
cv2.destroyAllWindows()
