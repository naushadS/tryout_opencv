import cv2

CAP = cv2.VideoCapture(0)

while CAP.isOpened():
    _, FRAME = CAP.read()
    GRAY = cv2.cvtColor(FRAME, cv2.COLOR_BGR2GRAY)
    cv2.imshow('grayframe', GRAY)

    # The waitKey(0) function returns - 1 when no input is made whatsoever.
    # As soon the event occurs i.e. a Button is pressed it returns a 32-bit integer.
    # The 0xFF in this scenario is representing binary 11111111 a 8 bit binary,
    # since we only require 8 bits to represent a character we AND waitKey(0) to 0xFF.
    # As a result, an integer is obtained below 255.
    # ord(char) returns the ASCII value of the character which would be again maximum 255.
    # Hence by comparing the integer to the ord(char) value, we can check for a key pressed event and break the loop.

    #1.waitKey(0) will display the window infinitely until any keypress (it is suitable for image display).

    #2.waitKey(1) will display a frame for 1 ms, after which display will be automatically closed

    #So, if you use waitKey(0) you see a still image until you actually press something while for waitKey(1) the function will show a frame for 1 ms only.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

CAP.release()
cv2.destroyAllWindows()
