import cv2
import numpy as np
# size - 400x400
LOGO = cv2.imread('3_jio.jpg', cv2.IMREAD_UNCHANGED)

# resize it to 200x200
LOGO = cv2.resize(LOGO, (100, 100))
LOGO_ROWS, LOGO_COLS, _ = LOGO.shape

CAP = cv2.VideoCapture(0)

while CAP.isOpened():
    _, FRAME = CAP.read()

    FRAME_ROWS, FRAME_COLS, _ = FRAME.shape

    #Region Of Image
    ROI = FRAME[FRAME_ROWS - LOGO_ROWS:FRAME_ROWS, FRAME_COLS - LOGO_COLS:FRAME_COLS]

    LOGO2GRAY = cv2.cvtColor(LOGO, cv2.COLOR_BGR2GRAY)

    _, MASK = cv2.threshold(LOGO2GRAY, 220, 255, cv2.THRESH_BINARY_INV)
    FRAME_FG = cv2.bitwise_and(LOGO, LOGO, mask=MASK)


    MASK_INV = cv2.bitwise_not(MASK)
    FRAME_BG = cv2.bitwise_and(ROI, ROI, mask=MASK_INV)

    LOGO_EMBOSSED_ON_ROI = cv2.add(FRAME_BG, FRAME_FG)
    FRAME[FRAME_ROWS - LOGO_ROWS:FRAME_ROWS, FRAME_COLS - LOGO_COLS:FRAME_COLS] = LOGO_EMBOSSED_ON_ROI

    cv2.imshow('video feed', FRAME)
    # cv2.imshow('logo', LOGO)
    # cv2.imshow('logo to gray', LOGO2GRAY)
    # cv2.imshow('mask', MASK)
    # cv2.imshow('frame_bg', FRAME_BG)
    # cv2.imshow('mask_inv', MASK_INV)
    # cv2.imshow('frame_fg', FRAME_FG)
    # cv2.imshow('logo embossed on roi',LOGO_EMBOSSED_ON_ROI)

    # The waitKey(0) function returns - 1 when no input is made whatsoever.
    # As soon the event occurs i.e. a Button is pressed it returns a 32-bit integer.
    # The 0xFF in this scenario is representing binary 11111111 a 8 bit binary,
    # since we only require 8 bits to represent a character we AND waitKey(0) to 0xFF.
    # As a result, an integer is obtained below 255.
    # ord(char) returns the ASCII value of the character which would be again maximum 255.
    # Hence by comparing the integer to the ord(char) value, we can check for a key pressed event and break the loop.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

CAP.release()
cv2.destroyAllWindows()
