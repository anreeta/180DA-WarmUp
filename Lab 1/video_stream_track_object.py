"""
    Usage Instructions:
    python video_stream_track_object.py COLOR_SCHEME
        COLOR_SCHEME is the color scheme ("HSV", "RGB" in which to perform analysis in)

    References:
    * 'Getting Started with Videos' page: Capturing Video from Camera section
        https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
    * For min/max HSV parameters/understanding how color is specified: 
        https://docs.opencv.org/3.4.15/da/d97/tutorial_threshold_inRange.html
    * For the idea of sorting contours/structure of thresholding and creating boudning box:
        https://stackoverflow.com/questions/47574173/how-can-i-draw-a-rectangle-around-a-colored-object-in-open-cv-python

    Improvements:
    * Determining thresholding for HSV and RGB values
    * Command-line options to python code, and support for multiple color schemes

"""

import numpy as np
import cv2
import sys

cap = cv2.VideoCapture(0)

COLORSCHEME = {"HSV": cv2.COLOR_BGR2HSV, "RGB": cv2.COLOR_BGR2RGB}

# HSV parameters
max_value = 255
max_value_H = 360//2
low_H = 0
low_S = 0
low_V = 0
high_H = max_value_H
high_S = max_value
high_V = max_value

if len(sys.argv) != 2:
    print("Provide a COLOR_SCHEME command line parameter")
    exit(1)

# Parameters for color of monochrome object
if sys.argv[1] == "HSV":
    LOWER_COLOR = (90, low_S+30, low_V+30) # (125, 50, 20)
    UPPER_COLOR = (110, high_S-30, high_V-30) #(330, 100, 100)
elif sys.argv[1] == "RGB":
    LOWER_COLOR = (50, 50, 50)
    UPPER_COLOR = (100, 100, 255)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    ## Convert to different color scheme and add bounding box
    color_convert = cv2.cvtColor(frame, COLORSCHEME[sys.argv[1]])

    #ret, thresh = cv2.threshold(color_convert, 0, 255, cv2.THRESH_BINARY)
    thresh = cv2.inRange(color_convert, LOWER_COLOR, UPPER_COLOR)

    cv2.imshow('thresh', thresh)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours[0])
    cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # # ## Create the bounding box
    if cont_sorted:
        x,y,w,h = cv2.boundingRect(cont_sorted[0])
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    # # # Display the resulting frame with boudning box
    cv2.imshow('frame',frame)

    # Quit if q key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()