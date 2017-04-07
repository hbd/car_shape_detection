import cv2
import sys
import numpy as np

video_capture = cv2.VideoCapture(0)

while True:

    # Take each frame
    ret, frame = video_capture.read()

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask = mask)

    # Get the contours of the [blue] objects
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)

    # Get frame width and height
    frameWidth = video_capture.get(3)
    frameHeight = video_capture.get(4)

    # Find the largest contour
    if contours:
        idx_biggest = 0
        val_biggest = cv2.contourArea(contours[0])

        for i in xrange(0, len(contours)):
            if cv2.contourArea(contours[i]) > val_biggest:
                val_biggest = cv2.contourArea(contours[i])
                idx_biggest = i

        # Get coordinates of center of largest contour,
        # draw rectangle around largest contour,
        # and place an X at the center of the largest contour
        M = cv2.moments(contours[idx_biggest])
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        x, y, w, h = cv2.boundingRect(contours[idx_biggest])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "X", (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if (cX <= frameWidth/3):
            cv2.putText(frame, "LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        elif ((cX > frameWidth/3) and (cX < frameWidth/3 * 2)):
            cv2.putText(frame, "CENTER", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.putText(frame, "RIGHT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow('contours', frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
