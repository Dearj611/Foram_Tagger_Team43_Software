# Script to time image processing algorithm
import cv2 as cv
import numpy as np
import os
from datetime import datetime
startTime = datetime.now()


def pre_processing(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]
    kernel = np.ones((5,5),np.uint8)
    thresh = cv.erode(thresh, kernel, iterations=2)
    thresh = cv.dilate(thresh, kernel, iterations=3)
    return thresh


def get_boxes(img, val):
    '''
    img provided must be grayscaled
    '''
    img = pre_processing(img)
    threshold = val
    canny_output = cv.Canny(img, threshold, threshold * 2)
    _, contours, _ = cv.findContours(canny_output, cv.RETR_TREE,
                                     cv.CHAIN_APPROX_SIMPLE)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
    return boundRect


def filter_boxes(boxes):
    '''
    This function removes boxes that are too small
    It also removes bounding boxes that essentially bound the same region
    Can include more filtering functionality in the future
    '''
    boxes = [arr for arr in boxes if(arr[2]*arr[3]) > 1000]
    boxes = sorted(boxes, key=lambda x:x[0])
    to_return = []
    for i in range(len(boxes)-1):
        if abs(boxes[i][0]-boxes[i+1][0])<5 and abs(boxes[i][1]-boxes[i+1][1])<5:
            if abs(boxes[i][2]-boxes[i+1][2])<5 or abs(boxes[i][3]-boxes[i+1][3])<5:
                pass
            else:
                to_return.append(boxes[i])
        else:
            to_return.append(boxes[i])
    return to_return


def segment(path):
    for files in os.listdir(path):
        img = cv.imread(os.path.join(path, files))
        filter_boxes(get_boxes(img, 100))


segment('./parent')
print(datetime.now() - startTime)