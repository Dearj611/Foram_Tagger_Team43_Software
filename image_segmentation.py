from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
import math
from matplotlib import pyplot as plt
rng.seed(12345)

def pre_processing(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]
    kernel = np.ones((5,5),np.uint8)
    thresh = cv.erode(thresh,kernel,iterations = 2)
    thresh = cv.dilate(thresh,kernel,iterations = 3)
    return thresh

def get_boxes(val):
    threshold = val
    canny_output = cv.Canny(src_gray, threshold, threshold * 2)
    _, contours, _ = cv.findContours(canny_output, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])
    print(boundRect)
    return boundRect

def remove_outliers(boxes):
    '''
    After getting more data, this function will use far more math
    '''
    boxes = list(zip(boxes,[i[2]*i[3] for i in boxes]))
    mean = sum(map(lambda x:x[1],boxes))/len(boxes)
    boxes = [i for i in boxes if i[1]>4000]
    to_return = [i[0] for i in boxes]
    print(to_return)
    print(len(to_return))
    return to_return

def remove_duplicates(boxes):
    '''
    For some reason i get very similar bounding boxes
    This function removes them
    '''
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
    print(to_return)
    return to_return

def visualize(img, boxes):
    for i in range(len(boxes)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])), \
          (int(boxes[i][0]+boxes[i][2]), int(boxes[i][1]+boxes[i][3])), color, 3)
        cv.putText(img, str(i), (int(boxes[i][0]), int(boxes[i][1])), \
            cv.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv.LINE_AA)
    plt.imshow(img)
    plt.show()

def get_forams(img, boxes):
    forams = [img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] for box in boxes]
    return forams

def normalize(forams, boxes):
    max_width = max(boxes, key=lambda x:x[2])[2]
    max_height = max(boxes, key=lambda x:x[3])[3]
    for i in range(len(boxes)):
        to_add_width = max_width-boxes[i][2]
        to_add_height = max_height-boxes[i][3]
        cv.copyMakeBorder(forams[i], math.ceil(to_add_height/2), math.floor(to_add_height/2),\
            math.ceil(to_add_width/2), math.ceil(to_add_width/2),cv.BORDER_CONSTANT)


img = cv.imread('../img/G.ruber-um-1.tif')
src_gray = pre_processing(img)
boxes = remove_outliers(remove_duplicates(get_boxes(100)))
forams = get_forams(img, boxes)
normalize(forams, boxes)
plt.imshow(forams[6])
plt.show()
# visualize(img, boxes)
