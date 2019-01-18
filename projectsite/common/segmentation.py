from __future__ import print_function
import cv2 as cv
import numpy as np
import os
from django.conf import settings
from upload.models import Img, ImgParent
import tempfile
from matplotlib import pyplot as plt


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


def get_all_forams(uploaded_file):
    '''
    Applying the bounding boxes to the original image
    to create a new list of images
    Returns 
    '''
    # uploaded_file = uploaded_file.read()
    with tempfile.TemporaryDirectory() as dirpath:
        with open(os.path.join(dirpath, uploaded_file.name), 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
            img = cv.imread(os.path.join(dirpath, uploaded_file.name))
            boxes = filter_boxes(get_boxes(img, 100))
            forams = [img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] for box in boxes]
    return forams

    
def store_to_db(img, forams, toStore, ext):
    '''
    img: the original image
    forams: numpy array
    toStore: directory to store in
    ext: the file extension
    '''
    number_of_files = len(next(os.walk(toStore))[2])
    parent_location = os.path.join(toStore, str(number_of_files), ext)
    cv.imwrite(parent_location, img)
    parent_image = ImgParent(imgLocation=parent_location)
    parent_image.save()
    number_of_files += 1
    for foram in forams:    #stores segmented images
        img_location = os.path.join(toStore, str(number_of_files), ext)
        cv.imwrite(img_location, foram)
        new_image = Img(imgLocation=img_location,
                        species='G. ruber',         # this name is temporary
                        parentImage=parent_image)
        new_image.save()
        number_of_files += 1