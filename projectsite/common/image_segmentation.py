from __future__ import print_function
import cv2 as cv
import numpy as np
import random as rng
import math
import os, errno
import statistics
import regex as re
from matplotlib import pyplot as plt
from django.conf import settings
from upload.models import Img, ImgParent, Species
import uuid
from django.db.models import F

rng.seed(12345)

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
    # print(boundRect)
    return boundRect


def remove_outliers(arr):
    '''
    This uses the IQR method
    '''
    arr = sort_by_extremes(arr)
    mean = sum(arr)/len(arr)
    iqr = np.percentile(arr, 75)-np.percentile(arr, 25)
    flag = False
    if (max(arr)-mean) > iqr*1.5 or (mean-min(arr) > iqr*1.5):
        flag = True
    while flag:
        old_len = len(arr)
        new_len = len(arr)
        for i in range(len(arr)):
            if abs(arr[i]-mean) > iqr*1.5:
                arr.pop(i)
                new_len -= 1
                break
        if old_len != new_len:
            mean = sum(arr)/len(arr)
            iqr = np.percentile(arr, 75)-np.percentile(arr, 25)
        else:
            break
    return arr


def remove_outliers_2(arr):
    '''
    This uses the more common IQR method
    The other remove_outliers method removes way too much correct data
    '''
    arr = [i for i in arr if i > 1000]
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3-q1
    min_outlier = q1 - (1.5*iqr)
    max_outlier = q3 + (1.5*iqr)
    arr = [i for i in arr if i < max_outlier and i > min_outlier]
    return arr

def get_outliers(arr):
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3-q1
    min_outlier = q1 - (1.5*iqr)
    max_outlier = q3 + (1.5*iqr)
    return min_outlier, max_outlier


def sort_by_extremes(arr):
    '''
    a = [1,3,6,2,9,19] becomes
    a = [1,19,2,9,3,6]
    Helper function for remove_outliers
    '''
    arr = sorted(arr)
    to_return = []
    i = 0
    while arr != []:
        if i % 2 == 0:
            to_return.append(arr.pop(0))
        else:
            to_return.append(arr.pop())
        i += 1
    return to_return


def filter_boxes(boxes):
    '''
    This function removes boxes that are too small
    It also removes bounding boxes that essentially bound the same region
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


def visualize_all(img, boxes):
    for i in range(len(boxes)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])),
          (int(boxes[i][0]+boxes[i][2]), int(boxes[i][1]+boxes[i][3])), color, 3)
        cv.putText(img, str(i), (int(boxes[i][0]), int(boxes[i][1])),
                   cv.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv.LINE_AA)
    plt.imshow(img)
    plt.show()
    print('visualize_all running')


def visualize_one(img, box):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv.rectangle(img, (int(box[0]), int(box[1])),
          (int(box[0]+box[2]), int(box[1]+box[3])), color, 3)
    plt.imshow(img)
    plt.show()


def get_forams(img, box):
    return img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]


def normalize(forams, boxes):
    '''
    Makes the forams equal size
    '''
    max_width = max(boxes, key=lambda x:x[2])[2]
    max_height = max(boxes, key=lambda x:x[3])[3]
    for i in range(len(boxes)):
        to_add_width = max_width-boxes[i][2]
        to_add_height = max_height-boxes[i][3]
        cv.copyMakeBorder(forams[i], math.ceil(to_add_height/2), math.floor(to_add_height/2),
            math.ceil(to_add_width/2), math.ceil(to_add_width/2), cv.BORDER_CONSTANT)


def freedman_diaconis(arr):
    '''
    used to calculate the number of bins for a histogram
    wiki this equation
    '''
    iqr = np.percentile(arr, 75)-np.percentile(arr, 25)
    bin_width = (2*iqr)/(len(arr)**(1/3))
    print(int(bin_width))
    return int(bin_width)


def some_stats(arr):
    print('data points', len(arr))
    print('mean', sum(arr)/len(arr))
    # print('mode', statistics.mode(arr))
    print('median', statistics.median(arr))
    print('min', min(arr))
    print('max', max(arr))
    print('range', max(arr)-min(arr))
    print('interquartile range', np.percentile(arr, 75)-np.percentile(arr, 25))
    print('standard deviation', statistics.pstdev(arr))


def get_species_name(string):
    name = []
    arr = string.split(' ')
    for i in arr:
        if re.search('[0-9]', i):
            pass
        elif i == 'um':
            pass
        else:
            name.append(i)
    name = ' '.join(name)
    if ' ' not in name:
        i = 0
        while name[i] != '.':
            i += 1
        name = name[:i+1] + ' ' + name[i+1:]
    return name


def get_all_forams(img):
    '''
    Applying the bounding boxes to the original image
    to create a new list of images
    '''
    boxes = filter_boxes(get_boxes(img, 100))
    forams = [img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] for box in boxes]
    return forams


def store_to_db(parent_img, forams, species, toStore, ext):
    '''
    parent_img: the original image
    forams: numpy array
    toStore: directory to store in
    ext: the file extension
    The first part stores the parent image in the parent dir
    the For loop stores the segmented images
    '''
    try:
        species_obj = Species.objects.get(name=species)
    except Species.DoesNotExist:
        species_obj = Species(name=species, total=0)
        species_obj.save()
    species_counter = 0
    parent_dir = os.path.join(toStore, 'parent')
    try:
        os.mkdir(parent_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    parent_location = os.path.join(parent_dir, uuid.uuid4().hex) + ext
    cv.imwrite(parent_location, parent_img)
    parent_image = ImgParent(imgLocation=parent_location)
    parent_image.save()
    try:
        os.mkdir(os.path.join(toStore, species))    # create child directory    
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    for foram in forams:    # stores segmented images
        filename = uuid.uuid4().hex
        img_location = os.path.join(toStore, 'train', species, filename) + ext
        cv.imwrite(img_location, foram)
        new_image = Img(imgLocation=img_location,
                        species=species_obj,
                        parentImage=parent_image)
        species_counter += 1
        new_image.save()
    species_obj.total = F('total') + species_counter
    species_obj.save()

    
# The original file I used was G.ruber-um-1.tif
# toStore = 
# get_and_store('../../img', './media')
def get_and_store(imgDir, toStore):
    '''
    The function populates the database and a directory
    imgDir: the source of all the parent images
    toStore: the directory where you want to store the images
    '''
    counter = 0
    for dirpath, directory, filename in os.walk(imgDir):
        if len(filename) == 0:
            continue
        for files in filename:  # filename is a list of files
            parent_img = cv.imread(os.path.join(dirpath, files))
            forams = get_all_forams(parent_img)
            store_to_db(parent_img, forams, get_species_name(files), toStore,
                        os.path.splitext(files)[1])


def filter_table():
    '''
    Removes images that are too small and outliers from the database
    '''
    all_species = Img.objects.values_list('species', flat=True).distinct()
    for species in all_species:
        all_objects = Img.objects.all().filter(species=species)
        img = [cv.imread(str(img.imgLocation)) for img in all_objects]
        all_dimensions = [i.shape[0]*i.shape[1] for i in img]
        min_outlier, max_outlier = get_outliers(all_dimensions)
        for i in all_objects:
            img = cv.imread(str(i.imgLocation))
            dimensions = img.shape[0] * img.shape[1]
            if dimensions < 1000:
                i.delete()
            elif dimensions < min_outlier or dimensions > max_outlier:
                i.delete()

