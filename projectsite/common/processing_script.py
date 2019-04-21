from __future__ import print_function
import cv2 as cv
import numpy as np
import random as rng
import math
import os, sys, errno
import statistics
import regex as re
from matplotlib import pyplot as plt
from django.conf import settings
from upload.models import Img, ImgParent, Species
import uuid
import tempfile
from azure.storage.blob import BlockBlobService, PublicAccess
from django.db.models import F
from azure.common import AzureMissingResourceHttpError
rng.seed(12345)

def pre_processing(img):
    img = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]
    kernel = np.ones((5,5),np.uint8)
    thresh = cv.erode(thresh, kernel, iterations=2)
    thresh = cv.dilate(thresh, kernel, iterations=3)
    return thresh


def get_boxes(img, val):
    '''
    img: img path
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


def get_outliers(arr):
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3-q1
    min_outlier = q1 - (1.5*iqr)
    max_outlier = q3 + (1.5*iqr)
    return min_outlier, max_outlier


def filter_boxes(boxes):
    '''
    This function removes bounding boxes that are too small
    It also removes boxes that essentially bound the same region
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


def visualize_one(img, box):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv.rectangle(img, (int(box[0]), int(box[1])),
          (int(box[0]+box[2]), int(box[1]+box[3])), color, 3)
    plt.imshow(img)
    plt.show()


#'/home/camelcars/Documents/ucl2/systemsEng/872 1-1-0.2a G. ruber 300-355 um 1.jpg'
def draw_on_image(img, boxes):
    '''
    Returns an image with bounding boxes drawn on
    '''
    img = img.copy()
    for i in range(len(boxes)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])),
          (int(boxes[i][0]+boxes[i][2]), int(boxes[i][1]+boxes[i][3])), color, 3)
        cv.putText(img, str(i), (int(boxes[i][0]), int(boxes[i][1])),
                   cv.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv.LINE_AA)
    return img

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
    name = '-'.join(name)
    if '-' not in name:
        i = 0
        while name[i] != '.':
            i += 1
        name = name[:i+1] + '-' + name[i+1:]
    name = ''.join(name.split('.')).lower()
    return name


def store_to_remote_db(parent_img, forams, edited_image, species, block_blob_service, container):
    '''
    parent_img: numpy array
    forams: numpy array
    species: str
    The first part stores the parent image in the parent container
    the For loop stores the segmented images
    '''
    try:
        species_obj = Species.objects.get(name=species)
    except Species.DoesNotExist:
        species_obj = Species(name=species)
        species_obj.save()
    with tempfile.TemporaryDirectory() as dirpath:  # store parent and edited parent
        parent_name = uuid.uuid4().hex + '.jpg'
        edited_name = uuid.uuid4().hex + '.jpg'
        cv.imwrite(os.path.join(dirpath, parent_name), parent_img)
        cv.imwrite(os.path.join(dirpath, edited_name), edited_image)
        block_blob_service.create_blob_from_path(container,
                                                os.path.join('parent', parent_name),
                                                os.path.join(dirpath, parent_name))
        block_blob_service.create_blob_from_path(container, 
                                                os.path.join('parent-edited', edited_name), 
                                                os.path.join(dirpath, edited_name))
        parent_obj = ImgParent(imgLocation=os.path.join('parent', parent_name),
                               imgEdited=os.path.join('parent-edited', edited_name))
        parent_obj.save()
        for num, foram in enumerate(forams):    # stores segmented images
            child_name = uuid.uuid4().hex + '.jpg'
            cv.imwrite(os.path.join(dirpath, child_name), foram)
            block_blob_service.create_blob_from_path(d,a,)
            block_blob_service.create_blob_from_path(container, 
                                                     os.path.join(species, child_name),
                                                     os.path.join(dirpath, child_name))
            new_image = Img(imgLocation=os.path.join(species, child_name),
                            species=species_obj,
                            parentImage=parent_obj,
                            number_on_image=num)
            new_image.save()


# The original file I used was G.ruber-um-1.tif
# toStore = 
# from common import processing_script as proc
# proc.get_and_store('../../source-img/')
def get_and_store(imgDir):
    '''
    The function populates the database and a directory
    imgDir: the source of all the parent images
    '''
    block_blob_service = BlockBlobService(os.environ['AZ_STORAGE_ACCOUNT_NAME'], os.environ['AZ_STORAGE_KEY'])
    container = 'media'
    block_blob_service.create_container(container)
    block_blob_service.set_container_acl(container, public_access=PublicAccess.Container)
    for dirpath, directory, filename in os.walk(imgDir):
        if len(filename) == 0:
            continue
        for files in filename:  # filename is a list of files
            parent_img = cv.imread(os.path.join(dirpath, files))
            boxes = filter_boxes(get_boxes(parent_img, 100))
            forams = [parent_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] for box in boxes]
            edited_img = draw_on_image(parent_img, boxes)
            store_to_remote_db(parent_img, forams, edited_img, get_species_name(files),
                               block_blob_service, container)


def filter_table():
    '''
    Removes images that are too small and outliers from the database
    for each species
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


def clean_account():
    '''Removes all containers'''
    block_blob_service = BlockBlobService(account_name='forampics', account_key='4nwt5cexYaNCgmsk5NrLLm5lmRprYobFVepz+hhb6b7hv2f6zifM1EPmoqT7SMTsUYvWSe3nREd/dS6g8Thjmg==')
    containers = block_blob_service.list_containers()
    for c in containers:
        if c.name == 'allstaticfiles':
            pass
        else:
            block_blob_service.delete_container(c.name)