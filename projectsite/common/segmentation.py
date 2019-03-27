from __future__ import print_function
import cv2 as cv
import numpy as np
import os, errno
import random as rng
from upload.models import Img, ImgParent, Species
import tempfile
import uuid
from django.db.models import F
from azure.storage.blob import BlockBlobService, PublicAccess
from common import inference
import json


def draw_on_image(img, boxes):
    '''
    Returns an image with bounding boxes drawn on
    '''
    for i in range(len(boxes)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.rectangle(img, (int(boxes[i][0]), int(boxes[i][1])),
          (int(boxes[i][0]+boxes[i][2]), int(boxes[i][1]+boxes[i][3])), color, 3)
        cv.putText(img, str(i), (int(boxes[i][0]), int(boxes[i][1])),
                   cv.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv.LINE_AA)
    return img


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
    with tempfile.TemporaryDirectory() as dirpath:
        with open(os.path.join(dirpath, uploaded_file.name), 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)
            img = cv.imread(os.path.join(dirpath, uploaded_file.name))
            boxes = filter_boxes(get_boxes(img, 100))
            forams = [img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]] for box in boxes]
    return forams, img

    
def store_to_db(parent_img, forams, species, toStore):
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
    parent_location = os.path.join(parent_dir, uuid.uuid4().hex) + '.jpg'
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
        img_location = os.path.join(toStore, species, filename) + '.jpg'
        cv.imwrite(img_location, foram)
        new_image = Img(imgLocation=img_location,
                        species=species_obj,
                        parentImage=parent_image)
        species_counter += 1
        new_image.save()
    species_obj.total = F('total') + species_counter
    species_obj.save()
    return parent_image


def store_to_remote_db(parent_img, forams, species, block_blob_service):
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
    species_counter = 0
    with tempfile.TemporaryDirectory() as dirpath:
        filename = uuid.uuid4().hex + '.jpg'
        filepath = os.path.join(dirpath, filename)
        cv.imwrite(filepath, parent_img)
        block_blob_service.create_container('parent')
        block_blob_service.set_container_acl('parent', public_access=PublicAccess.Container)
        block_blob_service.create_blob_from_path('parent', filename, filepath)
        parent_image = ImgParent(imgLocation=os.path.join('parent', filename))
        parent_image.save()
        for foram in forams:    # stores segmented images
            filename = uuid.uuid4().hex + '.jpg'
            filepath = os.path.join(dirpath, filename)
            cv.imwrite(filepath, foram)
            block_blob_service.create_container(species)
            block_blob_service.set_container_acl(species, public_access=PublicAccess.Container)
            block_blob_service.create_blob_from_path(species, filename, filepath)
            new_image = Img(imgLocation=os.path.join(species, filename),
                            species=species_obj,
                            parentImage=parent_image)
            species_counter += 1
            new_image.save()
        species_obj.total = F('total') + species_counter
        species_obj.save()

'''
Much of what segmentation does is the same as the processing script. However I
created a class here because the functions above are randomly processing stuff
then randomly returning certain things. It is much better to implement a class,
so that I can keep tract of the state
'''
class Foram:
    def __init__(self, uploaded_file):
        self.block_blob_service = BlockBlobService(os.environ['AZ_STORAGE_ACCOUNT_NAME'], os.environ['AZ_STORAGE_KEY'])
        self.container = 'media'
        self.parent_obj = None
        self.species_obj = None
        with tempfile.TemporaryDirectory() as dirpath:
            with open(os.path.join(dirpath, uploaded_file.name), 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
                self.parent_img = cv.imread(os.path.join(dirpath, uploaded_file.name))
                self.boxes = filter_boxes(get_boxes(self.parent_img, 100))
                self.forams = [self.parent_img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] for box in self.boxes]

    def store_parents(self):
        with tempfile.TemporaryDirectory() as dirpath:
            parent_name = uuid.uuid4().hex + '.jpg'
            edited_name = uuid.uuid4().hex + '.jpg'
            cv.imwrite(os.path.join(dirpath, parent_name), self.parent_img)
            cv.imwrite(os.path.join(dirpath, edited_name), draw_on_image(self.parent_img, self.boxes))
            self.block_blob_service.create_blob_from_path(self.container,
                                                          os.path.join('parent', parent_name),
                                                          os.path.join(dirpath, parent_name))
            self.block_blob_service.create_blob_from_path(self.container, 
                                                          os.path.join('parent-edited', edited_name), 
                                                          os.path.join(dirpath, edited_name))
            self.parent_obj = ImgParent(imgLocation=os.path.join('parent', parent_name),
                                        imgEdited=os.path.join('parent-edited', edited_name))
            self.parent_obj.save()

    def store_children(self):
        with tempfile.TemporaryDirectory() as dirpath:
            for num, foram in enumerate(self.forams):    # stores segmented images
                child_name = uuid.uuid4().hex + '.jpg'
                species = self.species_obj[num].name
                cv.imwrite(os.path.join(dirpath, child_name), foram)
                self.block_blob_service.create_blob_from_path(self.container, 
                                                              os.path.join(species, child_name),
                                                              os.path.join(dirpath, child_name))
                new_image = Img(imgLocation=os.path.join(species, child_name),
                                species=self.species_obj[num],
                                parentImage=self.parent_obj,
                                number_on_image=num)
                new_image.save()

    def set_species(self):
        '''
        Sets self.species_obj to an array of species objects
        '''
        species_obj_list = []
        data = [foram.tolist() for foram in self.forams]
        predictions = inference.run(json.dumps({"data": data}))
        predictions = json.loads(predictions)["data"]
        predictions = [(key, value["true_class"]) for key, value in predictions.items()]
        predictions = sorted(predictions, key=lambda x: x[0])
        predictions = [i[1] for i in predictions]
        for species in predictions:
            try:
                species_obj = Species.objects.get(name=species)
            except Species.DoesNotExist:
                species_obj = Species(name=species)
                species_obj.save()
            species_obj_list.append(species_obj)
        self.species_obj = species_obj_list

