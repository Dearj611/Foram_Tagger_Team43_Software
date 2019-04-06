from __future__ import print_function
import cv2 as cv
import numpy as np
import os, errno
import random as rng
from upload.models import Img, ImgParent, Species
import uuid
from django.db.models import F
from azure.storage.blob import PublicAccess
from common import inference
import json
from azure.storage.blob import BlockBlobService

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
'''
Much of what segmentation does is the same as the processing script. However I
created a class here because the functions above are randomly processing stuff
then randomly returning certain things. It is much better to implement a class,
so that I can keep track of the state. Also the state is heavily used in the views
'''
class Foram:
    '''
    I chose to not include the helper functions above as they do not
    include database transactions
    '''
    container = 'media'
    block_blob_service = BlockBlobService(os.environ['AZ_STORAGE_ACCOUNT_NAME'], os.environ['AZ_STORAGE_KEY'])

    def __init__(self, uploaded_file, dirpath):
        self.uploaded_file = uploaded_file
        self.dirpath = dirpath
        self.parent_obj = None
        self.species_obj = None
        self.parent_img = None
        self.boxes = None
        self.forams = None

    def segment(self):
        with open(os.path.join(self.dirpath, self.uploaded_file.name), 'wb+') as destination:
            for chunk in self.uploaded_file.chunks():
                destination.write(chunk)
            self.parent_img = cv.imread(os.path.join(self.dirpath, self.uploaded_file.name))
            self.boxes = filter_boxes(get_boxes(self.parent_img, 100))
            self.forams = [self.parent_img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] for box in self.boxes]

    def store_parents(self):
        parent_name = uuid.uuid4().hex + '.jpg'
        edited_name = uuid.uuid4().hex + '.jpg'
        cv.imwrite(os.path.join(self.dirpath, parent_name), self.parent_img)
        cv.imwrite(os.path.join(self.dirpath, edited_name), draw_on_image(self.parent_img, self.boxes))
        self.block_blob_service.create_blob_from_path(self.container,
                                                      os.path.join('parent', parent_name),
                                                      os.path.join(self.dirpath, parent_name))
        self.block_blob_service.create_blob_from_path(self.container, 
                                                      os.path.join('parent-edited', edited_name), 
                                                      os.path.join(self.dirpath, edited_name))
        self.parent_obj = ImgParent(imgLocation=os.path.join('parent', parent_name),
                                    imgEdited=os.path.join('parent-edited', edited_name))
        self.parent_obj.save()

    def store_children(self):
        for num, foram in enumerate(self.forams):    # stores segmented images
            child_name = uuid.uuid4().hex + '.jpg'
            species = self.species_obj[num].name
            cv.imwrite(os.path.join(self.dirpath, child_name), foram)
            self.block_blob_service.create_blob_from_path(self.container, 
                                                          os.path.join(species, child_name),
                                                          os.path.join(self.dirpath, child_name))
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

    @classmethod
    def delete_foram(cls, id):
        img = Img.objects.get(id=id)
        cls.block_blob_service.delete_blob(cls.container, img.imgLocation.name)
        img.delete()

    @classmethod
    def update_species(cls, id, species):
        try:
            corrected_species = Species.objects.get(name=species)
        except Species.DoesNotExist:
            corrected_species = Species(species)
            corrected_species.save()
        img = Img.objects.get(id=id)
        new_location = os.path.join(corrected_species.name, os.path.basename(img.imgLocation.name))
        img_url = (img.imgLocation.url[:8] + img.imgLocation.url[7:])[1:]
        copy_blob = cls.block_blob_service.copy_blob(cls.container,
                                                     new_location,
                                                     img_url)
        if copy_blob.status == 'success':
            cls.block_blob_service.delete_blob(cls.container, img.imgLocation.name)
        else:
            print(copy_blob.status)
        Img.objects.filter(pk=id).update(imgLocation=new_location, species=corrected_species)
