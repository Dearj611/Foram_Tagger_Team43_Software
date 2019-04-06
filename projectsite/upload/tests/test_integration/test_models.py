from django.test import TestCase
import numpy as np
from unittest import mock
import os
import sys
import cv2 as cv
from azure.storage.blob import BlockBlobService
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from common.segmentation import Foram
import common.segmentation as seg
from upload.models import Img, ImgParent, Species
from azure.common import AzureMissingResourceHttpError

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class TestForamClass(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.blob_service = BlockBlobService(os.environ['AZ_STORAGE_ACCOUNT_NAME'], os.environ['AZ_STORAGE_KEY'])
        cls.dirpath = os.path.join(BASE_DIR, 'fixtures')
        cls.foram = Foram(None, cls.dirpath)
        cls.foram.parent_img = cv.imread(os.path.join(BASE_DIR, 'fixtures/parent.jpg'))
        cls.foram.boxes = seg.filter_boxes(seg.get_boxes(cls.foram.parent_img, 100))
        cls.foram.forams = [cls.foram.parent_img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] for box in cls.foram.boxes]
        cls.foram.store_parents()
        cls.foram.set_species()
        cls.foram.store_children()

    def test_store_parents(self):
        self.assertEqual(len(ImgParent.objects.all()), 1)

    def test_store_species(self):
        self.assertEqual((len(self.foram.species_obj)), len(self.foram.forams))
    
    def test_store_children(self):
        child_img = Img.objects.all()
        self.assertEqual(len(child_img), len(self.foram.forams))
        self.assertEqual(set([img.species for img in child_img]), set(Species.objects.all()))
        self.assertEqual(set([img.parentImage for img in child_img]), set(ImgParent.objects.all()))
        self.assertEqual([img.number_on_image for img in child_img], [i for i in range(len(self.foram.forams))])

    def test_delete_foram(self):
        num_of_img = len(Img.objects.all())
        img = Img.objects.all()[0]
        self.foram.delete_foram(img.id)
        new_num_of_img = len(Img.objects.all())
        self.assertEqual(num_of_img, new_num_of_img+1)
        with self.assertRaises(AzureMissingResourceHttpError):
            self.blob_service.get_blob_properties(self.foram.container,
                                                              img.imgLocation.name)
        # This line is needed due to how the django.testcase works
        self.blob_service.create_blob_from_path(
            self.foram.container,
            img.imgLocation.name,
            os.path.join(self.dirpath, os.path.basename(img.imgLocation.name))
        )

    @classmethod
    def tearDownClass(cls):
        for files in os.listdir(cls.dirpath):
            if len(files) > 10:
                os.remove(os.path.join(cls.dirpath, files))
        all_img = []
        all_img += [img.imgLocation.name for img in Img.objects.all()]
        all_img += [parent.imgLocation.name for parent in ImgParent.objects.all()]
        all_img += [parent.imgEdited.name for parent in ImgParent.objects.all()]
        for name in all_img:
            cls.foram.block_blob_service.delete_blob(cls.foram.container, name)