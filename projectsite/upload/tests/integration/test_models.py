from django.test import TestCase
import numpy as np
from unittest import mock
import os
import sys
import cv2 as cv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from common.segmentation import Foram
import common.segmentation as seg
from upload.models import Img, ImgParent, Species


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# class ImgModelTest(TestCase):
#     '''
#     This test ensures that images are being saved correctly in the database
#     '''
#     @classmethod
#     def setUpTestData(cls):
#         img = np.zeros((3,3,3), np.uint8)
#         forams = np.zeros((3,3,3,3), np.uint8)
#         with mock.patch('common.segmentation.cv.imwrite') as mocked_cv, \
#             mock.patch('common.segmentation.os.path.isfile') as mock_is_file, \
#             mock.patch('common.segmentation.os.listdir') as mock_listdir, \
#             mock.patch('common.segmentation.os.mkdir') as mock_dir:
#             mock_is_file.return_value = True
#             mock_listdir.return_value = ['file1', 'file2']
#             mock_dir = True
#             common.segmentation.store_to_db(img, forams, 'G. ruber', './random', '.jpg')

#     def test_correct_number_of_objects(self):
#         self.assertEqual(len(Img.objects.all()), 3)
#         self.assertEqual(len(ImgParent.objects.all()), 1)
    
#     def test_parent_url(self):
#         img = [i for i in ImgParent.objects.all()]
#         self.assertEqual(os.path.dirname(img[0].imgLocation.url),
#                          '/media/random/parent')

#     def test_child_location_is_same(self):
#         '''
#         url = '/media/random/G.%20ruber/f1a85a4299e5426687396b5d2b5a10ef.jpg'
#         location = <ImageFieldFile: ./random/G. ruber/f1a85a4299e5426687396b5d2b5a10ef.jpg
#         '''
#         img = [i for i in Img.objects.all()]
#         url = [os.path.dirname(str(i.imgLocation)) for i in img]
#         self.assertEqual(url, [url[0] for i in range(len(url))])


# class SpeciesTest(TestCase):
#     '''
#     Ensures that the species table is functioning correctly
#     '''
#     @classmethod
#     def setUpTestData(cls):
#         img = np.zeros((3,3,3), np.uint8)
#         forams = np.zeros((3,3,3,3), np.uint8)
#         with mock.patch('common.segmentation.cv.imwrite') as mocked_cv, \
#             mock.patch('common.segmentation.os.path.isfile') as mock_is_file, \
#             mock.patch('common.segmentation.os.listdir') as mock_listdir, \
#             mock.patch('common.segmentation.os.mkdir') as mock_dir:
#             mock_is_file.return_value = True
#             mock_listdir.return_value = ['file1', 'file2']
#             mock_dir = True
#             common.segmentation.store_to_db(img, forams, 'G. ruber', './random', '.jpg')
#             common.segmentation.store_to_db(img, forams, 'G. scitula', './random', '.jpg')
#             common.segmentation.store_to_db(img, forams, 'N. humerosa', './random', '.jpg')
#             common.segmentation.store_to_db(img, forams, 'G. ruber', './random', '.jpg')

#     def test_number_of_one_species(self):
#         self.assertEqual(6, Species.objects.get(pk='G. ruber').total)

#     def test_distinct_number_of_species(self):
#         self.assertEqual(3, len(Species.objects.values_list('name', flat=True).distinct()))

#     def test_distinct_directories_exist(self):
#         dir1 = [img for img in Img.objects.all() if os.path.basename(
#             os.path.dirname(str(img.imgLocation))) == 'G. ruber']
#         dir2 = [img for img in Img.objects.all() if os.path.basename(
#             os.path.dirname(str(img.imgLocation))) == 'G. scitula']
#         dir3 = [img for img in Img.objects.all() if os.path.basename(
#             os.path.dirname(str(img.imgLocation))) == 'N. humerosa']
#         self.assertNotEqual(len(dir1), len(dir2))
#         self.assertEqual(len(dir2), len(dir3))


# class ImgTest(TestCase):
#     @classmethod
#     def setUpTestData(cls):
#         Img.objects.create(imgLocation='ff'
#                            species=
#                            parentImage=
#                            number_on_image=)

class FakeBlockBlobService(object):
    def __init__(self):
        pass

    def create_blob_from_path(self, *args):
        pass
    
    def delete_blob(self, *args):
        pass
    
    def copy_blob(self, *args):
        pass

class FakeUploadedFile(object):
    pass


class TestForamClass(TestCase):
    @classmethod
    def setUpTestData(cls):
        cls.foram = Foram(None, os.path.join(BASE_DIR, 'fixtures'))
        cls.foram.parent_img = cv.imread(os.path.join(BASE_DIR, 'fixtures/parent.jpg'))
        cls.foram.boxes = seg.filter_boxes(seg.get_boxes(cls.foram.parent_img, 100))
        cls.foram.forams = [cls.foram.parent_img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] for box in cls.foram.boxes]
        cls.foram.store_parents()
        cls.foram.set_species()
        cls.foram.store_children()

    def test_store_parents(self):
        self.assertEquals(len(ImgParent.objects.all()) == 1)

    def test_store_species(self):
        self.assertEquals((len(self.foram.species_obj)) == len(self.foram.forams))
    
    def test_store_children(self):
        child_img = Img.objects.all()
        self.assertEquals(len(child_img) == len(self.foram.forams))
        self.assertEquals(set([img.species for img in child_img]) == {[Species.objects.all()[0].name]})
        self.assertEquals(set([img.parentImage for img in child_img]) == {[ImgParent.objects.all()[0]]})
        self.assertEquals([img.number_on_image for img in child_img] == [i for i in range(len(self.foram.forams))])

    def remove_blobs(self):
        all_img = []
        all_img += [img.imgLocation.name for img in Img.objects.all()]
        all_img += [parent.imgLocation.name for parent in ImgParent.objects.all()]
        all_img += [parent.imgEdited.name for parent in ImgParent.objects.all()]
        for name in all_img:
            self.foram.block_blob_service.delete_blob(self.foram.container, name)
    
        
