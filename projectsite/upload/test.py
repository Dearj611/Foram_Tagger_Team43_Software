from django.test import TestCase
from .models import Img, ImgParent
import common.segmentation
import numpy as np
from unittest import mock


class ImgModelTest(TestCase):
    @classmethod
    def setUpTestData(cls):
        img = np.zeros((3,3,3), np.uint8)
        forams = np.zeros((3,3,3,3), np.uint8)
        with mock.patch('common.segmentation.cv.imwrite') as mocked_cv, \
            mock.patch('common.segmentation.os.path.isfile') as mock_is_file, \
            mock.patch('common.segmentation.os.listdir') as mock_listdir:
            mock_is_file.return_value = True
            mock_listdir.return_value = ['file1', 'file2']
            common.segmentation.store_to_db(img, forams, 'G. ruber', './random', '.jpg')

    def test_correct_img_locations(self):
        self.assertEqual(len(Img.objects.all()), 3)
        self.assertEqual(len(ImgParent.objects.all()), 1)
        imgs = [i for i in ImgParent.objects.all()] + \
               [i for i in Img.objects.all()]
        correct_url = ['/media/random/2.jpg', '/media/random/3.jpg',
                       '/media/random/4.jpg', '/media/random/5.jpg']
        correct_location = ['./random/2.jpg', './random/3.jpg',
                            './random/4.jpg', './random/5.jpg']
        # arr_location = [i.imgLocation for i in imgs]
        # arr_url = [i.imgLocation.url for i in imgs]
        # print(arr_location)
        # print(arr_url)
        self.assertEqual([i.imgLocation.url for i in imgs], correct_url)
        self.assertEqual([i.imgLocation for i in imgs], correct_location)
            
            
            
        