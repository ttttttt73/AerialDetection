from .coco import CocoDataset
import numpy as np


class ROOFDataset(CocoDataset):

    CLASSES = ('facility', 'rooftop', 
                'flatroof', 'solarpanel_flat', 
                'solarpanel_slope', 'parkinglot', 
                'heliport')

