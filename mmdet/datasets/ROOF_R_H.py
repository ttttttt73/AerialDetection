from .coco import CocoDataset
import numpy as np


class ROOFDataset_r_h(CocoDataset):

    CLASSES = ('facility', 'rooftop', 
                'flatroof', 'solarpanel_flat', 
                'solarpanel_slope', 'parkinglot', 
                'heliport_r', 'heliport_h')

