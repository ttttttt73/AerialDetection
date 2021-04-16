from .coco import CocoDataset
import numpy as np


class ROOFDataset_r_h(CocoDataset):
    # dota1_RoI_roof 꺼
    # CLASSES = ('facility', 'rooftop', 'flatroof', 'solarpanel_flat', 'solarpanel_slope', 'parkinglot', 'heliport_r', 'heliport_h')
    
    # augment
    CLASSES = ('facility', 'flatroof', 'rooftop', 'parkinglot', 'solarpanel_flat', 'solarpanel_slope', 'heliport_r', 'heliport_h')

