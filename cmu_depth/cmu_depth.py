from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import pickle as pk
import os
import random
import yaml
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.transforms import fliplr_joints,affine_transform,gen_trans_from_patch_cv
from dataset.ArielDataset import ArielDataset
logger = logging.getLogger(__name__)

class panoptic_db():
    def __init__(self, annotation_file=''):
        with open(annotation_file, 'rb') as file:
            self.data = json.load(file)
        anns = []
        for ann in self.data['annotations']:
            anns.append(ann['id'])
        imgs = []
        for im in self.data['images']:
            imgs.append(im['id'])
        self.anns = anns
        self.imgs = imgs
        ## Load cams
        for cam in self.data['cameras']:
            self.data['cameras'][cam]['K'] = np.matrix(np.array(self.data['cameras'][cam]['K']).reshape(3,3) )
            self.data['cameras'][cam]['distCoef'] = np.array(self.data['cameras'][cam]['distCoef'])
            self.data['cameras'][cam]['R'] = np.matrix(np.array(self.data['cameras'][cam]['R']).reshape(3,3) )
            self.data['cameras'][cam]['t'] = np.array(self.data['cameras'][cam]['t']).reshape(3,1)


    def get_img(self,im_id):
        return self.data['images'][self.imgs.index(im_id)]


    def get_ann(self,ann_id):
        return self.data['annotations'][self.anns.index(ann_id)]


class CMUDepthDataset(ArielDataset): #change a name
    def __init__(self, cfg, dataset, is_train, transform=None, dotiny=False):
        super().__init__(cfg, dataset, is_train, transform)
        self.dataset = dataset
        self.p_loader = panoptic_db(dataset.ANNOTATION_FILE)
        self.db       = self._get_db()
        if dotiny:
            tiny_inds = np.random.randint(0,len(self.db),300)
            self.db = [ self.db[i] for i in tiny_inds]
        logger.info('=> load {} samples'.format(len(self.db)))


    def _get_db(self):
        db = []
        for i, current_img in enumerate(self.p_loader.data['images']):
            selected_im = self.p_loader.get_img(current_img['id'])
            bbox = np.array(self.p_loader.get_ann(current_img['id'])['bbox'])
            cam = self.p_loader.data['cameras'][ selected_im['cam'] ]
            bbox = self.p_loader.data['annotations']
            db.append({
                'id': i,
                'image': self.dataset.IMAGE_ROOT + 'kinoptic_rgb/' + selected_im['image_name'],
                'depth': self.dataset.IMAGE_ROOT + 'kinoptic_depth/' +selected_im['image_name'],
                'confidence': self.dataset.IMAGE_ROOT + 'kinoptic_confidence/' + selected_im['image_name'],
                'bbox': bbox,
                'cam_name': selected_im['cam']
            })
        return db


    def __len__(self,):
        return len(self.db)
