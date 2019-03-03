from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from PIL import Image
import numpy as np
import cv2
from glob import glob
from plyfile import PlyData, PlyElement
import errno
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg
import numpy as np
from plyfile import PlyData, PlyElement
from mpl_toolkits.mplot3d import Axes3D
import pptk
from argparse import ArgumentParser
from torch.utils import data

class PanopticDataset(data.Dataset):
    """
        Load panoptic dataset.

        Attributes:
            img_dir_path: path of kinetic images
            rgb_img_dir_path: path of rgb images
            depth_img_dir_path: path of depth images for the rgb sensor
            bbox_dir_path: path of bounding boxes for the rgb images obtained from mask-rcnn
            ptclouds_dir_path: path of point cloud for kinect images
            hd_projections_dir: projection of point cloud in HD space in 2D
            corresponding_3d_pts_dir: 3D points corresponding to HD projection
            kinect_camera: kinect camera number
            hd_camera_list: a list of HD camera numbers
            start_frame: start frame to load
            end_frame: end frame to load
            ret: a dictionary containing kinect image, rgb image, depth, masks (confidence for depth),
                 bounding boxes, depth cropped by bounding boxes, point clouds and hd projection

    """
    def __init__(self, img_dir_path, depth_img_dir_path, rgb_img_dir_path, bbox_dir_path, ptclouds_dir_path, hd_projections_dir, corresponding_3d_pts_dir, kinect_camera, hd_camera_list, start_frame, end_frame):
        self.img_dir_path = img_dir_path
        self.rgb_img_dir_path = rgb_img_dir_path
        self.depth_img_dir_path = depth_img_dir_path
        self.bbox_dir_path = bbox_dir_path
        self.ptclouds_dir_path = ptclouds_dir_path
        self.kinect_camera = kinect_camera
        self.hd_camera_list = hd_camera_list
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.hd_projections_dir = hd_projections_dir
        self.corresponding_3d_pts_dir = corresponding_3d_pts_dir
        self.ret = {}
        self.__get_output__()


    def __get_output__(self):
        imgs = []
        rgb_imgs = []
        depth_imgs = []
        cropped_depth_imgs = []
        ptclouds = []
        projections_list = []
        masks = []
        self.ret['bbox'] =self.__load_bbox__()
        for index in range(self.start_frame, self.end_frame+1):
            img = self.__read_img__(index)
            imgs.append(img)
            rgb_img = self.__read_rgb_img__(index)
            rgb_imgs.append(rgb_img)
            depth_img,cropped_depth_img, mask = self.__read_depth_img__(index)
            depth_imgs.append(depth_img)
            masks.append(mask)
            cropped_depth_imgs.append(cropped_depth_img)
            ptcloud = self.__read_ptcloud__(index)
            ptclouds.append(ptcloud)
            camera_dict = self.__read_projection__(index)
            projections_list.append(camera_dict)
        self.ret['img'] = imgs
        self.ret['rgb_img'] = rgb_imgs
        self.ret['depth'] = depth_imgs
        self.ret['masks'] = masks
        self.ret['cropped_depth'] = cropped_depth_imgs
        self.ret['ptcloud']= ptclouds
        self.ret['projections'] = projections_list
        return self.ret

    def __getitem__(self, index):
        # only masks and ground truth depth are used in the loss layer
        return self.ret['masks'][index], self.ret['depth'][index]


    def __len__(self):
        return len(self.ret['img'])


    def __read_projection__(self, index):
        camera_dict = {}
        for hd_camera in self.hd_camera_list:
            projections_hd_2d_path = self.hd_projections_dir + hd_camera + '/' + 'projections_2d_hd%08d.mat' %index
            if not os.path.exists(projections_hd_2d_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), projections_hd_2d_path)
            projections_hd_2d = sio.loadmat(projections_hd_2d_path)
            projections_hd_3d_path = self.corresponding_3d_pts_dir + hd_camera + '/' + 'projections_3d_hd%08d.mat' %index
            if not os.path.exists(projections_hd_3d_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), projections_hd_3d_path)
            projections_hd_3d = sio.loadmat(projections_hd_3d_path)
            camera_dict['2D_%s' %hd_camera] = projections_hd_2d
            camera_dict['3D_%s' %hd_camera] = projections_hd_3d
        return camera_dict


    def __read_img__(self, index):
        img_path = self.img_dir_path + self.kinect_camera + '/' + self.kinect_camera + '_%08d.jpg' %index
        if not os.path.exists(img_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), img_path)
        img = cv2.imread(img_path)
        return img


    def __read_depth_img__(self, index):
        img_path = self.depth_img_dir_path  + '/' +'depth%08d.jpg' %index
        confidence_path = self.depth_img_dir_path  + '/' +'confidence%08d.jpg' %index
        if not os.path.exists(img_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), img_path)
        img = cv2.imread(img_path, 0)
        confidence = cv2.imread(confidence_path, 0)
        cropped_img = self.__crop_depth_img__(img, index)
        return img, cropped_img, confidence


    def __read_rgb_img__(self, index):
        img_path = self.rgb_img_dir_path  + '/' +'rgb%08d.jpg' %index
        if not os.path.exists(img_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), img_path)
        img = cv2.imread(img_path, 0)
        return img


    def __crop_depth_img__(self, img, index):
        bbox = self.ret['bbox'][index - self.start_frame]
        row_start = int(bbox[1])
        row_end = int(bbox[3])
        col_start = int(bbox[0])
        col_end = int(bbox[2])
        cropped_depth = img[row_start:row_end, col_start:col_end]
        return cropped_depth


    def __load_bbox__(self):
        if self.bbox_dir_path is not None:
            bbox = np.loadtxt(self.bbox_dir_path)
            bbox = bbox[:, 2:7]
            return bbox
        else: #get image dimension
            img = self.__read_img__(self.start_frame)
            num_images = self.end_frame - self.start_frame + 1
            box = [0, 0, np.shape(img)[1], np.shape(img)[0]]
            return np.tile(box, (num_images, 1))


    def __read_ptcloud__(self, index):
        ptcloud_path = self.ptclouds_dir_path + 'ptcloud_hd%08d.ply' %index
        if not os.path.exists(ptcloud_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), ptcloud_path)
        ptcloud = PlyData.read(ptcloud_path)
        return ptcloud

def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--img_dir_path', type=str, default='/home/wanyue/Desktop/panoptic-toolbox/161029_flute1/kinectImgs/')
    parser.add_argument('-kcam', '--kinect_camera', type=str, default='50_01')
    parser.add_argument('-hdcam', '--hd_camera_list', nargs='+', type=str, default={'00_16', '00_30'}) # might be wrong here
    parser.add_argument('-ptcloud', '--ptclouds_dir_path', type=str, default='/home/wanyue/Desktop/panoptic-toolbox/161029_flute1/kinoptic_ptclouds/')
    parser.add_argument('-hd_proj', '--hd_projections_dir', type=str, default='/home/wanyue/Desktop/panoptic-toolbox/161029_flute1/hd_projections/')
    parser.add_argument('-c_pts', '--corresponding_3d_pts_dir', type=str, default='/home/wanyue/Desktop/panoptic-toolbox/161029_flute1/hd_projections_3D/')
    parser.add_argument('-depth', '--depth_img_dir_path', type=str, default='/home/wanyue/Desktop/panoptic-toolbox/161029_flute1/kinoptic_depth_rgb')
    parser.add_argument('-rgb', '--rgb_img_dir_path', type=str, default='/home/wanyue/Desktop/panoptic-toolbox/161029_flute1/kinoptic_rgb_rgb')
    parser.add_argument('-bbox', '--bbox_dir_path', type=str, default='/home/wanyue/github/my_maskrcnn/demo/camera_50_01.txt')
    parser.add_argument('-s', '--start_frame', type=int, default=500)
    parser.add_argument('-e', '--end_frame', type=int, default=510)
    args = parser.parse_args()
    # assume one track first
    my_data_loader = PanopticDataset(args.img_dir_path, args.depth_img_dir_path, args.rgb_img_dir_path, None, args.ptclouds_dir_path, args.hd_projections_dir, args.corresponding_3d_pts_dir, args.kinect_camera, args.hd_camera_list, args.start_frame, args.end_frame)
    # get all output
    output = my_data_loader.__get_output__()

if __name__ == '__main__':
    main()
