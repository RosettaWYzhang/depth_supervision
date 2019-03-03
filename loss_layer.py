"""
A pytorch layer to render depth
"""
import os
import argparse
import torch
import torchvision
import torch.nn as nn
import numpy as np
import tqdm
import imageio
import scipy.io as sio
import json
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
from PIL import Image
import glob
from skimage.transform import rescale, resize, downscale_local_mean
from tensorboardX import SummaryWriter
import pdb
from data_loader import PanopticDataset
import neural_renderer as nr
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')


class LossLayer(nn.Module):
    """
        Pytorch layer to calculate loss for rendered depth.

        Args:
            image_size: size of rendered image.

        Attributes:
            renderer: neural renderer in depth mode

    """
    def __init__(self, image_size=256):
        super(LossLayer, self).__init__()
        renderer = nr.Renderer(camera_mode='look_at', image_size=image_size, anti_aliasing=False,perspective = False)
        renderer.eye = nr.get_points_from_angles(2,0,0)
        self.renderer = renderer

    def forward(self, vertices, faces, confidence, depth_ref):
        '''call renderer to predict depth and calculate loss

        Args:
            vertices: vertices from smlp model
            faces: faces from smlp model
            confidence: how much to trust the ground truth depth
            depth_ref: ground truth for depth

        Returns:
            depth_pred: depth predicted by the renderer from 3D mesh
            loss: l2 loss
        '''
        depth_pred = self.renderer(vertices, faces, mode='depth')
        depth_pred[depth_pred == 100] = 0 # ignore background value
        ignore = depth_pred <= 1e-6 # prevent division by zero
        depth_pred = 1./depth_pred
        depth_pred[ignore] = 0
        depth_pred = depth_pred / torch.max(depth_pred)
        loss = torch.sum((depth_pred * confidence - depth_ref * confidence) ** 2)
        return depth_pred, loss


class VertexPredictor(nn.Module):
    """
        Dummy layer to return predicted vertex.

        Args:
            filename_obj: path of .obj file containing vertices and faces.

        Attributes:
            vertices: obtained directly from obj file to be backpropagated

    """
    def __init__(self, filename_obj, batch_size=10, image_size=256):
        super(VertexPredictor, self).__init__()
        vertices, faces = nr.load_obj(filename_obj)
        vertices[:,1]  = - vertices[:,1]
        vertices = vertices[None, :, :]
        vertices = np.tile(vertices, (batch_size,1,1))
        faces = faces[None, :, :]
        faces = np.tile(faces, (batch_size,1,1))
        self.vertices = nn.Parameter(torch.from_numpy(vertices))
        self.register_buffer('faces', torch.from_numpy(faces))

    def forward(self, confidence, depth_ref):
        # dummy function
        return self.vertices, self.faces


def process_minibatches(confidence, depth_ref, image_size=256):
    '''crop and resize confidence and depth_ref in batches'''
    confidence_output = list(map(process_confidence, torch.unbind(confidence, 0)))
    confidence_output = torch.stack(confidence_output, 0).float()
    depth_output = list(map(process_depth_ref, torch.unbind(depth_ref, 0)))
    depth_output = torch.stack(depth_output, 0).float()
    return confidence_output, depth_output


def process_confidence(confidence, image_size=256, threshold=0.8):
    '''crop confidence using bounding boxes (dummy for now), and resize it'''
    confidence = confidence.data.numpy()
    confidence = confidence.astype(np.float32)
    confidence = confidence[200:1037, 750:1168] # select a random bounding box for now
    imrsz = resize(confidence/255,[image_size,image_size],order=0 ,anti_aliasing=True)
    imrsz  = torch.from_numpy((imrsz > threshold).astype(np.float32))
    return imrsz.cuda()


def process_depth_ref(image_ref, image_size=256):
    '''crop depth using bounding boxes (dummy for now), and resize it'''
    image_ref = image_ref.data.numpy()
    image_ref = image_ref.astype(np.float32)
    image_ref = torch.from_numpy(image_ref)
    image_ref = image_ref[200:1037, 750:1168] # select a random bounding box for now
    image_ref = torchvision.transforms.ToPILImage()(image_ref[None, :, :])
    image_ref = torchvision.transforms.Resize((image_size,image_size))(image_ref)
    image_ref = torchvision.transforms.ToTensor()(image_ref)[0].cuda()
    return image_ref


def save_training_image(depth_pred, depth_ref, confidence, count, directory, image_size=256):
    '''concatenate and save predicted depth, ground truth and confidence during training'''
    result_vis = np.zeros((image_size, image_size * 3));
    result_vis[:,0:image_size] = depth_pred.detach().cpu().numpy()[0]
    result_vis[:,image_size:image_size*2] = depth_ref.detach().cpu().numpy()[0]
    result_vis[:,image_size*2:image_size*3] = confidence.detach().cpu().numpy()[0]
    imsave(directory + '_tmp_%04d.png' % count, result_vis)


def train_model(train_loader, filename_obj, num_epochs, batch_size, vis_directory):
    '''backpropagate loss and update model parameter vertices'''
    vertex_predictor = VertexPredictor(filename_obj, batch_size=batch_size)
    vertex_predictor.cuda()
    optimizer = torch.optim.Adam(vertex_predictor.parameters(), lr=0.001)
    loss_layer = LossLayer()
    loss_layer.cuda()
    writer = SummaryWriter()
    for i in range(num_epochs):
        for confidence, depth_ref in train_loader:
            confidence, depth_ref  = process_minibatches(confidence, depth_ref)
            vertices_predicted, faces = vertex_predictor(confidence, depth_ref)
            depth_pred, loss = loss_layer(vertices_predicted, faces, confidence, depth_ref)
            if i%50 == 0: # save intermediate result at every 50 epochs
                save_training_image(depth_pred, depth_ref, confidence, i, vis_directory)
            print(i, loss.item())
            writer.add_scalar('data/loss', loss, i)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-vis', '--vis_directory', type=str, default='/home/wanyue/Desktop/neural_renderer/examples/vis/')
    parser.add_argument('-obj', '--filename_obj', type=str, default=os.path.join(data_dir, 'camera_16_mesh_01.obj'))
    # arguments needed by DataLoader
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

    # working example
    panoptic_dataset = PanopticDataset(args.img_dir_path, args.depth_img_dir_path, args.rgb_img_dir_path, None, args.ptclouds_dir_path, args.hd_projections_dir, args.corresponding_3d_pts_dir, args.kinect_camera, args.hd_camera_list, args.start_frame, args.end_frame)
    num_epochs = 500
    batch_size = 2
    train_loader = torch.utils.data.DataLoader(panoptic_dataset, batch_size=batch_size)
    train_model(train_loader, args.filename_obj, num_epochs, batch_size, args.vis_directory)



if __name__ == '__main__':
    main()
