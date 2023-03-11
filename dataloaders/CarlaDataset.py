########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import glob
import os
import random
from torchvision import transforms
from dataloaders.laserscan import LaserScan

class CarlaDataset(Dataset):

    def __init__(self, kitti_depth_path, setname='train', transform=None, norm_factor=256, invert_depth=False,
                 load_rgb=False, kitti_rgb_path=None, rgb2gray=False, hflip=False, sensor='kitti'):

        self.kitti_depth_path = kitti_depth_path # home/user/Documents/carla_data/DataGenerator/data/test
        self.setname = setname # 'train'
        self.transform = transform # CenterCrop(size=(352, 1216))
        self.norm_factor = norm_factor # 256
        self.invert_depth = invert_depth # False
        self.load_rgb = load_rgb # False
        self.kitti_rgb_path = kitti_rgb_path # Not really used
        self.rgb2gray = rgb2gray # True, but probably not really used
        self.hflip = hflip # False
        self.sensor = sensor # 'kitti' for velodyne with 64 channels

        if setname in ['train', 'val']:
            depth_path = os.path.join(self.kitti_depth_path, setname)
            self.depth = np.array(sorted(glob.glob(os.path.join(depth_path, sensor+'_velodyne') + '/*.bin', recursive=True)))
            self.gt = np.array(sorted(glob.glob(os.path.join(depth_path, 'velodyne') + '/*.bin', recursive=True)))
        elif setname == 'selval': # richard: not yet modified
            depth_path = os.path.join(self.kitti_depth_path, 'depth_selection', 'val_selection_cropped')
            self.depth = np.array(sorted(glob.glob(depth_path + "/velodyne_raw/*.png", recursive=True)))
            self.gt = np.array(sorted(glob.glob(depth_path + "/groundtruth_depth/*.png", recursive=True)))
        elif setname == 'test': # richard: not yet modified
            depth_path = os.path.join(self.kitti_depth_path, 'depth_selection', 'test_depth_completion_anonymous')
            self.depth = np.array(sorted(glob.glob(depth_path + "/velodyne_raw/*.png", recursive=True)))
            self.gt = np.array(sorted(glob.glob(depth_path + "/velodyne_raw/*.png", recursive=True)))

        assert(len(self.gt) == len(self.depth))

        self.sensor_fov_up = 10.0
        self.sensor_fov_down = -30.0
        self.max_points = 150000

        if self.sensor == 'kitti':
            self.sensor_img_H = 64
            self.sensor_img_W = 2048
            self.sensor_img_means = torch.Tensor([0,0,0,0,0]) # richard update
            self.sensor_img_stds = torch.Tensor([1,1,1,1,1]) # range, x, y, z, signal
        else:
            self.sensor_img_H = 64
            self.sensor_img_W = 2048
            self.sensor_img_means = torch.Tensor([0,0,0,0,0]) # richard update
            self.sensor_img_stds = torch.Tensor([1,1,1,1,1]) # range, x, y, z, signal
        self.gt_sensor_img_H = 128
        self.gt_sensor_img_W = 4096
        self.gt_sensor_fov_up = 10.0
        self.gt_sensor_fov_down = -30.0
        self.gt_max_points = 4000000
        self.gt_sensor_img_means = torch.Tensor([0,0,0,0,0]) # richard update
        self.gt_sensor_img_stds = torch.Tensor([1,1,1,1,1]) # range, x, y, z, signal

        # img_means: #range,x,y,z,signal
        #   - 12.12
        #   - 10.88
        #   - 0.23
        #   - -1.04
        #   - 0.21
        # img_stds: #range,x,y,z,signal
        #   - 12.32
        #   - 11.47
        #   - 6.91
        #   - 0.86
        #   - 0.16

    def __len__(self):
        return len(self.depth)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        DA = False
        flip_sign = False
        rot = False
        drop_points = False
        # richard looks into this
        # if self.transform:
        #     if random.random() > 0.5:
        #         if random.random() > 0.5:
        #             DA = True
        #         if random.random() > 0.5:
        #             flip_sign = True
        #         if random.random() > 0.5:
        #             rot = True
        #         drop_points = random.uniform(0, 0.5)

        depth_path = self.depth[item]
        depth_scan = LaserScan(project=True,
                         H=self.sensor_img_H,
                         W=self.sensor_img_W,
                         fov_up=self.sensor_fov_up,
                         fov_down=self.sensor_fov_down,
                         DA=DA,
                         rot=rot,
                         flip_sign=flip_sign,
                         drop_points=drop_points)

        # open and obtain scan
        depth_scan.open_scan(depth_path)
        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points = depth_scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(depth_scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(depth_scan.unproj_range)
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(depth_scan.remissions)

        # get points and labels
        proj_range = torch.from_numpy(depth_scan.proj_range).clone()
        proj_xyz = torch.from_numpy(depth_scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(depth_scan.proj_remission).clone()
        proj_mask = torch.from_numpy(depth_scan.proj_mask)
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(depth_scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(depth_scan.proj_y)
        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                          proj_xyz.clone().permute(2, 0, 1),
                          proj_remission.unsqueeze(0).clone()])
        proj = (proj - self.sensor_img_means[:, None, None]
                ) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()

        # get name and sequence
        path_norm = os.path.normpath(depth_path)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")
        input = dict()
        input['proj'] = proj
        input['proj_mask'] = proj_mask
        input['path_seq'] = path_seq
        input['path_name'] = path_name
        input['proj_x'] = proj_x
        input['proj_y'] = proj_y
        input['proj_range'] = proj_range
        input['unproj_range'] = unproj_range
        input['proj_xyz'] = proj_xyz
        input['unproj_xyz'] = unproj_xyz
        input['proj_remission'] = proj_remission
        input['unproj_remissions'] = unproj_remissions
        input['unproj_n_points'] = unproj_n_points

        gt_path = self.gt[item]
        gt_scan = LaserScan(project=True,
                         H=self.gt_sensor_img_H,
                         W=self.gt_sensor_img_W,
                         fov_up=self.gt_sensor_fov_up,
                         fov_down=self.gt_sensor_fov_down,
                         DA=False,
                         rot=False,
                         flip_sign=False,
                         drop_points=False)

        # open and obtain scan
        gt_scan.open_scan(gt_path)
        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points = gt_scan.points.shape[0]
        unproj_xyz = torch.full((self.gt_max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(gt_scan.points)
        unproj_range = torch.full([self.gt_max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(gt_scan.unproj_range)
        unproj_remissions = torch.full([self.gt_max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(gt_scan.remissions)

        # get points and labels
        proj_range = torch.from_numpy(gt_scan.proj_range).clone()
        proj_xyz = torch.from_numpy(gt_scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(gt_scan.proj_remission).clone()
        proj_mask = torch.from_numpy(gt_scan.proj_mask)
        proj_x = torch.full([self.gt_max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(gt_scan.proj_x)
        proj_y = torch.full([self.gt_max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(gt_scan.proj_y)
        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                          proj_xyz.clone().permute(2, 0, 1),
                          proj_remission.unsqueeze(0).clone()])
        proj = (proj - self.gt_sensor_img_means[:, None, None]
                ) / self.gt_sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()

        # get name and sequence
        path_norm = os.path.normpath(gt_path)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")
        gt = dict()
        gt['proj'] = proj
        gt['proj_mask'] = proj_mask
        gt['path_seq'] = path_seq
        gt['path_name'] = path_name
        gt['proj_x'] = proj_x
        gt['proj_y'] = proj_y
        gt['proj_range'] = proj_range
        gt['unproj_range'] = unproj_range
        gt['proj_xyz'] = proj_xyz
        gt['unproj_xyz'] = unproj_xyz
        gt['proj_remission'] = proj_remission
        gt['unproj_remissions'] = unproj_remissions
        gt['unproj_n_points'] = unproj_n_points

        # # Read depth input and gt
        # depth = Image.open(self.depth[item])
        # gt = Image.open(self.gt[item])
        #
        # # Read RGB images
        # if self.load_rgb:
        #     if self.setname in ['train', 'val']:
        #         gt_path = self.gt[item]
        #         idx = gt_path.find('2011')
        #         seq_name = gt_path[idx:idx+26]
        #         idx2 = gt_path.find('groundtruth')
        #         camera_name = gt_path[idx2+12:idx2+20]
        #         fname = gt_path[idx2+21:]
        #         rgb_path = os.path.join(self.kitti_rgb_path, self.setname, seq_name, camera_name, 'data', fname)
        #         rgb = Image.open(rgb_path)
        #     elif self.setname == 'selval':
        #         depth_path = self.depth[item]
        #         tmp = depth_path.split('velodyne_raw')
        #         rgb_path = tmp[0] + 'image' + tmp[1] + 'image' + tmp[2]
        #         rgb = Image.open(rgb_path)
        #     elif self.setname == 'test':
        #         depth_path = self.depth[item]
        #         tmp = depth_path.split('velodyne_raw')
        #         rgb_path = tmp[0] + 'image' + tmp[1]
        #         rgb = Image.open(rgb_path)
        #
        #     if self.rgb2gray:
        #         t = transforms.Grayscale(1)
        #         rgb = t(rgb)
        #
        # # Apply transformations if given
        # if self.transform is not None:
        #     depth = self.transform(depth)
        #     gt = self.transform(gt)
        #     if self.load_rgb:
        #         rgb = self.transform(rgb)
        #
        # flip_prob = np.random.uniform(0.0, 1.0) > 0.5
        # if self.hflip and flip_prob:
        #     depth, gt = transforms.functional.hflip(depth),  transforms.functional.hflip(gt)
        #     if self.load_rgb:
        #         rgb = transforms.functional.hflip(rgb)
        #
        # # Convert to numpy
        # depth = np.array(depth, dtype=np.float32)
        # gt = np.array(gt, dtype=np.float32)
        #
        # # Normalize the depth
        # depth = depth / self.norm_factor  #[0,1]
        # gt = gt / self.norm_factor
        #
        # # Expand dims into Pytorch format
        # depth = np.expand_dims(depth, 0)
        # gt = np.expand_dims(gt, 0)
        #
        # # Convert to Pytorch Tensors
        # depth = torch.from_numpy(depth)  #    (depth, dtype=torch.float)
        # gt = torch.from_numpy(gt)  #tensor(gt, dtype=torch.float)
        #
        # # Convert depth to disparity
        # if self.invert_depth:
        #     depth[depth==0] = -1
        #     depth = 1 / depth
        #     depth[depth==-1] = 0
        #
        #     gt[gt==0] = -1
        #     gt = 1 / gt
        #     gt[gt==-1] = 0
        #
        # # Convert RGB image to tensor
        # if self.load_rgb:
        #     rgb = np.array(rgb, dtype=np.float32)
        #     rgb /= 255
        #     if self.rgb2gray: rgb = np.expand_dims(rgb,0)
        #     else : rgb = np.transpose(rgb,(2,0,1))
        #     rgb = torch.from_numpy(rgb)
        #     input = torch.cat((rgb, depth), 0)
        # else:
        #     input = depth

        return input, gt
