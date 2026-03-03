import numpy as np
import random
import warnings
import os
import time
from glob import glob
from skimage import color, io
from PIL import Image, ImageEnhance
from core.utils.transform import *
import math

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

import torch
from torchvision.transforms import ColorJitter, functional, Compose
import torch.nn.functional as F

def get_middlebury_images():
    root = "datasets/Middlebury/MiddEval3"
    with open(os.path.join(root, "official_train.txt"), 'r') as f:
        lines = f.read().splitlines()
    return sorted([os.path.join(root, 'trainingQ', f'{name}/im0.png') for name in lines])

def get_eth3d_images():
    return sorted(glob('datasets/ETH3D/two_view_training/*/im0.png'))

def get_kitti_images():
    return sorted(glob('datasets/KITTI/training/image_2/*_10.png'))

def transfer_color(image, style_mean, style_stddev):
    reference_image_lab = color.rgb2lab(image)
    reference_stddev = np.std(reference_image_lab, axis=(0,1), keepdims=True)# + 1
    reference_mean = np.mean(reference_image_lab, axis=(0,1), keepdims=True)

    reference_image_lab = reference_image_lab - reference_mean
    lamb = style_stddev/reference_stddev
    style_image_lab = lamb * reference_image_lab
    output_image_lab = style_image_lab + style_mean
    l, a, b = np.split(output_image_lab, 3, axis=2)
    l = l.clip(0, 100)
    output_image_lab = np.concatenate((l,a,b), axis=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        output_image_rgb = color.lab2rgb(output_image_lab) * 255
        return output_image_rgb

class AdjustGamma(object):

    def __init__(self, gamma_min, gamma_max, gain_min=1.0, gain_max=1.0):
        self.gamma_min, self.gamma_max, self.gain_min, self.gain_max = gamma_min, gamma_max, gain_min, gain_max

    def __call__(self, sample):
        gain = random.uniform(self.gain_min, self.gain_max)
        gamma = random.uniform(self.gamma_min, self.gamma_max)
        return functional.adjust_gamma(sample, gamma, gain)

    def __repr__(self):
        return f"Adjust Gamma {self.gamma_min}, ({self.gamma_max}) and Gain ({self.gain_min}, {self.gain_max})"

class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True, yjitter=False, saturation_range=[0.6,1.4], gamma=[1,1,1,1]):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 1.0
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.yjitter = yjitter
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.4, contrast=0.4, saturation=saturation_range, hue=0.5/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2, bounds=[50, 100]):
        """ Occlusion augmentation """

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow):
        # randomly sample scale
        ht, wd = img1.shape[:2]

        if max(ht, wd) > 2.0 * max(self.crop_size):  # when the input image resolution is too high, downsample it by a certain ratio first.
            pre_scale = (max(self.crop_size) * 1.5) / max(ht, wd)  
            img1 = cv2.resize(img1, None, fx=pre_scale, fy=pre_scale, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=pre_scale, fy=pre_scale, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=pre_scale, fy=pre_scale, interpolation=cv2.INTER_LINEAR)
            flow = flow * [pre_scale, pre_scale]

            # update ht, wd
            ht, wd = img1.shape[:2]

        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht), 
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if (np.random.rand() < self.spatial_aug_prob) or (ht < self.crop_size[0]) or (wd < self.crop_size[1]):
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)

            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf': # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h': # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v': # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        if self.yjitter:
            y0 = np.random.randint(2, img1.shape[0] - self.crop_size[0] - 2)
            x0 = np.random.randint(2, img1.shape[1] - self.crop_size[1] - 2)

            y1 = y0 + np.random.randint(-2, 2 + 1)
            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y1:y1+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        else:
            y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
            x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])
            
            img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
            flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        return img1, img2, flow


    def __call__(self, img1, img2, flow):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow = self.spatial_transform(img1, img2, flow)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)

        return img1, img2, flow

class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False, yjitter=False, saturation_range=[0.7,1.3], gamma=[1,1,1,1]):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = Compose([ColorJitter(brightness=0.3, contrast=0.3, saturation=saturation_range, hue=0.3/3.14), AdjustGamma(*gamma)])
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5
        
    def color_transform(self, img1, img2):
        image_stack = np.concatenate([img1, img2], axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        img1, img2 = np.split(image_stack, 2, axis=0)
        return img1, img2

    def eraser_transform(self, img1, img2):
        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(50, 100)
                dy = np.random.randint(50, 100)
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid>=1]
        flow0 = flow[valid>=1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:,0]).astype(np.int32)
        yy = np.round(coords1[:,1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, img1, img2, flow, valid):
        # randomly sample scale

        ht, wd = img1.shape[:2]

        if max(ht, wd) > 2.0 * max(self.crop_size):  # when the input image resolution is too high, downsample it by a certain ratio first.
            pre_scale = (max(self.crop_size) * 1.5) / max(ht, wd)  
            img1 = cv2.resize(img1, None, fx=pre_scale, fy=pre_scale, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=pre_scale, fy=pre_scale, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=pre_scale, fy=pre_scale, interpolation=cv2.INTER_LINEAR)
            flow = flow * [pre_scale, pre_scale]

            # update ht, wd
            ht, wd = img1.shape[:2]

        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht), 
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if (np.random.rand() < self.spatial_aug_prob) or (ht < self.crop_size[0]) or (wd < self.crop_size[1]):
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow, valid = self.resize_sparse_flow_map(flow, valid, fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob and self.do_flip == 'hf': # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]

            if np.random.rand() < self.h_flip_prob and self.do_flip == 'h': # h-flip for stereo
                tmp = img1[:, ::-1]
                img1 = img2[:, ::-1]
                img2 = tmp

            if np.random.rand() < self.v_flip_prob and self.do_flip == 'v': # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, img1.shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, img1.shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        valid = valid[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        
        return img1, img2, flow, valid


    def __call__(self, img1, img2, flow, valid):
        img1, img2 = self.color_transform(img1, img2)
        img1, img2 = self.eraser_transform(img1, img2)
        img1, img2, flow, valid = self.spatial_transform(img1, img2, flow, valid)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        valid = np.ascontiguousarray(valid)

        return img1, img2, flow, valid

    
# def PinholeEulerAnglesToRotationMatrix(theta):
#     R_x = np.array([[1, 0, 0],
#                     [0, math.cos(theta[0]), -math.sin(theta[0])],
#                     [0, math.sin(theta[0]), math.cos(theta[0])]
#                     ])

#     R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
#                     [0, 1, 0],
#                     [-math.sin(theta[1]), 0, math.cos(theta[1])]
#                     ])

#     R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
#                     [math.sin(theta[2]), math.cos(theta[2]), 0],
#                     [0, 0, 1]
#                     ])

#     R = np.dot(R_z, np.dot(R_y, R_x))

#     return R


# class Augmentor:
#     def __init__(
#             self,
#             image_height=384,
#             image_width=512,
#             max_disp=150,
#             scale_min=0.6,
#             scale_max=1.0,
#             seed=0,
#             camera_type='pinhole',
#             albumentations_aug=True,
#             white_balance_aug=True,
#             rgb_noise_aug=True,
#             motion_blur_aug=True,
#             local_blur_aug=True,
#             global_blur_aug=True,
#             chromatic_aug=True,
#             camera_motion_aug=True

#     ):
#         super().__init__()
#         self.image_height = image_height
#         self.image_width = image_width
#         self.max_disp = max_disp
#         self.scale_min = scale_min
#         self.scale_max = scale_max
#         self.rng = np.random.RandomState(seed)
#         self.camera_type = camera_type

#         self.albumentations_aug = albumentations_aug
#         self.white_balance_aug = white_balance_aug
#         self.rgb_noise_aug = rgb_noise_aug
#         self.motion_blur_aug = motion_blur_aug
#         self.local_blur_aug = local_blur_aug
#         self.global_blur_aug = global_blur_aug
#         self.chromatic_aug = chromatic_aug
#         self.camera_motion_aug = camera_motion_aug

#         intrinsic = (778, 778, 488, 681)

#         self.K_mat = np.array(
#             [[intrinsic[0], 0.0, intrinsic[2]],
#             [0.0, intrinsic[1], intrinsic[3]],
#             [0.0, 0.0, 1.0]])

#     def chromatic_augmentation(self, img):
#         random_brightness = np.random.uniform(0.8, 1.2)
#         random_contrast = np.random.uniform(0.8, 1.2)
#         random_gamma = np.random.uniform(0.8, 1.2)

#         img = Image.fromarray(img)

#         enhancer = ImageEnhance.Brightness(img)
#         img = enhancer.enhance(random_brightness)
#         enhancer = ImageEnhance.Contrast(img)
#         img = enhancer.enhance(random_contrast)

#         gamma_map = [
#                         255 * 1.0 * pow(ele / 255.0, random_gamma) for ele in range(256)
#                     ] * 3
#         img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

#         img_ = np.array(img)

#         return img_

#     def padding(self, data, pad_len):
#         # top_ext = data[-pad_len:]
#         bott_ext = data[0: pad_len]
#         return np.concatenate((data, bott_ext), axis=0)

#     def __call__(self, dataset_name, left_img, right_img, left_disp, error=None, wire_mask=None):

#         # random crop
#         h, w = left_img.shape[:2]
#         ch, cw = self.image_height, self.image_width
        
#         if ch > h or cw > w:
#             scale = max((ch + 1) / h, (cw + 1) / w)
#             left_img = cv2.resize(left_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
#             right_img = cv2.resize(right_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
#             left_disp = cv2.resize(left_disp[:,:,None], None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR).squeeze().astype(np.float32) * scale
#             h, w = left_img.shape[:2]

#         assert ch <= h and cw <= w, (left_img.shape, h, w, ch, cw)
#         offset_x = np.random.randint(w - cw + 1)
#         offset_y = np.random.randint(h - ch + 1)

#         left_img = left_img[offset_y: offset_y + ch, offset_x: offset_x + cw]
#         right_img = right_img[offset_y: offset_y + ch, offset_x: offset_x + cw]
#         left_disp = left_disp[offset_y: offset_y + ch, offset_x: offset_x + cw]

#         if error is not None:
#             error = error[offset_y: offset_y + ch, offset_x: offset_x + cw]
#         if wire_mask is not None:
#             wire_mask = wire_mask[offset_y: offset_y + ch, offset_x: offset_x + cw]

#         right_img_ori = right_img.copy()

#         # disp mask
#         resize_scale = 1.0
#         # disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0)
#         if error is not None:
#             # error_mask = error/(left_disp+1e-6) <1.0
#             error_mask = error < 0.5
#             # error_mask_int = error_mask.astype(np.uint8)
#             # print(error_mask_int.sum(), error_mask_int.sum()/(error_mask.shape[0]*error_mask.shape[1]))
#             disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0) & error_mask
#         else:
#             disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0)
#         # disp_mask = disp_mask.astype("float32")

#         if self.local_blur_aug and self.rng.binomial(1, 0.2):

#             # 左图模糊增广
#             p_l = self.rng.binomial(1, 0.5)
#             if p_l < 0.5:
#                 brightness = self.rng.uniform(low=-40, high=40)
#                 mask_l = self.rng.choice([None, 'local_mask'])
#                 if mask_l == 'local_mask':
#                     mask_l = mask_ge(left_img.shape, self.rng, weights=[0.5, 0.5])
#                 left_img, _ = image_blur_mask(left_img, self.rng, mask_l, brightness)

#             # 右图模糊增广
#             p_r = self.rng.binomial(1, 0.5)
#             if p_r < 0.5:
#                 mask_r = self.rng.choice([None, 'local_mask'])
#                 brightness = self.rng.uniform(low=-40, high=40)
#                 if mask_r == 'local_mask':
#                     mask_r = mask_ge(left_img.shape, self.rng, weights=[0.5, 0.5])
#                 right_img, _ = image_blur_mask(right_img, self.rng, mask_r, brightness)

#         if self.rgb_noise_aug and self.rng.binomial(1, 0.5):
#             sigma = self.rng.uniform(low=1, high=5)
#             left_img = RGB_noise_aug(left_img, sigma, self.rng)
#             right_img = RGB_noise_aug(right_img, sigma, self.rng)

#         left_img = left_img.astype(np.uint8)
#         right_img = right_img.astype(np.uint8)

#         if self.chromatic_aug and self.rng.binomial(1, 0.4):

#             left_img = self.chromatic_augmentation(left_img)
#             right_img = self.chromatic_augmentation(right_img)

#         # Diff chromatic # White balance
#         if self.white_balance_aug and self.rng.binomial(1, 0.5):
#             random_ratio_L = self.rng.uniform(-0.3, 0.3)
#             random_ratio_R = self.rng.uniform(-0.15, 0.15) + random_ratio_L
#             left_img = white_balance_augmentation(left_img, ratio=random_ratio_L)
#             right_img = white_balance_augmentation(right_img, ratio=random_ratio_R)

#             # global aug # 模拟失焦
#         if self.global_blur_aug and self.rng.binomial(1, 0.2):

#             # 左图模糊增广
#             p_l = self.rng.binomial(1, 0.5)
#             if p_l < 0.5:
#                 kernel_size = self.rng.randint(2, 7) * 2 + 1
#                 left_img, _ = image_blur_all(left_img, (kernel_size, kernel_size))

#             # 右图模糊增广
#             p_r = self.rng.binomial(1, 0.5)
#             if p_r < 0.5:
#                 # kernel = self.rng.randint(5, 15)
#                 kernel_size = self.rng.randint(2, 7) * 2 + 1
#                 right_img, _ = image_blur_all(right_img, (kernel_size, kernel_size))

#         # 2. spatial augmentation
#         # 2.1) rotate & vertical shift for right image
#         if self.camera_motion_aug and self.rng.binomial(1, 0.2):
#             sigma = 0.25
#             mu = 0  # mean and standard deviation
#             ag_0, ag_1, ag_2 = np.fmod(np.random.normal(mu, sigma, size=3), 3)

#             angle, pixel = (0.3, 0., 0.1), 2 # 横向偏移为0
#             px = self.rng.uniform(-pixel, pixel)
#             # ag = np.deg2rad([ag_0, ag_1, ag_2])
#             ag = np.deg2rad(angle)

#             self.K_mat_new = self.K_mat.copy()
#             self.K_mat_new[1, 2] += px

#             R_mat = PinholeEulerAnglesToRotationMatrix(ag)
#             H_mat = self.K_mat_new.dot(R_mat).dot(np.linalg.inv(self.K_mat))
#             H_mat = H_mat / H_mat[2][2]
#             right_img = cv2.warpPerspective(
#                 right_img, H_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
#             )

#         left_img = left_img.astype(np.float32)
#         right_img = right_img.astype(np.float32)
#         # random occlusion
#         if self.rng.binomial(1, 0.5):
#             sx = int(self.rng.uniform(50, 100))
#             sy = int(self.rng.uniform(50, 100))
#             cx = int(self.rng.uniform(sx, right_img.shape[0] - sx))
#             cy = int(self.rng.uniform(sy, right_img.shape[1] - sy))
#             right_img[cx - sx: cx + sx, cy - sy: cy + sy] = np.mean(
#                 np.mean(right_img, 0), 0
#             )[np.newaxis, np.newaxis]

#         # color mask 过滤掉过曝, 欠曝的像素
#         # color_mask = np.logical_and(np.mean(left_img, axis=2) < 235, np.mean(left_img, axis=2) > 20)
#         # disp_mask = np.logical_and(disp_mask, color_mask)

#         left_img = np.ascontiguousarray(left_img)
#         right_img = np.ascontiguousarray(right_img)
#         right_img_ori = np.ascontiguousarray(right_img_ori)
#         left_disp = np.ascontiguousarray(left_disp)
#         disp_mask = np.ascontiguousarray(disp_mask)
#         wire_mask = np.ascontiguousarray(wire_mask)

#         return left_img, right_img, right_img_ori, left_disp, disp_mask, wire_mask