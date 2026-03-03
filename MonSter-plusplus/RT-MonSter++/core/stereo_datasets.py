import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp
import gzip
from collections import defaultdict
from typing import List, Optional
from dataclasses import dataclass
import cv2
from pytorch3d.implicitron.dataset.types import (
    FrameAnnotation as ImplicitronFrameAnnotation,
    load_dataclass
)

import sys
sys.path.append(os.getcwd())

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor

@dataclass
class DynamicReplicaFrameAnnotation(ImplicitronFrameAnnotation):
    """A dataclass used to load annotations from json."""

    camera_name: Optional[str] = None

class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None, test_mode=False):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        

        self.is_test = test_mode
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []
        self.viewpoint_list = []
        self.metadata_list = []

    def __getitem__(self, index):


        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        if len(self.viewpoint_list) == 0:
            disp = self.disparity_reader(self.disparity_list[index])
        else:
            disp = self.disparity_reader(self.disparity_list[index], self.viewpoint_list[index], self.metadata_list[index])
        
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 512

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)

        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:

                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 512) & (flow[1].abs() < 512)

        if self.img_pad is not None:

            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        flow = flow[:1]
        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()


    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        if len(self.viewpoint_list) > 0:
            copy_of_self.viewpoint_list = v * copy_of_self.viewpoint_list
            copy_of_self.metadata_list = v * copy_of_self.metadata_list
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)


class SceneFlowDatasets(StereoDataset):
    def __init__(self, aug_params=None, root="/data/StereoData/sceneflow", dstype='frames_finalpass', things_test=False):
        super(SceneFlowDatasets, self).__init__(aug_params, test_mode=things_test)
        assert os.path.exists(root)
        self.root = root
        self.dstype = dstype

        if things_test:
            self._add_things("TEST")
        else:
            self._add_things("TRAIN")
            self._add_monkaa("TRAIN")
            self._add_driving("TRAIN")

    def _add_things(self, split='TRAIN'):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        right_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/right/*.png')) )  
        left_images = [ im.replace('right', 'left') for im in right_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for idx, (img1, img2, disp) in enumerate(zip(left_images, right_images, disparity_images)):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from FlyingThings {self.dstype}")

    def _add_monkaa(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        right_images = sorted( glob(osp.join(root, self.dstype, split, '*/right/*.png')) )  
        left_images = [ im.replace('right', 'left') for im in right_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Monkaa {self.dstype}")


    def _add_driving(self, split="TRAIN"):
        """ Add FlyingThings3D data """

        original_length = len(self.disparity_list)
        root = self.root
        right_images = sorted( glob(osp.join(root, self.dstype, split, '*/*/*/right/*.png')) )  
        left_images = [ im.replace('right', 'left') for im in right_images ]
        disparity_images = [ im.replace(self.dstype, 'disparity').replace('.png', '.pfm') for im in left_images ]

        for img1, img2, disp in zip(left_images, right_images, disparity_images):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]
        logging.info(f"Added {len(self.disparity_list) - original_length} from Driving {self.dstype}")


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root="/data/StereoData/eth3d/", split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)
        assert os.path.exists(root)

        image1_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im0.png')) )
        image2_list = sorted( glob(osp.join(root, f'two_view_{split}/*/im1.png')) )
        disp_list = sorted( glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm')) ) if split == 'training' else [osp.join(root, 'two_view_training_gt/playground_1l/disp0GT.pfm')] * len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class SintelStereo(StereoDataset):
    def __init__(self, aug_params=None, root="/data3/StereoData/sintelstereo"):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispSintelStereo)

        image1_list = sorted( glob(osp.join(root, 'training/*_left/*/frame_*.png')) )
        image2_list = sorted( glob(osp.join(root, 'training/*_right/*/frame_*.png')) )
        disp_list = sorted( glob(osp.join(root, 'training/disparities/*/frame_*.png')) ) * 2

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-2:] == disp.split('/')[-2:]
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class FallingThings(StereoDataset):
    def __init__(self, aug_params=None, root='/data3/StereoData/fallingthings'):
        super().__init__(aug_params, reader=frame_utils.readDispFallingThings)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/*/*/*left.jpg'))
        image2_list = sorted(glob(root + '/*/*/*right.jpg'))
        disp_list = sorted(glob(root + '/*/*/*left.depth.png'))

        image1_list += sorted(glob(root + '/*/*/*/*left.jpg'))
        image2_list += sorted(glob(root + '/*/*/*/*right.jpg'))
        disp_list += sorted(glob(root + '/*/*/*/*left.depth.png'))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class TartanAir(StereoDataset):
    def __init__(self, aug_params=None, root='/data3/StereoData/tartanair'):
        super().__init__(aug_params, reader=frame_utils.readDispTartanAir)
        assert os.path.exists(root)


        image1_list = sorted( glob(osp.join(root, '*/*/*/*/image_left/*.png')) )
        image2_list = sorted( glob(osp.join(root, '*/*/*/*/image_right/*.png')) )
        disp_list = sorted( glob(osp.join(root, '*/*/*/*/depth_left/*.npy')) )

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoData/kitti', image_set='training'):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)

        root_12 = '/data/StereoData/kitti/2012/'
        image1_list = sorted(glob(os.path.join(root_12, image_set, 'colored_0/*_10.png')))
        image2_list = sorted(glob(os.path.join(root_12, image_set, 'colored_1/*_10.png')))
        disp_list = sorted(glob(os.path.join(root_12, 'training', 'disp_occ/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ/000085_10.png')]*len(image1_list)

        root_15 = '/data/StereoData/kitti/2015/'
        image1_list += sorted(glob(os.path.join(root_15, image_set, 'image_2/*_10.png')))
        image2_list += sorted(glob(os.path.join(root_15, image_set, 'image_3/*_10.png')))
        disp_list += sorted(glob(os.path.join(root_15, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class KITTI_2012(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoData/kitti', image_set='training'):
        super(KITTI_2012, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)

        root_12 = '/data/StereoData/kitti/2012/'  #'/data/lwj/kitti-1210/2012/'#
        image1_list = sorted(glob(os.path.join(root_12, image_set, 'colored_0/*_10.png')))
        image2_list = sorted(glob(os.path.join(root_12, image_set, 'colored_1/*_10.png')))
        disp_list = sorted(glob(os.path.join(root_12, 'training', 'disp_occ/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ/000085_10.png')]*len(image1_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class KITTI_2015(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoData/kitti', image_set='training'):
        super(KITTI_2015, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root)

        root_15 = '/data/StereoData/kitti/2015/'  #'/data/lwj/kitti-1210/2015/'#
        image1_list = sorted(glob(os.path.join(root_15, image_set, 'image_2/*_10.png')))
        image2_list = sorted(glob(os.path.join(root_15, image_set, 'image_3/*_10.png')))
        disp_list = sorted(glob(os.path.join(root_15, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(root, 'training/disp_occ_0/000085_10.png')]*len(image1_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class VKITTI2(StereoDataset):
    def __init__(self, aug_params=None, root='/mnt/truenas_nfs/dataset/vkitti2'):
        super(VKITTI2, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispVKITTI2)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/rgb/Camera_0/rgb*.jpg')))
        image2_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/rgb/Camera_1/rgb*.jpg')))
        disp_list = sorted(glob(os.path.join(root, 'Scene*/*/frames/depth/Camera_0/depth*.png')))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='/data/StereoData/middlebury', split='MiddEval3', resolution='H'):
        super(Middlebury, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispMiddlebury)
        assert os.path.exists(root)
        assert split in ["2005", "2006", "2014", "2021", "MiddEval3"]
        if split == "2005":
            scenes = list((Path(root) / "2005").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disparity_list += [str(scene / "disp1.png")]    
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:       
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"), str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [str(scene / "disp1.png")]        
        elif split == "2006":
            scenes = list((Path(root) / "2006").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disparity_list += [str(scene / "disp1.png")]    
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:       
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"), str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [str(scene / "disp1.png")]
        elif split == "2014":
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E", "L", ""]:
                    self.image_list += [ [str(scene / "im0.png"), str(scene / f"im1{s}.png")] ]
                    self.disparity_list += [ str(scene / "disp0.pfm") ]
        elif split == "2021":
            scenes = list((Path(root) / "2021/data").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "im0.png"), str(scene / "im1.png")]]
                self.disparity_list += [str(scene / "disp0.pfm")]
                for s in ["0", "1", "2", "3"]:
                    if os.path.exists(str(scene / f"ambient/L0/im0e{s}.png")):
                        self.image_list += [[str(scene / f"ambient/L0/im0e{s}.png"), str(scene / f"ambient/L0/im1e{s}.png")]]
                        self.disparity_list += [str(scene / "disp0.pfm")]
        else:
            image1_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/im0.png')))
            image2_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/im1.png')))
            disp_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/disp0GT.pfm')))
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]

class CREStereoDataset(StereoDataset):
    def __init__(self, aug_params=None, root='/data3/StereoData/crestereo'):
        super(CREStereoDataset, self).__init__(aug_params, reader=frame_utils.readDispCREStereo)
        assert os.path.exists(root)

        image1_list = sorted(glob(os.path.join(root, '*/*_left.jpg')))
        image2_list = sorted(glob(os.path.join(root, '*/*_right.jpg')))
        disp_list = sorted(glob(os.path.join(root, '*/*_left.disp.png')))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class InStereo2K(StereoDataset):
    def __init__(self, aug_params=None, root="/data3/StereoData/instereo2k"):
        super(InStereo2K, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispInStereo2K)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/train/*/*/left.png') + glob(root + '/test/*/left.png'))
        image2_list = sorted(glob(root + '/train/*/*/right.png') + glob(root + '/test/*/right.png'))
        disp_list = sorted(glob(root + '/train/*/*/left_disp.png') + glob(root + '/test/*/left_disp.png'))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class CARLA(StereoDataset):
    def __init__(self, aug_params=None, root='/data3/StereoData/carla-highres'):
        super(CARLA, self).__init__(aug_params)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/trainingF/*/im0.png'))
        image2_list = sorted(glob(root + '/trainingF/*/im1.png'))
        disp_list = sorted(glob(root + '/trainingF/*/disp0GT.pfm'))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class DrivingStereo(StereoDataset):
    def __init__(self, aug_params=None, root='/mnt/truenas_nfs/dataset/drivingstereo/', image_set='rainy'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispDrivingStereo_half)
        assert os.path.exists(root)
        image1_list = sorted(glob(os.path.join(root, image_set, 'left-image-half-size/*.jpg')))
        image2_list = sorted(glob(os.path.join(root, image_set, 'right-image-half-size/*.jpg')))
        disp_list = sorted(glob(os.path.join(root, image_set, 'disparity-map-half-size/*.png')))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class FoundationStereoDataset(StereoDataset):
    def __init__(self, aug_params=None, root='/data3/StereoData/FSD'):  
        super().__init__(aug_params, reader=frame_utils.readFoundationStereo)
        assert os.path.exists(root)
        image1_list = sorted(glob(os.path.join(root, '*/dataset/data/left/rgb/*.jpg')))
        image2_list = sorted(glob(os.path.join(root, '*/dataset/data/right/rgb/*.jpg')))
        disp_list = sorted(glob(os.path.join(root, '*/dataset/data/left/disparity/*.png')))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [ [img1, img2] ]
            self.disparity_list += [ disp ]

class IRS(StereoDataset):
    def __init__(self, aug_params=None, root='/data3/StereoData/irs'):
        super(IRS, self).__init__(aug_params, reader=frame_utils.readDispIRS)
        assert os.path.exists(root)

        image1_list = sorted(glob(root + '/*/*/left_image/l_*.png'))
        image2_list = sorted(glob(root + '/*/*/right_image/r_*.png'))
        disp_list = sorted(glob(root + '/*/*/depth/d_*.exr'))

        assert len(image1_list) == len(image2_list) == len(disp_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            assert img1.split('/')[-3] == disp.split('/')[-3]
            assert img1.split('.')[0].split('_')[-1] == disp.split('.')[0].split('_')[-1]
            if 'QAOfficeAndSecurityRoom2_Night' in img1: # bad scenes
                continue
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]

class Booster(StereoDataset):
    def __init__(self, aug_params=None, root='/data3/StereoData/booster/', split='train'): 
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispBooster) 

        folder_list = sorted(glob(osp.join(root, split+'/balanced/*')))
        for folder in folder_list:
            image1_list = sorted(glob(osp.join(folder, 'camera_00/im*.png')))
            image2_list = sorted(glob(osp.join(folder, 'camera_02/im*.png')))
            if split=="train":
                for img1 in image1_list:
                    for img2 in image2_list:
                        self.image_list += [[img1, img2]]
                        self.disparity_list += [osp.join(folder, 'disp_00.npy')]
            else:
                for img1, img2 in zip(image1_list, image2_list):
                    self.image_list += [[img1, img2]]
                    self.disparity_list += [osp.join(folder, 'disp_00.npy')]
                    
class ThreeDKenBurns(StereoDataset):
    def __init__(self, aug_params=None, root='/data3/StereoData/3dkenburns'):
        super().__init__(aug_params, reader=frame_utils.readDisp3DKenBurns)

        image1_list = sorted(glob(osp.join(root, '*/*l-image.png')))
        image2_list = sorted(glob(osp.join(root, '*/*r-image.png')))
        disp1_list = sorted(glob(osp.join(root, '*/*l-depth.exr')))

        assert len(image1_list) == len(image2_list) == len(disp1_list)

        for img1, img2, disp1 in zip(image1_list, image2_list, disp1_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp1]


class DynamicReplica(StereoDataset):
    def __init__(self, aug_params=None, root='/data3/StereoData/dynamicstereo', split='train'):
        super(DynamicReplica, self).__init__(aug_params, reader=frame_utils.readDispDynamicReplica)
        assert os.path.exists(root)

        frame_annotations_file = f'frame_annotations_{split}.jgz'
        with gzip.open(osp.join(root, split, frame_annotations_file), 'rt', encoding='utf8') as zipfile:
            frame_annots_list = load_dataclass(zipfile, List[DynamicReplicaFrameAnnotation])

        seq_annot = defaultdict(lambda: defaultdict(list))
        for frame_annot in frame_annots_list:
            seq_annot[frame_annot.sequence_name][frame_annot.camera_name].append(frame_annot)

        for seq_name in seq_annot.keys():

            image1_list, image2_list, disp_list = [], [], []
            viewpoint1_list, viewpoint2_list = [], []
            metadata1_list, metadata2_list = [], []
            for cam in ['left', 'right']:
                for frame_data in seq_annot[seq_name][cam]:
                    if cam == 'left':
                        image1_list.append(osp.join(root, split, frame_data.image.path))
                        disp_list.append(osp.join(root, split, frame_data.depth.path))
                        viewpoint1_list.append(frame_data.viewpoint)
                        metadata1_list.append([frame_data.sequence_name, frame_data.image.size])
                    else:
                        image2_list.append(osp.join(root, split, frame_data.image.path))
                        viewpoint2_list.append(frame_data.viewpoint)
                        metadata2_list.append([frame_data.sequence_name, frame_data.image.size])

            for idx, (img1, img2, disp, view1, view2, meta1, meta2) in enumerate(zip(image1_list, image2_list, disp_list, viewpoint1_list, viewpoint2_list, metadata1_list, metadata2_list)):
                self.image_list += [ [img1, img2] ]
                self.disparity_list += [ disp ]
                self.viewpoint_list += [ [view1, view2] ]
                self.metadata_list += [ [meta1, meta2] ]

class VA(StereoDataset):
    def __init__(self, aug_params=None, root='/data3/StereoData/VA'):
        super().__init__(aug_params, reader=frame_utils.readDispVA)

        image1_list = sorted(glob(osp.join(root, 'camera_0/*.png')))
        image2_list = sorted(glob(osp.join(root, 'camera_1/*.png')))
        disp1_list = sorted(glob(osp.join(root, 'gt_depth_camera_0/*.pfm')))

        assert len(image1_list) == len(image2_list) == len(disp1_list)

        for img1, img2, disp1 in zip(image1_list, image2_list, disp1_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp1]

class SimSIN(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/data_geo/SimSIN/'):
        super().__init__(aug_params, reader=frame_utils.readDispSimSIN)

        image1_list = sorted(glob(osp.join(root, 'baseline_13cm/*/*/*/*_0.png')))
        image2_list = sorted(glob(osp.join(root, 'baseline_13cm/*/*/*/*_1.png')))
        disp1_list = sorted(glob(osp.join(root, 'baseline_13cm/*/*/*/*_0_depth.npy')))

        assert len(image1_list) == len(image2_list) == len(disp1_list)

        for img1, img2, disp1 in zip(image1_list, image2_list, disp1_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp1]


class UnrealStereo4K(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/data_geo/UnrealStereo4K/'):
        super().__init__(aug_params, reader=frame_utils.readDispUnrealStereo4K)

        image1_list = sorted(glob(osp.join(root, '*/Image0/*.jpg')))
        image2_list = sorted(glob(osp.join(root, '*/Image1/*.jpg')))
        disp1_list = sorted(glob(osp.join(root, '*/Disp0/*.npy')))

        assert len(image1_list) == len(image2_list) == len(disp1_list)

        for img1, img2, disp1 in zip(image1_list, image2_list, disp1_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp1]

class Spring(StereoDataset):
    def __init__(self, aug_params=None, root='/data3/StereoData/spring'):
        super().__init__(aug_params, reader=frame_utils.readDispSpring)

        image1_list = sorted(glob(osp.join(root, 'train/*/frame_left/*.png')))
        image2_list = [p.replace('frame_left', 'frame_right') for p in image1_list]
        disp1_list  = [p.replace('frame_left', 'disp1_left').replace('.png', '.dsp5') for p in image1_list]
        assert len(image1_list) == len(image2_list) == len(disp1_list)
        for img1, img2, disp1 in zip(image1_list, image2_list, disp1_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp1]

class StereoBlur(StereoDataset):
    def __init__(self, aug_params=None, root='/data2/data_geo/StereoBlurDataset/'):
        super().__init__(aug_params, sparse=True, reader=frame_utils.readDispStereoBlur)

        image1_list = sorted(glob(osp.join(root, 'HD720*/image_left*/*.png')))
        image2_list = [p.replace('image_left', 'image_right') for p in image1_list]
        disp1_list  = [p.replace('image_left_blur_ga', 'disparity_left').replace('image_left', 'disparity_left').replace('.png', '.exr') for p in image1_list]
        assert len(image1_list) == len(image2_list) == len(disp1_list)
        for img1, img2, disp1 in zip(image1_list, image2_list, disp1_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp1]
        # self.clear_img_list = [[img1.replace("/image_left/", "/image_left_blur_ga/"), img2.replace("/image_right/", "/image_right_blur_ga/")] for img1, img2 in self.image_list]


def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """
    # print('args.img_gamma', args.img_gamma)
    aug_params = {'crop_size': list(args.image_size), 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = list(args.saturation_range)
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip


    train_dataset = None
    print('train_datasets', args.train_datasets)
    for dataset_name in args.train_datasets:
        if dataset_name == 'sceneflow':
            new_dataset = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            logging.info(f"Adding {len(new_dataset)} samples from SceneFlow")
        elif dataset_name == 'kitti':
            new_dataset = KITTI(aug_params)
            logging.info(f"Adding {len(new_dataset)} samples from KITTI")
        elif dataset_name == 'vkitti':
            new_dataset = VKITTI2(aug_params)
            logging.info(f"Adding {len(new_dataset)} samples from VKITTI2")
        elif dataset_name == 'driving_scene':
            kitti = KITTI(aug_params)
            logging.info(f"Adding {len(kitti)} samples from KITTI")
            vkitti2 = VKITTI2(aug_params)
            logging.info(f"Adding {len(vkitti2)} samples from VKITTI2")
            new_dataset = kitti * 200 + vkitti2 
        elif dataset_name == 'sintel_stereo':
            new_dataset = SintelStereo(aug_params)*140
            logging.info(f"Adding {len(new_dataset)} samples from Sintel Stereo")
        elif dataset_name == 'falling_things':
            new_dataset = FallingThings(aug_params)*5
            logging.info(f"Adding {len(new_dataset)} samples from FallingThings")
        elif dataset_name.startswith('tartan_air'):
            new_dataset = TartanAir(aug_params, keywords=dataset_name.split('_')[2:])
            logging.info(f"Adding {len(new_dataset)} samples from Tartain Air")
        elif dataset_name.startswith('zero_shot'):
            va = VA(aug_params)
            logging.info(f"Adding {len(va)} samples from VA Dataset")
            booster = Booster(aug_params)
            logging.info(f"Adding {len(booster)} samples from Booster Dataset")
            irs = IRS(aug_params)
            logging.info(f"Adding {len(irs)} samples from IRS Dataset")
            dynamicstereo = DynamicReplica(aug_params)
            logging.info(f"Adding {len(dynamicstereo)} samples from DynamicStereo Dataset")
            threedkenburns = ThreeDKenBurns(aug_params)
            logging.info(f"Adding {len(threedkenburns)} samples from ThreeDKenBurns Dataset")
            spring = Spring(aug_params)
            logging.info(f"Adding {len(spring)} samples from Spring Dataset")
            crestereo = CREStereoDataset(aug_params)
            logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")                 
            instereo2k = InStereo2K(aug_params)
            logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
            carla = CARLA(aug_params)
            logging.info(f"Adding {len(carla)} samples from CARLA")
            vkitti2 = VKITTI2(aug_params)
            logging.info(f"Adding {len(vkitti2)} samples from VKITTI2")
            tartanair = TartanAir(aug_params)
            logging.info(f"Adding {len(tartanair)} samples from Tartain Air")
            sceneflow = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            logging.info(f"Adding {len(sceneflow)} samples from SceneFlow")
            fsd = FoundationStereoDataset(aug_params)
            logging.info(f"Adding {len(fsd)} samples from FoundationStereoDataset")
            fallingthings = FallingThings(aug_params)
            logging.info(f"Adding {len(fallingthings)} samples from FallingThings")
            sintel = SintelStereo(aug_params)
            logging.info(f"Adding {len(sintel)} samples from Sintel Stereo")
            stereoblur = StereoBlur(aug_params)
            logging.info(f"Adding {len(stereoblur)} samples from StereoBlur")
            unrealstereo = UnrealStereo4K(aug_params)
            logging.info(f"Adding {len(unrealstereo)} samples from UnrealStereo4K")
            simsin = SimSIN(aug_params)
            logging.info(f"Adding {len(simsin)} samples from SimSIN")
            new_dataset = fsd + tartanair + crestereo + threedkenburns + irs + dynamicstereo + fallingthings + sceneflow + vkitti2 * 4 + spring * 4 + va * 10 + sintel * 10 + instereo2k * 40 + booster * 60 + carla * 100 +  stereoblur * 2 + simsin + unrealstereo * 10
        elif dataset_name == 'eth3d_train':
            fds = FoundationStereoDataset(aug_params)
            logging.info(f"Adding {len(fds)} samples from FoundationStereoDataset")
            tartanair = TartanAir(aug_params)
            logging.info(f"Adding {len(tartanair)} samples from Tartain Air")
            sceneflow = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            logging.info(f"Adding {len(sceneflow)} samples from SceneFlow")
            crestereo = CREStereoDataset(aug_params)
            logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")
            instereo2k = InStereo2K(aug_params)
            logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
            sintel = SintelStereo(aug_params)
            logging.info(f"Adding {len(sintel)} samples from Sintel Stereo")
            eth3d = ETH3D(aug_params)
            logging.info(f"Adding {len(eth3d)} samples from ETH3D")
            new_dataset = fds + tartanair + sceneflow + sintel * 50 + eth3d * 1000 + instereo2k  * 100 + crestereo * 2#tartanair +   + crestereo * 2
            logging.info(f"Adding {len(new_dataset)} samples from ETH3D Mixture Dataset")
        elif dataset_name == 'eth3d_finetune':
            crestereo = CREStereoDataset(aug_params)
            logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")            
            eth3d = ETH3D(aug_params)
            logging.info(f"Adding {len(eth3d)} samples from ETH3D")
            instereo2k = InStereo2K(aug_params)
            logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
            new_dataset = eth3d * 1000 + instereo2k * 10 + crestereo
            logging.info(f"Adding {len(new_dataset)} samples from ETH3D Mixture Dataset")
        elif dataset_name == 'middlebury_train':
            tartanair = TartanAir(aug_params)
            logging.info(f"Adding {len(tartanair)} samples from Tartain Air")
            sceneflow = SceneFlowDatasets(aug_params, dstype='frames_finalpass')
            logging.info(f"Adding {len(sceneflow)} samples from SceneFlow")
            fallingthings = FallingThings(aug_params)
            logging.info(f"Adding {len(fallingthings)} samples from FallingThings")
            carla = CARLA(aug_params)
            logging.info(f"Adding {len(carla)} samples from CARLA")
            crestereo = CREStereoDataset(aug_params)
            logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")             
            instereo2k = InStereo2K(aug_params)
            logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
            mb2005 = Middlebury(aug_params, split='2005')
            logging.info(f"Adding {len(mb2005)} samples from Middlebury 2005")
            mb2006 = Middlebury(aug_params, split='2006')
            logging.info(f"Adding {len(mb2006)} samples from Middlebury 2006")
            mb2014 = Middlebury(aug_params, split='2014')
            logging.info(f"Adding {len(mb2014)} samples from Middlebury 2014")
            mb2021 = Middlebury(aug_params, split='2021')
            logging.info(f"Adding {len(mb2021)} samples from Middlebury 2021")
            mbeval3 = Middlebury(aug_params, split='MiddEval3', resolution='H')
            logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
            new_dataset = tartanair + sceneflow + fallingthings + instereo2k * 50 + carla * 50 + crestereo + mb2005 * 200 + mb2006 * 200 + mb2014 * 200 + mb2021 * 200 + mbeval3 * 200
            logging.info(f"Adding {len(new_dataset)} samples from Middlebury Mixture Dataset")
        elif dataset_name == 'middlebury_finetune':
            crestereo = CREStereoDataset(aug_params)
            logging.info(f"Adding {len(crestereo)} samples from CREStereo Dataset")                 
            instereo2k = InStereo2K(aug_params)
            logging.info(f"Adding {len(instereo2k)} samples from InStereo2K")
            carla = CARLA(aug_params)
            logging.info(f"Adding {len(carla)} samples from CARLA")
            mb2005 = Middlebury(aug_params, split='2005')
            logging.info(f"Adding {len(mb2005)} samples from Middlebury 2005")
            mb2006 = Middlebury(aug_params, split='2006')
            logging.info(f"Adding {len(mb2006)} samples from Middlebury 2006")
            mb2014 = Middlebury(aug_params, split='2014')
            logging.info(f"Adding {len(mb2014)} samples from Middlebury 2014")
            mb2021 = Middlebury(aug_params, split='2021')
            logging.info(f"Adding {len(mb2021)} samples from Middlebury 2021")
            mbeval3 = Middlebury(aug_params, split='MiddEval3', resolution='H')
            logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
            mbeval3_f = Middlebury(aug_params, split='MiddEval3', resolution='F')
            logging.info(f"Adding {len(mbeval3)} samples from Middlebury Eval3")
            fallingthings = FallingThings(aug_params)
            logging.info(f"Adding {len(fallingthings)} samples from FallingThings")
            new_dataset = crestereo + instereo2k * 50 + carla * 50 + mb2005 * 200 + mb2006 * 200 + mb2014 * 200 + mb2021 * 200 + mbeval3 * 200 + mbeval3_f * 400 + fallingthings * 5
            logging.info(f"Adding {len(new_dataset)} samples from Middlebury Mixture Dataset")

        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    return train_dataset


if __name__ == '__main__':
    import matplotlib
    import matplotlib.pyplot as plt
    import cv2


    def gray_2_colormap_np(img, cmap='rainbow', max=None):
        img = img.cpu().detach().numpy().squeeze()
        assert img.ndim == 2
        img[img < 0] = 0
        mask_invalid = img < 1e-10
        if max == None:
            img = img / (img.max() + 1e-8)
        else:
            img = img / (max + 1e-8)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=1.1)
        cmap_m = matplotlib.cm.get_cmap(cmap)
        map = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap_m)
        colormap = (map.to_rgba(img)[:, :, :3] * 255).astype(np.uint8)
        colormap[mask_invalid] = 0

        return colormap


    def viz_disp(disp, scale=1, COLORMAP=cv2.COLORMAP_JET):
        disp_np = (torch.abs(disp[0].squeeze())).data.cpu().numpy()
        disp_np = (disp_np * scale).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_np, COLORMAP)
        return disp_color
    plot_dir = './temp/plots'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    dataset = VKITTI2()
    for i in range(5):
        _, *data_blob = dataset[i]
        image1, image2, disp_gt, valid = [x[None] for x in data_blob]
        image1_np = image1[0].squeeze().cpu().numpy()
        image1_np = (image1_np - image1_np.min()) / (image1_np.max() - image1_np.min()) * 255.0
        image1_np = image1_np.astype(np.uint8)

        disp_color = viz_disp(disp_gt, scale=5)
        cv2.imwrite(os.path.join(plot_dir, f'{i}_disp_gt.png'), disp_color)

        disp_gt_np = gray_2_colormap_np(disp_gt[0].squeeze())
        cv2.imwrite(os.path.join(plot_dir, f'{i}_disp_gt1.png'), disp_gt_np[:, :, ::-1])

        image1 = image1[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        cv2.imwrite(os.path.join(plot_dir, f'{i}_img1.png'), image1)

