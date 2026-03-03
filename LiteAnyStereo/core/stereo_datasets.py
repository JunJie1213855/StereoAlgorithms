import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import os
import copy
import random
from pathlib import Path
from glob import glob
import os.path as osp
from .utils import frame_utils


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None, real_world=False):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []
        self.real_world = real_world

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        try:
            disp = self.disparity_reader(self.disparity_list[index])
            if isinstance(disp, tuple):
                disp, valid = disp
            else:
                valid = disp < 192

            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])

            img1 = np.array(img1).astype(np.uint8)
            img2 = np.array(img2).astype(np.uint8)

            disp = np.array(disp).astype(np.float32)

            flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

            # grayscale images
            if len(img1.shape) == 2:
                img1 = np.tile(img1[..., None], (1, 1, 3))
            else:
                img1 = img1[..., :3]

            if len(img2.shape) == 2:
                img2 = np.tile(img2[..., None], (1, 1, 3))
            else:
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
                valid = (flow[0].abs() < 192) & (flow[1].abs() < 192)

            if self.img_pad is not None:
                padH, padW = self.img_pad
                img1 = F.pad(img1, [padW] * 2 + [padH] * 2)
                img2 = F.pad(img2, [padW] * 2 + [padH] * 2)

            flow = flow[:1]
            return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

        except Exception as e:
            # Useful for locating the file later:
            print(f"[WARN] Skipping corrupted sample:\n  {self.image_list[index]}\n  Err={e}")
            return self.__getitem__(0)

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self

    def __len__(self):
        return len(self.image_list)


class ETH3D(StereoDataset):
    def __init__(self, aug_params=None, root='./data/datasets/ETH3D', split='training'):
        super(ETH3D, self).__init__(aug_params, sparse=True)

        image1_list = sorted(glob(osp.join(root, f'two_view_{split}/*/im0.png')))
        image2_list = sorted(glob(osp.join(root, f'two_view_{split}/*/im1.png')))
        disp_list = sorted(glob(osp.join(root, 'two_view_training_gt/*/disp0GT.pfm'))) \
            if split == 'training' \
            else [osp.join(root,'two_view_training_gt/playground_1l/disp0GT.pfm')] * len(image1_list)

        for img1, img2, disp in zip(image1_list, image2_list, disp_list):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]



class KITTI(StereoDataset):
    def __init__(self, aug_params=None, image_set='training', year=2015):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        if year == 2012:
            root_12 = './data/datasets/kitti12'
            image1_list = sorted(glob(os.path.join(root_12, image_set, 'colored_0/*_10.png')))
            image2_list = sorted(glob(os.path.join(root_12, image_set, 'colored_1/*_10.png')))
            disp_list = sorted(
                glob(os.path.join(root_12, image_set, 'disp_occ/*_10.png')))

        if year == 2015:
            root_15 = './data/datasets/kitti15'
            image1_list = sorted(glob(os.path.join(root_15, image_set, 'image_2/*_10.png')))
            image2_list = sorted(glob(os.path.join(root_15, image_set, 'image_3/*_10.png')))
            disp_list = sorted(
                glob(os.path.join(root_15, image_set, 'disp_occ_0/*_10.png')))

        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]


class Middlebury(StereoDataset):
    def __init__(self, aug_params=None, root='./data/datasets/Middlebury', split='2014', resolution='F'):
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
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"),
                                             str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [str(scene / "disp1.png")]
        elif split == "2006":
            scenes = list((Path(root) / "2006").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "view1.png"), str(scene / "view5.png")]]
                self.disparity_list += [str(scene / "disp1.png")]
                for illum in ["1", "2", "3"]:
                    for exp in ["0", "1", "2"]:
                        self.image_list += [[str(scene / f"Illum{illum}/Exp{exp}/view1.png"),
                                             str(scene / f"Illum{illum}/Exp{exp}/view5.png")]]
                        self.disparity_list += [str(scene / "disp1.png")]
        elif split == "2014":
            scenes = list((Path(root) / "2014").glob("*"))
            for scene in scenes:
                for s in ["E", "L", ""]:
                    self.image_list += [[str(scene / "im0.png"), str(scene / f"im1{s}.png")]]
                    self.disparity_list += [str(scene / "disp0.pfm")]
        elif split == "2021":
            scenes = list((Path(root) / "2021/data").glob("*"))
            for scene in scenes:
                self.image_list += [[str(scene / "im0.png"), str(scene / "im1.png")]]
                self.disparity_list += [str(scene / "disp0.pfm")]
                for s in ["0", "1", "2", "3"]:
                    if os.path.exists(str(scene / f"ambient/L0/im0e{s}.png")):
                        self.image_list += [
                            [str(scene / f"ambient/L0/im0e{s}.png"), str(scene / f"ambient/L0/im1e{s}.png")]]
                        self.disparity_list += [str(scene / "disp0.pfm")]
        else:
            image1_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/im0.png')))
            image2_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/im1.png')))
            disp_list = sorted(glob(os.path.join(root, "MiddEval3", f'training{resolution}', '*/disp0GT.pfm')))
            assert len(image1_list) == len(image2_list) == len(disp_list) > 0, [image1_list, split]
            for img1, img2, disp in zip(image1_list, image2_list, disp_list):
                self.image_list += [[img1, img2]]
                self.disparity_list += [disp]


class DrivingStereoWeather(StereoDataset):
    def __init__(self, aug_params=None, root='./data/datasets/DrivingStereoWeather', image_set='rainy'):
        super(DrivingStereoWeather, self).__init__(aug_params, sparse=True,
                                                   reader=frame_utils.readDispDrivingStereoFull)
        assert os.path.exists(root), f"DrivingStereoWeather root not found: {root}"

        image1_list = sorted(glob(os.path.join(root, image_set, 'left-image-full-size/*.png')))
        image2_list = sorted(glob(os.path.join(root, image_set, 'right-image-full-size/*.png')))
        disp_list = sorted(glob(os.path.join(root, image_set, 'disparity-map-full-size/*.png')))
        for idx, (img1, img2, disp) in enumerate(zip(image1_list, image2_list, disp_list)):
            self.image_list += [[img1, img2]]
            self.disparity_list += [disp]