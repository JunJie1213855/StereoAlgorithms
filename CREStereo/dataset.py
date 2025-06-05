import os
import cv2
import glob
import numpy as np
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
# from megengine.data.dataset import Dataset


class Augmentor:
    def __init__(
        self,
        image_height=384,
        image_width=512,
        max_disp=256,
        scale_min=0.6,
        scale_max=1.0,
        seed=0,
    ):
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.max_disp = max_disp
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rng = np.random.RandomState(seed)

    def chromatic_augmentation(self, img):
        random_brightness = np.random.uniform(0.8, 1.2)
        random_contrast = np.random.uniform(0.8, 1.2)
        random_gamma = np.random.uniform(0.8, 1.2)

        img = Image.fromarray(img)

        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random_brightness)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random_contrast)

        gamma_map = [
            255 * 1.0 * pow(ele / 255.0, random_gamma) for ele in range(256)
        ] * 3
        img = img.point(gamma_map)  # use PIL's point-function to accelerate this part

        img_ = np.array(img)

        return img_

    def __call__(self, left_img, right_img, left_disp):
        # 1. chromatic augmentation
        left_img = self.chromatic_augmentation(left_img)
        right_img = self.chromatic_augmentation(right_img)

        # 2. spatial augmentation
        # 2.1) rotate & vertical shift for right image
        if self.rng.binomial(1, 0.5):
            angle, pixel = 0.1, 2
            px = self.rng.uniform(-pixel, pixel)
            ag = self.rng.uniform(-angle, angle)
            image_center = (
                self.rng.uniform(0, right_img.shape[0]),
                self.rng.uniform(0, right_img.shape[1]),
            )
            rot_mat = cv2.getRotationMatrix2D(image_center, ag, 1.0)
            right_img = cv2.warpAffine(
                right_img, rot_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )
            trans_mat = np.float32([[1, 0, 0], [0, 1, px]])
            right_img = cv2.warpAffine(
                right_img, trans_mat, right_img.shape[1::-1], flags=cv2.INTER_LINEAR
            )

        # 2.2) random resize
        resize_scale = self.rng.uniform(self.scale_min, self.scale_max)

        left_img = cv2.resize(
            left_img,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )
        right_img = cv2.resize(
            right_img,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )

        disp_mask = (left_disp < float(self.max_disp / resize_scale)) & (left_disp > 0)
        disp_mask = disp_mask.astype("float32")
        disp_mask = cv2.resize(
            disp_mask,
            None,
            fx=resize_scale,
            fy=resize_scale,
            interpolation=cv2.INTER_LINEAR,
        )

        left_disp = (
            cv2.resize(
                left_disp,
                None,
                fx=resize_scale,
                fy=resize_scale,
                interpolation=cv2.INTER_LINEAR,
            )
            * resize_scale
        )

        # 2.3) random crop
        h, w, c = left_img.shape
        dx = w - self.image_width
        dy = h - self.image_height
        dy = self.rng.randint(min(0, dy), max(0, dy) + 1)
        dx = self.rng.randint(min(0, dx), max(0, dx) + 1)

        M = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
        left_img = cv2.warpAffine(
            left_img,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        right_img = cv2.warpAffine(
            right_img,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        left_disp = cv2.warpAffine(
            left_disp,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        disp_mask = cv2.warpAffine(
            disp_mask,
            M,
            (self.image_width, self.image_height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )

        # 3. add random occlusion to right image
        if self.rng.binomial(1, 0.5):
            sx = int(self.rng.uniform(50, 100))
            sy = int(self.rng.uniform(50, 100))
            cx = int(self.rng.uniform(sx, right_img.shape[0] - sx))
            cy = int(self.rng.uniform(sy, right_img.shape[1] - sy))
            right_img[cx - sx : cx + sx, cy - sy : cy + sy] = np.mean(
                np.mean(right_img, 0), 0
            )[np.newaxis, np.newaxis]

        return left_img, right_img, left_disp, disp_mask

"需要左右图以及左右图的视差图"
class CREStereoDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.imgs = glob.glob(os.path.join(root, "**/*_left.jpg"), recursive=True)
        print(self.imgs)
        self.augmentor = Augmentor(
            image_height=384,
            image_width=512,
            max_disp=256,
            scale_min=0.6,
            scale_max=1.0,
            seed=0,
        )
        self.rng = np.random.RandomState(0)
    "获取视差图"
    def get_disp(self, path):
        disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return disp.astype(np.float32) / 32

    def __getitem__(self, index):
        # find path
        "寻找你的图像路径"
        left_path = self.imgs[index]
        "利用左图获取图像的名称"
        prefix = left_path[: left_path.rfind("_")]

        "这里要注意了，其实这里我们可以自己设置如何读取图像，不必按照作者给的路径读取"
        right_path = prefix + "_right.jpg"
        left_disp_path = prefix + "_left_disp.png"
        right_disp_path = prefix + "_right_disp.png"

        "读取图像"
        # read img, disp
        left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
        right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)
        left_disp = self.get_disp(left_disp_path)
        right_disp = self.get_disp(right_disp_path)

        if self.rng.binomial(1, 0.5):
            left_img, right_img = np.fliplr(right_img), np.fliplr(left_img)
            left_disp, right_disp = np.fliplr(right_disp), np.fliplr(left_disp)
        left_disp[left_disp == np.inf] = 0

        # augmentaion
        left_img, right_img, left_disp, disp_mask = self.augmentor(
            left_img, right_img, left_disp
        )

        left_img = left_img.transpose(2, 0, 1).astype("uint8")
        right_img = right_img.transpose(2, 0, 1).astype("uint8")

        return {
            "left": left_img,
            "right": right_img,
            "disparity": left_disp,
            "mask": disp_mask,
        }

    def __len__(self):
        return len(self.imgs)

"自己的数据"
class MyCREStereoDataset(Dataset):
    def __init__(self, root_dir,left_file,right_file,left_disp_file,right_disp_file):
        super().__init__()
        "root_dir:数据的文件目录的路径"
        "left_file:左图像的文件夹名称"
        "right_file:右图像的文件夹名称"
        "left_disp_file:左视差图像的文件夹名称"
        "right_disp_file:右视差图像的文件夹名称"
        "例如 E:\\dataset\\left\\left_01.png"
        "那么root_dir=E:\\dataset;left_file=left即可"

        "文件名"
        self.root_dir = root_dir
        self.left_file = left_file
        self.right_file = right_file
        self.left_disp_file = left_disp_file
        self.right_disp_file = right_disp_file
        "文件路径"
        self.left_path = os.path.join(self.root_dir,self.left_file)
        self.right_path = os.path.join(self.root_dir, self.right_file)
        self.left_disp_path = os.path.join(self.root_dir, self.left_disp_file)
        self.right_disp_path = os.path.join(self.root_dir, self.right_disp_file)
        "内部图片名称"
        self.left_name = os.listdir(self.left_path)
        self.right_name = os.listdir(self.right_path)
        self.left_disp_name = os.listdir(self.left_disp_path)
        self.right_disp_name = os.listdir(self.right_disp_path)
        "左右图和左右视差图的路径"
        self.left_img_path = [os.path.join(self.left_path,name)for name in self.left_name]
        self.right_img_path = [os.path.join(self.right_path, name) for name in self.right_name]
        self.disp_l_path = [os.path.join(self.left_disp_path, name) for name in self.left_disp_name]
        self.disp_r_path = [os.path.join(self.right_disp_path, name) for name in self.right_disp_name]

        self.augmentor = Augmentor(
            image_height=384,
            image_width=512,
            max_disp=256,
            scale_min=0.6,
            scale_max=1.0,
            seed=0,
        )
        self.rng = np.random.RandomState(0)
    "获取视差图"
    def get_disp(self, path):
        disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return disp.astype(np.float32) / 32

    def __getitem__(self, index):
        "这里要注意了，其实这里我们可以自己设置如何读取图像，不必按照作者给的路径读取"
        left_path = self.left_img_path[index]
        right_path = self.right_img_path[index]
        left_disp_path = self.disp_l_path[index]
        right_disp_path = self.disp_r_path[index]
        "读取图像"
        # read img, disp
        left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
        right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)
        left_disp = self.get_disp(left_disp_path)
        right_disp = self.get_disp(right_disp_path)

        if self.rng.binomial(1, 0.5):
            left_img, right_img = np.fliplr(right_img), np.fliplr(left_img)
            left_disp, right_disp = np.fliplr(right_disp), np.fliplr(left_disp)
        left_disp[left_disp == np.inf] = 0

        # augmentaion
        left_img, right_img, left_disp, disp_mask = self.augmentor(
            left_img, right_img, left_disp
        )

        left_img = left_img.transpose(2, 0, 1).astype("uint8")
        right_img = right_img.transpose(2, 0, 1).astype("uint8")

        return {
            "left": left_img,
            "right": right_img,
            "disparity": left_disp,
            "mask": disp_mask,
        }

    def __len__(self):
        return len(self.left_img_path)

"KITTI数据集"
class KITTI(Dataset):
    def __init__(self, root_dir,left_file,right_file,left_disp_file,right_disp_file):
        super().__init__()
        "文件名"
        self.root_dir = root_dir
        self.left_file = left_file
        self.right_file = right_file
        self.left_disp_file = left_disp_file
        self.right_disp_file = right_disp_file
        "文件路径"
        self.left_path = os.path.join(self.root_dir,self.left_file)
        self.right_path = os.path.join(self.root_dir, self.right_file)
        self.left_disp_path = os.path.join(self.root_dir, self.left_disp_file)
        self.right_disp_path = os.path.join(self.root_dir, self.right_disp_file)
        "左右图和左右视差图的路径"
        self.left_img_path = glob.glob(os.path.join(self.left_path,"*_10.png"))
        self.right_img_path = glob.glob(os.path.join(self.right_path,"*_10.png"))
        self.disp_l_path = glob.glob(os.path.join(self.left_disp_path,"*_10.png"))
        self.disp_r_path = glob.glob(os.path.join(self.right_disp_path,"*_10.png"))

        self.augmentor = Augmentor(
            image_height=384,
            image_width=512,
            max_disp=256,
            scale_min=0.6,
            scale_max=1.0,
            seed=0,
        )
        self.rng = np.random.RandomState(0)
    "获取视差图"
    def get_disp(self, path):
        disp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return disp.astype(np.float32) / 32

    def __getitem__(self, index):
        "这里要注意了，其实这里我们可以自己设置如何读取图像，不必按照作者给的路径读取"
        left_path = self.left_img_path[index]
        right_path = self.right_img_path[index]
        left_disp_path = self.disp_l_path[index]
        right_disp_path = self.disp_r_path[index]
        "读取图像"
        # read img, disp
        left_img = cv2.imread(left_path, cv2.IMREAD_COLOR)
        right_img = cv2.imread(right_path, cv2.IMREAD_COLOR)
        left_disp = self.get_disp(left_disp_path)
        right_disp = self.get_disp(right_disp_path)

        if self.rng.binomial(1, 0.5):
            left_img, right_img = np.fliplr(right_img), np.fliplr(left_img)
            left_disp, right_disp = np.fliplr(right_disp), np.fliplr(left_disp)
        left_disp[left_disp == np.inf] = 0

        # augmentaion
        left_img, right_img, left_disp, disp_mask = self.augmentor(
            left_img, right_img, left_disp
        )

        left_img = left_img.transpose(2, 0, 1).astype("uint8")
        right_img = right_img.transpose(2, 0, 1).astype("uint8")

        return {
            "left": left_img,
            "right": right_img,
            "disparity": left_disp,
            "mask": disp_mask,
        }

    def __len__(self):
        return len(self.left_img_path)