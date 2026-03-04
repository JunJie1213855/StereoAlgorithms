import cv2
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt

# 双目相机参数
class stereoCamera(object):
    def __init__(self, param_path: str, baseline: float = None) -> None:
        """
        加载立体相机参数

        Args:
            param_path: YAML/XML文件路径或K.txt文件路径
            baseline: 可选的baseline值（米），如果使用K.txt文件则需要提供
        """
        self.param_path = param_path
        self.baseline = baseline

        # 判断文件类型
        if param_path.endswith('.txt'):
            # 使用K.txt格式（FoundationStereo风格）
            self._load_from_txt(param_path, baseline)
        elif param_path.endswith('.yaml') or param_path.endswith('.yml') or param_path.endswith('.xml'):
            # 使用YAML/XML格式（传统立体相机标定）
            self._load_from_yaml_xml(param_path)
        else:
            raise ValueError(f"Unsupported file format: {param_path}. Please use .txt, .yaml, or .xml")

    def _load_from_txt(self, txt_path: str, baseline: float):
        """从K.txt文件加载相机参数（FoundationStereo风格）"""
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.cam_matrix_left = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
            self.baseline = float(lines[1]) if baseline is None else baseline

        # 右相机内参（假设与左相机相同）
        self.cam_matrix_right = self.cam_matrix_left.copy()

        # 畸变系数（假设无畸变）
        self.distortion_l = np.zeros((5, 1))
        self.distortion_r = np.zeros((5, 1))

        # 图像尺寸（从K矩阵推断或使用默认值）
        self.width = int(self.cam_matrix_left[1, 2] * 2)  # 假设cx在图像中心
        self.height = int(self.cam_matrix_left[1, 2] * 2)

        # 传感器类型
        self.Camera_SensorType = "Pinhole"

        # 创建虚拟的R, T矩阵
        self.R = np.eye(3)
        self.T = np.array([[self.baseline], [0], [0]])

        # 创建Q矩阵用于reprojectImageTo3D
        self.Q = np.zeros((4, 4), dtype=np.float32)
        self.Q[0, 0] = 1.0
        self.Q[0, 3] = -self.cam_matrix_left[0, 2]
        self.Q[1, 1] = 1.0
        self.Q[1, 3] = -self.cam_matrix_left[1, 2]
        self.Q[2, 3] = self.cam_matrix_left[0, 0]
        self.Q[3, 2] = -1.0 / self.baseline

        # 对于K.txt格式，假设已经矫正，不需要remap
        self.map1x = None
        self.map1y = None
        self.map2x = None
        self.map2y = None

        print(f"Loaded camera parameters from {txt_path}")
        print(f"K matrix:\n{self.cam_matrix_left}")
        print(f"Baseline: {self.baseline} m")

    def _load_from_yaml_xml(self, param_path: str):
        """从YAML/XML文件加载立体相机参数（传统风格）"""
        self.file = cv2.FileStorage(param_path, cv2.FILE_STORAGE_READ)
        self.Camera_SensorType = self.file.getNode("Camera_SensorType").string()
        # 左相机内参
        self.cam_matrix_left = self.file.getNode("K_l").mat()
        # 右相机内参
        self.cam_matrix_right = self.file.getNode("K_r").mat()

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = self.file.getNode("D_l").mat()
        self.distortion_r = self.file.getNode("D_r").mat()

        # 检查是否存在旋转参数
        ret = self.file.getNode("R_l").empty() \
            or self.file.getNode("R_r").empty() \
            or self.file.getNode("P_l").empty() \
            or self.file.getNode("P_r").empty() \
            or self.file.getNode("Q").empty()
        if not ret:
            # 旋转矩阵
            self.R1 = self.file.getNode("R_l").mat()
            self.R2 = self.file.getNode("R_r").mat()
            # 平移矩阵
            self.P1 = self.file.getNode("P_l").mat()
            self.P2 = self.file.getNode("P_r").mat()
            # 重投影矩阵
            self.Q = self.file.getNode("Q").mat()

        # 相机的行列信息
        self.height = int(self.file.getNode("height").real())
        self.width = int(self.file.getNode("width").real())

        # 是否存在旋转和平移
        ret = self.file.getNode("R").empty() or self.file.getNode("t").empty()
        if not ret:
            self.R = self.file.getNode("R").mat()
            self.T = self.file.getNode("t").mat()
            # 从T矩阵计算baseline
            self.baseline = abs(self.T[0, 0])
            # 获取畸变参数
            self.R1, self.R2, self.P1, self.P2, self.Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
                self.cam_matrix_left,
                self.distortion_l,
                self.cam_matrix_right,
                self.distortion_r,
                (self.width, self.height),
                self.R,
                self.T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0
            )
        # 释放
        self.file.release()

        # 畸变参数获取
        if self.Camera_SensorType == "Fisheye":
            self.map1x, self.map1y = cv2.fisheye.initUndistortRectifyMap(
                self.cam_matrix_left, self.distortion_l, self.R1, self.P1, (self.width, self.height), cv2.CV_32FC1)
            self.map2x, self.map2y = cv2.fisheye.initUndistortRectifyMap(
                self.cam_matrix_right, self.distortion_r, self.R2, self.P2, (self.width, self.height), cv2.CV_32FC1)
        elif self.Camera_SensorType == "Pinhole":
            self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                self.cam_matrix_left, self.distortion_l, self.R1, self.P1, (self.width, self.height), cv2.CV_32FC1)
            self.map2x, self.map2y = cv2.initUndistortRectifyMap(
                self.cam_matrix_right, self.distortion_r, self.R2, self.P2, (self.width, self.height), cv2.CV_32FC1)

    def rectify(self, left_img: np.ndarray, right_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """图像矫正"""
        # 如果使用K.txt格式（假设已矫正），直接返回
        if self.map1x is None:
            return left_img, right_img

        # 矫正
        left_rectified = cv2.remap(left_img, self.map1x, self.map1y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        right_rectified = cv2.remap(right_img, self.map2x, self.map2y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite("rect_left.png", left_rectified)
        cv2.imwrite("rect_right.png", right_rectified)
        return left_rectified, right_rectified

    def transformTo3D(self, disp_img: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """转换为点云图"""
        return cv2.reprojectImageTo3D(disp_img, Q)

    def cat(self, img1, img2):
        """拼接左右图像用于可视化"""
        if img1.ndim == 2:
            size = img1.shape
            img = np.zeros((size[0], size[1] * 2))
            img[:, 0:size[1]] = img1
            img[:, size[1]:2 * size[1]] = img2
            for i in range(size[0]):
                if i % 32 == 0:
                    img[i, :] = 0
        else:
            size = img1.shape
            img = np.zeros((size[0], size[1] * 2, size[2]))
            img[:, 0:size[1], :] = img1
            img[:, size[1]:2 * size[1], :] = img2
            for i in range(size[0]):
                if i % 32 == 0:
                    img[i, :, :] = 0
        return img.astype(np.uint8)

    def Brief(self):
        """打印相机参数信息"""
        print("the left K : \n", self.cam_matrix_left)
        print("the right K : \n", self.cam_matrix_right)
        print("the distortion coeffs of left : ", self.distortion_l)
        print("the distortion coeffs of right : ", self.distortion_r)
        if hasattr(self, 'R') and self.R.size > 0:
            print("the rotation from the left to right : \n", self.R)
            print("the translation from the left to right : \n", self.T)
