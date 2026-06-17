import cv2
import numpy as np
from typing import Tuple
from matplotlib import pyplot as plt

# 双目相机参数
class stereoCamera(object):
    def __init__(self,param_path : str) -> None:
        self.file = cv2.FileStorage(param_path,cv2.FILE_STORAGE_READ)
        self.Camera_SensorType = self.file.getNode("Camera_SensorType").string()
        # 左相机内参
        self.cam_matrix_left = self.file.getNode("K_l").mat()
        # 右相机内参
        self.cam_matrix_right = self.file.getNode("K_r").mat()

        # 左右相机畸变系数:[k1, k2, p1, p2, k3]
        self.distortion_l = self.file.getNode("D_l").mat()
        self.distortion_r = self.file.getNode("D_r").mat()
        
        # 检查是否存在旋转参数
        ret =  self.file.getNode("R_l").empty() \
            or self.file.getNode("R_r").empty() \
            or self.file.getNode("P_l").empty() \
            or self.file.getNode("P_r").empty() \
            or self.file.getNode("Q").empty()
        if not ret :
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
            ## 获取畸变参数
            self.R1,self.R2,self.P1,self.P2,self.Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
                self.cam_matrix_left,
                self.distortion_l,
                self.cam_matrix_right,
                self.distortion_r,
                (self.width,self.height),
                self.R,
                self.T,
                flags= cv2.CALIB_ZERO_DISPARITY,
                alpha= 0
            )
        # 释放
        self.file.release()
        # 畸变参数获取
        if self.Camera_SensorType == "Fisheye" :
            self.map1x, self.map1y = cv2.fisheye.initUndistortRectifyMap(self.cam_matrix_left, self.distortion_l, self.R1, self.P1, (self.width, self.height), cv2.CV_32FC1)
            self.map2x, self.map2y = cv2.fisheye.initUndistortRectifyMap(self.cam_matrix_right, self.distortion_r, self.R2, self.P2, (self.width, self.height), cv2.CV_32FC1)
        elif self.Camera_SensorType == "Pinhole" :
            self.map1x, self.map1y = cv2.initUndistortRectifyMap(self.cam_matrix_left, self.distortion_l, self.R1, self.P1, (self.width, self.height), cv2.CV_32FC1)
            self.map2x, self.map2y = cv2.initUndistortRectifyMap(self.cam_matrix_right, self.distortion_r, self.R2, self.P2, (self.width, self.height), cv2.CV_32FC1)


    # 矫正
    def rectify(self,left_img :np.ndarray ,right_img: np.ndarray) -> Tuple[np.ndarray , np.ndarray]:
        # print("distortion rectify !")
        # 矫正
        left_rectified = cv2.remap(left_img, self.map1x, self.map1y, interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        right_rectified = cv2.remap(right_img, self.map2x, self.map2y, interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite("rect_left.png",left_rectified)
        cv2.imwrite("rect_right.png",right_rectified)
        return left_rectified,right_rectified
    
    # 转换为点云图
    def transformTo3D(self,disp_img :np.ndarray ,Q:np.ndarray) -> np.ndarray :
        return cv2.reprojectImageTo3D(disp_img,Q)
    
    def cat(self, img1, img2, num_colors: int = 32, line_thickness: int = 2):
        """拼接两张图像，在中间添加JET colorbar风格的彩色分隔线

        Args:
            img1: 左图像
            img2: 右图像
            num_colors: 分隔线间隔（数值越大间隔越宽）
            line_thickness: 分隔线粗细
        """
        size = img1.shape
        height, width = size[:2]

        if img1.ndim == 2:
            img = np.zeros((height, width * 2), dtype=np.uint8)
            img[:, 0:width] = img1
            img[:, width:2 * width] = img2
        else:
            img = np.zeros((height, width * 2, size[2]), dtype=np.uint8)
            img[:, 0:width, :] = img1
            img[:, width:2 * width, :] = img2

        # 使用彩色分隔线 - 根据行位置比例计算颜色
        for i in range(0, height, num_colors):
            # 使用行位置的比例 [0, 255]
            color_idx = int(i / height * 255)
            color = self._get_colorbar_color(color_idx)
            # 绘制粗分隔线
            for t in range(line_thickness):
                row = i + t
                if row < height:
                    if img1.ndim == 2:
                        img[row, :] = color[0]
                    else:
                        img[row, :, :] = color

        return img.astype(np.uint8)

    def _get_colorbar_color(self, idx: int) -> int:
        """获取JET风格colorbar的颜色 (RGB -> BGR)"""
        # JET colormap: 蓝 -> 青 -> 绿 -> 黄 -> 红
        # idx 范围 [0, 255]
        x = idx / 255.0

        if x < 0.25:
            # 蓝 -> 青
            r = 0
            g = int(4 * x * 255)
            b = 255
        elif x < 0.5:
            # 青 -> 绿
            r = 0
            g = 255
            b = int((1 - 4 * (x - 0.25)) * 255)
        elif x < 0.75:
            # 绿 -> 黄
            r = int(4 * (x - 0.5) * 255)
            g = 255
            b = 0
        else:
            # 黄 -> 红
            r = 255
            g = int((1 - 4 * (x - 0.75)) * 255)
            b = 0

        # 转换为 BGR 用于 OpenCV
        return np.array([b, g, r], dtype=np.uint8)
    def Brief(self):
        print("the left K : \n",self.cam_matrix_left)
        print("the right K : \n",self.cam_matrix_right)
        print("the distortion coeffs of left : ",self.distortion_l)
        print("the distortion coeffs of right : ",self.distortion_r)
        if not self.R.size == 0:
            print("the rotation from the left to right : \n",self.R)
            print("the translation from the left to right : \n",self.T)