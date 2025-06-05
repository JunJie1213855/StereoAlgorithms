from glob import glob
import os
import cv2

def kitti():
    index = 0
    # 2012
    imagel_list_2012 = sorted(glob(os.path.join('./data/old/KITTI/KITTI_2012/training/colored_0/*_10.png')))
    imager_list_2012 = sorted(glob(os.path.join('./data/old/KITTI/KITTI_2012/training/colored_1/*_10.png')))
    disp_list_2012 = sorted(glob(os.path.join('./data/old/KITTI/KITTI_2012/training/disp_occ/*_10.png')))
    for i in range(0,len(imagel_list_2012)):
        img_l = cv2.imread(imagel_list_2012[i])
        img_r = cv2.imread(imager_list_2012[i])
        disp = cv2.imread(disp_list_2012[i])
        cv2.imwrite("./data/new/KITTI/{}_left.jpg".format(index),img_l)
        cv2.imwrite("./data/new/KITTI/{}_right.jpg".format(index),img_r)
        cv2.imwrite("./data/new/KITTI/{}_left.disp.png".format(index),disp)
        index+=1
        cv2.imshow("2012_l",img_l)
        cv2.imshow("2012_r",img_r)
        cv2.waitKey(1)
    # 2015
    imagel_list_2015 = sorted(glob(os.path.join('./data/old/KITTI/KITTI_2015/training/image_2/*_10.png')))
    imager_list_2015 = sorted(glob(os.path.join('./data/old/KITTI/KITTI_2015/training/image_3/*_10.png')))
    disp_list_2015 = sorted(glob(os.path.join('./data/old/KITTI/KITTI_2015/training/disp_occ_0/*_10.png')))
    for i in range(0,len(imagel_list_2015)):
        img_l = cv2.imread(imagel_list_2015[i])
        img_r = cv2.imread(imager_list_2015[i])
        disp = cv2.imread(disp_list_2015[i])
        cv2.imwrite("./data/new/KITTI/{}_left.jpg".format(index),img_l)
        cv2.imwrite("./data/new/KITTI/{}_right.jpg".format(index),img_r)
        cv2.imwrite("./data/new/KITTI/{}_left.disp.png".format(index),disp)
        index+=1
        cv2.imshow("2015_l",img_l)
        cv2.imshow("2015_r",img_r)
        cv2.waitKey(1)


def middlebury():
    index = 0
    imagel_list = glob(os.path.join('./data/old/MiddleBury/trainingH/**/im0.png'))
    imager_list = glob(os.path.join('./data/old/MiddleBury/trainingH/**/im1.png'))
    disp_list = sorted(glob(os.path.join('./data/old/MiddleBury/trainingH/**/disp0GT.pfm')))
    for i in range(0,len(imagel_list)):
        img_l = cv2.imread(imagel_list[i])
        img_r = cv2.imread(imager_list[i])
        disp = cv2.imread(disp_list[i])
        cv2.imwrite("./data/new/MiddleBury/{}_left.jpg".format(index),img_l)
        cv2.imwrite("./data/new/MiddleBury/{}_right.jpg".format(index),img_r)
        cv2.imwrite("./data/new/MiddleBury/{}_left.disp.png".format(index),disp)
        index+=1
        cv2.imshow("l",img_l)
        cv2.imshow("r",img_r)
        cv2.waitKey(1)
    pass

if __name__ == "__main__":
    # kitti()
    middlebury()
    
