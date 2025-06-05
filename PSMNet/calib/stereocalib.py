import cv2
import numpy as np
import argparse
import os
import numpy as np
from tqdm import tqdm


"用于显示校正后的双目图像是否对齐"
def cat(img1,img2):
    "左右视图都是一样大小"
    if img1.ndim==2:
        size=img1.shape
        img=np.zeros((size[0],size[1]*2))
        img[:,0:size[1]]=img1
        img[:,size[1]:2*size[1]]=img2
        for i in range(size[0]):
            if i%32==0:
                img[i,:]=0
    else:
        size=img1.shape
        img=np.zeros((size[0],size[1]*2,size[2]))
        img[:,0:size[1],:]=img1
        img[:,size[1]:2*size[1],:]=img2
        for i in range(size[0]):
            if i%32==0:
                img[i,:,:]=0
    return img.astype(np.uint8)

def main(args:argparse.Namespace):
    "左右图像路径文件夹路径"
    left_root_dir = args.left_root_dir
    right_root_dir = args.right_root_dir
    "得到左右图像的文件名"
    left_image_name = os.listdir(left_root_dir)
    right_image_name = os.listdir(right_root_dir)
    "图像的绝对路径"
    left_image_path = [os.path.join(left_root_dir,name) for name in  left_image_name]
    right_image_path = [os.path.join(right_root_dir,name) for name in  right_image_name]
    print("左相机图像数目 : ",len(left_image_path))
    print("右相机图像数目 : ",len(right_image_path))
    # "终止条件"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,100,0)
    "棋盘格的尺寸大小"
    square = args.square_size
    "亚角点寻找半径"
    raduis = args.radius_size
    "棋盘格"
    chessboard = args.board_size
    chessboardsize = chessboard[0]*chessboard[1]
    "世界坐标点"
    objpoints=[]
    "图像坐标点"
    left_imgpoints=[]
    right_imgpoints=[]
    "世界点"
    objp=np.zeros((1,chessboardsize,3),np.float32)
    "创建棋盘格世界坐标"
    objp[0,:,:2]=np.mgrid[:chessboard[0],:chessboard[1]].T.reshape(-1,2)*square
    "棋盘格大小"

    index=0
    for i in tqdm(range(len(left_image_path))):
        "左右图像路径"
        left_path = left_image_path[i]
        right_path = right_image_path[i]
        "读取左右图像"
        left_image:np.ndarray = cv2.imread(left_path)
        right_image:np.ndarray = cv2.imread(right_path)
        if args.img_resizeoption==1:
            left_image = cv2.resize(left_image,None,fx=args.resize_times,fy=args.resize_times,interpolation=cv2.INTER_CUBIC)
            right_image = cv2.resize(right_image,None,fx=args.resize_times,fy=args.resize_times,interpolation=cv2.INTER_CUBIC)
        "灰度图"
        left_gray = cv2.cvtColor(left_image,cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

        ## 对比度增强
        # left_gray = cv2.equalizeHist(left_gray)
        # right_gray = cv2.equalizeHist(right_gray)

        "ret是是否找到角点的标志,corners就是角点集合"
        left_ret, left_corners = cv2.findChessboardCorners(left_gray,chessboard)
        right_ret, right_corners = cv2.findChessboardCorners(right_gray, chessboard)
        message = "未找到" if left_ret == False else "找到了"
        print("左图像路径 : ",left_path)
        print("右图像路径 : ",right_path)
        print("本次的角点 ",message)

        if ((right_ret and left_ret)==True):
            index += 1
            "亚像素角点做细化"
            left_corners  = cv2.cornerSubPix(left_gray,left_corners,raduis,(-1,-1),criteria)
            right_corners = cv2.cornerSubPix(right_gray, right_corners, raduis, (-1, -1), criteria)
            "显示"
            left_image = cv2.drawChessboardCorners(left_image, chessboard, left_corners, right_ret)
            right_image = cv2.drawChessboardCorners(right_image, chessboard, right_corners, right_ret)
            cv2.namedWindow("picure_left", cv2.WINDOW_NORMAL)
            cv2.namedWindow("picure_right", cv2.WINDOW_NORMAL)
            cv2.imshow("picure_left", left_image)
            cv2.imshow("picure_right", right_image)
            if  args.showdetail==1:
                print("just press the esc to next one")
                while(1):
                    if cv2.waitKey(1) == 27:
                        break
            else:
                cv2.waitKey(500)
            "添加世界点"
            objpoints.append(objp)
            "添加角点"
            left_imgpoints.append(left_corners)
            right_imgpoints.append(right_corners)
            # if(cv2.waitKey(10)==27):
            #     break
        else:
            # os.remove(left_image_path[i])
            # os.remove(right_image_path[i])
            # print("已经移除")
            pass
    cv2.destroyAllWindows()
    "整理配对的角点"
    objpoints = np.array(objpoints).reshape(index,1,chessboardsize,3)
    left_imgpoints = np.array(left_imgpoints).reshape(index,1,chessboardsize,2)
    right_imgpoints = np.array(right_imgpoints).reshape(index,1,chessboardsize,2)
    
    "算法使用选项"
    flag = 0
    # flag += cv2.CALIB_FIX_K3 
    # flag +=cv2.CALIB_FIX_PRINCIPAL_POINT
    # flag += cv2.CALIB_FIX_TANGENT_DIST
    # flag +=cv2.CALIB_FIX_TAUX_TAUY
    # flag += cv2.CALIB_RATIONAL_MODEL
    # flag +=cv2.CALIB_ZERO_TANGENT_DIST
    h,w,z = left_image.shape
    
    

    "左相机标定"
    retval, cameraMatrix1, distCoeffs1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints,left_imgpoints,(w,h),cameraMatrix=None,distCoeffs=None,rvecs=None, tvecs=None,flags=flag, criteria=criteria)
    print("左相机误差：%f"%retval)
    "右相机标定"
    retval, cameraMatrix2, distCoeffs2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints,right_imgpoints,(w,h),cameraMatrix=None,distCoeffs=None,rvecs=None, tvecs=None, flags=flag,criteria=criteria)
    print("右相机误差：%f"%retval)
    print("左相机内参 : \n",cameraMatrix1)
    print("右相机内参 : \n",cameraMatrix2)

    print("开始进行双目融合标定,计算双目的相关参数")
    "获取内参矩阵和畸变系数,R,T,E分别为旋转矩阵,平移矩阵,基本矩阵"
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        left_imgpoints,
        right_imgpoints,
        cameraMatrix1=cameraMatrix1,
        distCoeffs1=distCoeffs1,
        cameraMatrix2=cameraMatrix2,
        distCoeffs2=distCoeffs2,
        imageSize=(w,h),
        R=None,
        T=None,
        E=None,
        F=None,
        flags= flag,
        criteria=criteria
    )
    print("双相机误差：%f"%retval)
    print("K1 : \n",cameraMatrix1)
    print("D1 : \n",distCoeffs1)
    print("K2 : \n",cameraMatrix2)
    print("D2 : \n",distCoeffs2)
    print("the rotation matrix transform the left into right is:\n",R)
    print("the translation vector bewteen the O1 t0 O2:\n",T,"\n the base line :",np.linalg.norm(T))

    "找对应的旋转矩阵和新的相机内参矩阵"
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        cameraMatrix1,
        distCoeffs1,
        cameraMatrix2,
        distCoeffs2,
        (w,h),
        R,
        T,
        flags= flag + cv2.CALIB_ZERO_DISPARITY,
        alpha = 0
    )
    "保存数据"
    "cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2"
    "R1, R2, P1, P2, Q"
    if args.save_option ==1:
        file = cv2.FileStorage(args.save_path, cv2.FILE_STORAGE_WRITE)
        print("本次将会把参数保存,并且保存到",args.save_path)
        "写入数据"
        file.write("Camera_SensorType","Pinhole")
        file.write("Camera_NumType","Stereo")
        file.write("K_l", cameraMatrix1)
        file.write("D_l", distCoeffs1)
        file.write("K_r", cameraMatrix2)
        file.write("D_r", distCoeffs2)
        file.write("R", R)
        file.write("t", T)
        file.write("R_l", R1)
        file.write("R_r", R2)
        file.write("P_l", P1)
        file.write("P_r", P2)
        file.write("Q", Q)
        file.write("height",h)
        file.write("width",w)
        file.release()
    else:
        print("本次将不会保存参数")

    "挑选一对左右图像进行矫正,查看标定效果"
    
    left_image=cv2.imread(left_image_path[args.test_index])
    right_image=cv2.imread(right_image_path[args.test_index])
    # left_image=cv2.imread("E:\\dataset\\calibrate\\xiaohua\\data\\left00.jpg")
    # right_image=cv2.imread("E:\\dataset\\calibrate\\xiaohua\\data\\right00.jpg")
    if args.img_resizeoption==1:
        left_image = cv2.resize(left_image,None,fx=args.resize_times,fy=args.resize_times)
        right_image = cv2.resize(right_image,None,fx=args.resize_times,fy=args.resize_times)
    
    h,w = left_image.shape[:2]
    "左右映射图"
    left_map1,left_map2 = cv2.initUndistortRectifyMap(cameraMatrix1,distCoeffs1,R1,P1,(w,h),cv2.CV_32FC1)
    right_map1,right_map2 = cv2.initUndistortRectifyMap(cameraMatrix2,distCoeffs2,R2,P2,(w,h),cv2.CV_32FC1)

    "矫正图"
    left = cv2.remap(left_image, left_map1, left_map2, interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    right = cv2.remap(right_image, right_map1, right_map2, interpolation=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)

    "测试显示"
    picure=cat(left,right)
    cv2.namedWindow("left",cv2.WINDOW_NORMAL)
    cv2.namedWindow("right",cv2.WINDOW_NORMAL)
    cv2.namedWindow("concation",cv2.WINDOW_NORMAL)
    
    # cv2.imshow("left_origin",left_image)
    cv2.imshow("left",left)
    cv2.imshow("right",right)
    cv2.imshow("concation",picure)
    cv2.waitKey(0)

    if args.save_img == 1:
        print("保存图像")
        cv2.imwrite("./rect/left.jpg",left)
        cv2.imwrite("./rect/right.jpg",right)





if __name__=="__main__":
    print("双目标定")
    parse = argparse.ArgumentParser()
    ## 图像根目录
    parse.add_argument("-lr","--left_root_dir",default="E:\\dataset\\CameraCalib\\stereoMatch\\steroexample_tupie\\left")
    parse.add_argument("-rr","--right_root_dir",default="E:\\dataset\\CameraCalib\\stereoMatch\\steroexample_tupie\\right")
    ## 棋盘格查找和算法的一些选项
    parse.add_argument("--board_size",default= (6,8),type = list,help="the root dir of the src image")
    parse.add_argument("--square_size",default= 0.015 ,type= float,help="the size of the small square, and the norm is mm")
    parse.add_argument("--radius_size",default=(7,7),type=list,help = "the radius of corner point finding")
    ## 效果显示的选项
    parse.add_argument("--showdetail",default=0,help="show the corner point more details")
    ## 参数保存
    parse.add_argument("-so","--save_option",default=0,type= int,help= "save the intrinsic parameters")
    parse.add_argument("-sp","--save_path",default="./param/usbcamera.yaml",help="the path of xml or yaml file,which involve the parameters")
    ## 去畸变示例和保存畸变矫正图像
    parse.add_argument("--test_index",default=3,type=int,help="to show the result")
    parse.add_argument("--save_img",default=0,type=int,help="save the no distortion image ?")
    ## 图像尺寸重定义,为了适配立体匹配算法
    parse.add_argument("--img_resizeoption",default=0,help="the resize option tag")
    parse.add_argument("--resize_times",default=0.5,type=float,help= "the resize of image")
    
    args = parse.parse_args()
    main(args)