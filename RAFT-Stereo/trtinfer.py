# 导入必用依赖
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import cv2
import ctypes
from matplotlib import pyplot as plt
import pycuda.autoinit
import argparse



ctypes.CDLL("E:\\lib\\TensorRT-8.6.1.6\\lib\\nvinfer_plugin.dll", mode=ctypes.RTLD_GLOBAL)
trt.init_libnvinfer_plugins(None, "")
def visual(disp:np.ndarray):
    disp_vis = (disp - np.min(disp)) / (np.max(disp) - np.min(disp)) * 255
    disp_vis = disp_vis.astype(np.uint8)
    return cv2.applyColorMap(disp_vis,cv2.COLORMAP_INFERNO)

def main(args:argparse.Namespace):
    # 创建logger：日志记录器
    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine_file, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
    with engine.create_execution_context() as context:
        image1 = cv2.imread(args.left_path)
        image2 = cv2.imread(args.right_path)

        image1 = cv2.resize(image1, (args.img_size[1], args.img_size[0]))
        image2 = cv2.resize(image2, (args.img_size[1], args.img_size[0]))

        image1 = image1.transpose(2, 0, 1).astype(np.float32)
        image2 = image2.transpose(2, 0, 1).astype(np.float32)

        output_shape = (args.img_size[0],args.img_size[1])
        output_data = np.empty(output_shape, dtype=np.float32)

        # 在GPU上分配内存
        d_input_left = cuda.mem_alloc(image1.nbytes)
        d_input_right = cuda.mem_alloc(image2.nbytes)
        d_output = cuda.mem_alloc(output_data.nbytes)

        # 将输入数据复制到GPU内存
        image1 = np.ascontiguousarray(image1,image1.dtype)
        image2 = np.ascontiguousarray(image2,image2.dtype)
        cuda.memcpy_htod(d_input_left, image1)
        cuda.memcpy_htod(d_input_right, image2)
        # 运行推理
        context.execute_v2(bindings=[(d_input_left),(d_input_right), (d_output)])

        # 将输出数据从GPU内存复制回主机
        cuda.memcpy_dtoh(output_data, d_output)
        cv2.imshow("disparity visualization",visual(output_data))
        cv2.waitKey()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size",default=(352,640),type = int,nargs=2,help="the image size,(height,width)")
    parser.add_argument("--engine_file",default="raft.engine",type=str,help="the engine file")
    parser.add_argument("--left_path",default="rect/leftc.jpg",help="the path of left image")
    parser.add_argument("--right_path",default="rect/rightc.jpg",help="the path of left image")
    parser.add_argument("--output_dir",default="output",help="the output directory of output image")
    args = parser.parse_args()
    main(args)