# 导入必用依赖
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import cv2
import ctypes
from matplotlib import pyplot as plt
import pycuda.autoinit
from time import time


ctypes.CDLL(r"E:\\lib\\TensorRT-8.6.1.6\\lib\\nvinfer_plugin.dll", mode=ctypes.RTLD_GLOBAL)

trt.init_libnvinfer_plugins(None, "")

# 假设 'engine_path' 是你的序列化引擎文件路径
engine_path = 'raft.engine'

# 创建一个 TensorRT 运行时实例
runtime = trt.Runtime(trt.Logger())

# 从序列化文件加载引擎
with open(engine_path, 'rb') as f:
    engine_data = f.read()
engine = runtime.deserialize_cuda_engine(engine_data)

# 获取输入和输出的数量
num_bindings = engine.num_bindings

# 打印输入和输出的维度
for i in range(num_bindings):  # 假设每个输入和输出都有一个对应的绑定
    # 获取输入或输出的名称
    name = engine.get_binding_name(i)
    # 获取维度信息
    shape = engine.get_binding_shape(i)
    print(f"{name}: {shape}")