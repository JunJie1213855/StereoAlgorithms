import sys
sys.path.append('core')
import os
import argparse
import glob
import numpy as np
import cv2
import time
import onnxruntime as ort
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt


def load_image(imfile):
    img: np.ndarray = cv2.imread(imfile).astype(np.uint8)
    img = cv2.resize(img, (480, 752))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1).astype(np.float32)
    return img[None]


def pad_image(img, divis_by=32):
    """Pad image to be divisible by 32"""
    h, w = img.shape[2], img.shape[3]
    pad_h = (divis_by - h % divis_by) % divis_by
    pad_w = (divis_by - w % divis_by) % divis_by
    if pad_h > 0 or pad_w > 0:
        img = np.pad(img, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    return img, (h, w)


def unpad_image(img, original_shape):
    """Remove padding from image"""
    h, w = original_shape
    return img[:, :, :h, :w]


def demo(args: argparse.Namespace):
    # Create ONNX Runtime session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.use_cuda else ['CPUExecutionProvider']
    session = ort.InferenceSession(args.onnx_path, providers=providers)

    # Get input/output names
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    print(f"Input names: {input_names}")
    print(f"Output names: {output_names}")

    # Check input shape
    input_info = session.get_inputs()[0]
    print(f"Input shape: {input_info.shape}, dtype: {input_info.type}")

    # Create output directory
    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    print("begin to calculate !")
    left_images = sorted(glob.glob(args.left_imgs, recursive=True))
    right_images = sorted(glob.glob(args.right_imgs, recursive=True))
    print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

    for i in range(len(left_images)):
        print("path -->  left :", left_images[i], "right :", right_images[i])

    for (imfile1, imfile2) in zip(left_images, right_images):
        # Load and preprocess images
        left_img = load_image(imfile1)
        right_img = load_image(imfile2)

        # Pad images
        left_padded, original_shape = pad_image(left_img, divis_by=32)
        right_padded, _ = pad_image(right_img, divis_by=32)

        # Run inference
        start = time.time()
        outputs = session.run(output_names, {
            input_names[0]: left_padded,
            input_names[1]: right_padded
        })
        end = time.time()
        print("the consume time :", end - start)

        # Get disparity
        disp = outputs[0]

        # Unpad disparity
        disp = unpad_image(disp, original_shape)

        print(disp.squeeze().shape)

        # Save disparity
        filename = os.path.join(output_directory, "disp.png")
        plt.imsave(filename, disp.squeeze(), cmap='jet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', help="path to ONNX model", default='./model_752_480.onnx')
    parser.add_argument('--use_cuda', action='store_true', help='use CUDA execution provider')
    parser.add_argument('--output_directory', help="directory to save output", default="./demo-output-onnx/")

    # Input images
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frame", default="./rect_left.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frame", default="./rect_right.png")

    args = parser.parse_args()

    Path(args.output_directory).mkdir(exist_ok=True, parents=True)

    demo(args)
