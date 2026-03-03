from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from models import __models__
from utils import *
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Stereo Matching Prediction with Fast-ACVNet')
parser.add_argument('--model', default='Fast_ACVNet_plus', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--loadckpt', default='pretrained/FastACV++/generalization.ckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--left_img', type=str, default="data/rect_left.png", help='path to left image')
parser.add_argument('--right_img', type=str, default="data/rect_right.png", help='path to right image')
parser.add_argument('--output', type=str, default='disparity_output.png', help='output disparity image path')

# Parse arguments
args = parser.parse_args()

print("Loading model...")
# Load model
model = __models__[args.model](args.maxdisp, False)
model = nn.DataParallel(model)
model.cuda()

# Load pretrained weights
print(f"Loading checkpoint from: {args.loadckpt}")
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])
model.eval()

print("Model loaded successfully!")

# Create output directory if needed
output_dir = os.path.dirname(args.output)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"\nProcessing images:")
print(f"  Left:  {args.left_img}")
print(f"  Right: {args.right_img}")

# Load images
limg = Image.open(args.left_img).convert('RGB')
rimg = Image.open(args.right_img).convert('RGB')

w, h = limg.size
print(f"  Original size: {w}x{h}")

# Pad images to be divisible by 32
wi, hi = (w // 32 + 1) * 32, (h // 32 + 1) * 32

# Crop from bottom-right corner to maintain original image position
limg = limg.crop((w - wi, h - hi, w, h))
rimg = rimg.crop((w - wi, h - hi, w, h))

# Preprocess
limg_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(limg)
rimg_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(rimg)

limg_tensor = limg_tensor.unsqueeze(0).cuda()
rimg_tensor = rimg_tensor.unsqueeze(0).cuda()

print("Running inference...")
with torch.no_grad():
    pred_disp = model(limg_tensor, rimg_tensor)[-1]
    
    # Crop back to original size
    pred_disp = pred_disp[:, hi - h:, wi - w:]

# Convert to numpy
pred_np = pred_disp.squeeze().cpu().numpy()

# Clear GPU cache
torch.cuda.empty_cache()

# Save disparity map
print(f"\nSaving disparity map to: {args.output}")
pred_np_save = np.round(pred_np * 256).astype(np.uint16)
cv2.imwrite(args.output, cv2.applyColorMap(cv2.convertScaleAbs(pred_np_save, alpha=0.01), cv2.COLORMAP_JET), [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

print("Done!")
print(f"Disparity map shape: {pred_np.shape}")
print(f"Disparity range: [{pred_np.min():.2f}, {pred_np.max():.2f}]")