import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
from nets import Model
import os
from matplotlib import pyplot as plt

device = 'cuda'

#Ref: https://github.com/megvii-research/CREStereo/blob/master/test.py
def inference(left, right, model, n_iter=20):

	print("Model Forwarding...")
	imgL = left.transpose(2, 0, 1)
	imgR = right.transpose(2, 0, 1)
	imgL = np.ascontiguousarray(imgL[None, :, :, :])
	imgR = np.ascontiguousarray(imgR[None, :, :, :])

	imgL = torch.tensor(imgL.astype("float32")).to(device)
	imgR = torch.tensor(imgR.astype("float32")).to(device)

	imgL_dw2 = F.interpolate(
		imgL,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	imgR_dw2 = F.interpolate(
		imgR,
		size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
		mode="bilinear",
		align_corners=True,
	)
	# print(imgR_dw2.shape)
	with torch.inference_mode():
		pred_flow_dw2 = model(imgL_dw2, imgR_dw2, iters=n_iter, flow_init=None)

		pred_flow = model(imgL, imgR, iters=n_iter, flow_init=pred_flow_dw2)
	pred_disp = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

	return pred_disp

if __name__ == '__main__':

	left_img = cv2.imread("E:\\dataset\\ETH3d\\storage_room_2l\\im0.png")
	right_img = cv2.imread("E:\\dataset\\ETH3d\\storage_room_2l\\im1.png")


	in_h, in_w = left_img.shape[:2]
	eval_h, eval_w = (in_h-in_h%8, in_w - in_w%8)

	imgL = cv2.resize(left_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
	imgR = cv2.resize(right_img, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
	# Resize image in case the GPU memory overflows
	assert eval_h%8 == 0, "input height should be divisible by 8"
	assert eval_w%8 == 0, "input width should be divisible by 8"

	model_path = "models/crestereo_eth3d.pth"

	model = Model(max_disp=192, mixed_precision=False, test_mode=True)
	model.load_state_dict(torch.load(model_path), strict=True)
	model.to(device)
	model.eval()
	start=time.time()

	pred = inference(imgL, imgR, model, n_iter=32)

	end=time.time()
	print(str(end-start))


	t = float(in_w) / float(eval_w)
	disp = cv2.resize(pred, (in_w, in_h), interpolation=cv2.INTER_LINEAR) * t
	"存储pfm文件"
	cv2.imwrite("./disp/disp.pfm",disp)
	disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
	filename = os.path.join("demo-output", "disp.png")
	plt.imsave(filename, disp.squeeze(), cmap='jet')
	# combined_img = np.hstack((left_img, disp_vis))
	# cv2.namedWindow("output", cv2.WINDOW_NORMAL)
	# cv2.imshow("output", combined_img)
	cv2.imshow("output", disp_vis)
	# cv2.imshow("disp",disp)
	# _3D_image = cv2.reprojectImageTo3D(disp,Q)
	print(disp_vis.shape)
	cv2.waitKey(0)



