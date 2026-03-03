from __future__ import print_function, division
import sys
sys.path.append('core')

import numpy as np
import torch
from core.liteanystereo import LiteAnyStereo
from core.utils.utils import InputPadder
from PIL import Image
from Utils import *
import trimesh


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--stereo_file', default='./assets/Explorer_HD2K_SN28883284_20-36-55.png', type=str)
  parser.add_argument('--restore_ckpt', default='pretrained/LiteAnyStereo.pth', type=str, help='pretrained model path')
  parser.add_argument('--out_dir', default='./output-vis/20-36-55', type=str, help='the directory to save results')
  parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
  parser.add_argument('--z_far', default=3, type=float, help='max depth to clip in point cloud')
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
  parser.add_argument('--denoise_cloud', type=int, default=0, help='whether to denoise the point cloud')
  parser.add_argument('--denoise_nb_points', type=int, default=10, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.05, help='radius to use for outlier removal')
  args = parser.parse_args()

  # Left camera intrinsics
  K_left = np.array([
    [1068.6300, 0.0, 1097.8500],
    [0.0, 1068.7900, 632.3400],
    [0.0, 0.0, 1.0]
  ])

  # Right camera intrinsics
  K_right = np.array([
    [1065.8900, 0.0, 1112.9301],
    [0.0, 1065.9100, 631.8750],
    [0.0, 0.0, 1.0]
  ])

  baseline = 0.1201340  # m

  torch.autograd.set_grad_enabled(False)
  os.makedirs(args.out_dir, exist_ok=True)

  ckpt_dir = args.restore_ckpt
  logging.info(f"Using pretrained model from {ckpt_dir}")

  model = LiteAnyStereo()
  logging.basicConfig(level=logging.INFO,
                      format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

  if args.restore_ckpt is not None:
    assert args.restore_ckpt.endswith(".pth")
    logging.info("Loading checkpoint...")
    checkpoint = torch.load(args.restore_ckpt)

    target_model = model.module if hasattr(model, 'module') else model
    target_model.load_state_dict(checkpoint, strict=True)
    logging.info(f"Done loading checkpoint")

  model.cuda()
  model.eval()

  code_dir = os.path.dirname(os.path.realpath(__file__))
  img = imageio.imread(args.stereo_file)[..., :3]  # keep RGB only
  # img1 = imageio.imread(args.right_file)[..., :3]  # keep RGB only

  scale = args.scale
  assert scale<=1, "scale must be <=1"
  img = cv2.resize(img, fx=scale, fy=scale, dsize=None)

  H,W = img.shape[:2]
  img0 = img[:, :W//2, :]
  img1 = img[:, W//2:, :]

  # Convert halves to PIL Images
  im0_pil = Image.fromarray(img0)
  im1_pil = Image.fromarray(img1)

  # Save as a two-frame blinking GIF
  output_gif = os.path.join(args.out_dir, "img.gif")
  frames = [im0_pil, im1_pil]

  frames[0].save(
    output_gif,
    format="GIF",
    save_all=True,
    append_images=frames[1:],
    duration=[500, 500],  # per-frame durations (ms)
    loop=0,  # loop forever
    disposal=2,  # replace previous frame (no overlay)
    optimize=False  # keep frames separate
  )
  print("Saved:", output_gif)

  print(f"GIF saved at {output_gif}")

  img0_ori = img0.copy()
  logging.info(f"img0: {img0.shape}")

  img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
  img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
  padder = InputPadder(img0.shape, divis_by=32)
  img0, img1 = padder.pad(img0, img1)
  disp = model(img0, img1, test_mode=True)
  disp = padder.unpad(disp.float())
  disp = disp.data.cpu().numpy().reshape(H,W//2)
  vis = vis_disparity(disp)
  vis = np.concatenate([img0_ori, vis], axis=1)
  imageio.imwrite(f'{args.out_dir}/vis.png', vis)
  logging.info(f"Output saved to {args.out_dir}")

  if args.remove_invisible:
    yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx-disp
    invalid = us_right<0
    disp[invalid] = np.inf

  if args.get_pc:
    K_left[:2] *= scale
    depth = K_left[0, 0] * baseline / disp
    # np.save(f'{args.out_dir}/depth_meter.npy', depth)
    xyz_map = depth2xyzmap(depth, K_left)
    pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
    keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.z_far)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
    logging.info(f"PCL saved to {args.out_dir}")

    if args.denoise_cloud:
      logging.info("denoise point cloud...")
      cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
      inlier_cloud = pcd.select_by_index(ind)
      o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
      pcd = inlier_cloud

    # --- Save as GLB (glTF 2.0, POINTS) ---
    points = np.asarray(pcd.points)  # (N, 3)

    # Colors in Open3D are typically float in [0,1]; GLB expects either [0,1] floats or uint8.
    cols = np.asarray(pcd.colors) if len(pcd.colors) > 0 else None
    if cols is not None:
      if cols.dtype != np.uint8:
        cols = (np.clip(cols, 0.0, 1.0) * 255).astype(np.uint8)  # (N, 3) uint8

    # Build a trimesh point cloud and export to .glb
    cloud = trimesh.points.PointCloud(vertices=points, colors=cols)
    scene = trimesh.Scene()
    scene.add_geometry(cloud)
    glb_name = "cloud_denoise.glb" if args.denoise_cloud else "cloud.glb"
    glb_path = os.path.join(args.out_dir, glb_name)
    scene.export(glb_path)
    logging.info(f"GLB saved to {glb_path}")

    logging.info("Visualizing point cloud. Press ESC to exit.")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 0.5
    vis.get_render_option().background_color = np.array([1, 1, 1])
    vis.run()
    vis.destroy_window()

