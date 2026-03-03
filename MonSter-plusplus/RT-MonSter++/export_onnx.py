import argparse
import torch
from core.monster import Monster_ONNX
from core.utils.utils import InputPadder
import os



def export(args:argparse.Namespace):
    h, w = args.input_shape
    left :torch.Tensor = torch.randn(1, 3, h, w,dtype=torch.float32).to('cuda')
    right :torch.Tensor = torch.randn(1, 3, h, w,dtype=torch.float32).to('cuda')
    padder = InputPadder(left.shape, divis_by=32)
    left_rected, right_rected = padder.pad(left, right)
    
    # 模型
    print("load the monster stereo matching model !")
    model = torch.nn.DataParallel(Monster_ONNX(args), device_ids=[0])
    
    assert os.path.exists(args.restore_ckpt)
    checkpoint = torch.load(args.restore_ckpt)
    ckpt = dict()
    if 'state_dict' in checkpoint.keys():
        checkpoint = checkpoint['state_dict']
        for key in checkpoint:
            ckpt['module.' + key] = checkpoint[key]
    else:
        state_dict = checkpoint['model']
        for key in state_dict:
            # ckpt['module.' + key] = checkpoint[key]
            if key.startswith("module."):
                ckpt[key] = state_dict[key]  # 保持原样
            else:
                ckpt["module." + key] = state_dict[key]  # 添加 "module."

    model.load_state_dict(ckpt, strict=True)
    model = model.module
    model.to('cuda')
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            (left_rected, right_rected),
            args.onnx_path,
            input_names=["left","right"],
            output_names=["disparity"],
            verbose=True,
            opset_version=17,
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", default='./pretrained/Zero_shot.pth')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=4, help='number of flow-field updates during forward pass')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])

    # Architecture choices
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[32, 64, 96], help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_radius', type=int, default=[2, 2, 4], help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    

    # input
    parser.add_argument("--onnx-path",default="./model.onnx",help="the path of output onnx")
    parser.add_argument("--input-shape",default=(480, 640),nargs=2,help="the shape of input image")

    args = parser.parse_args()

    export(args)
