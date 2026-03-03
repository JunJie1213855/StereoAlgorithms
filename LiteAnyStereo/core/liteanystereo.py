import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .submodule import *
from .fnet import FeatureNet
from .aggregation import Aggregation2D


class LiteAnyStereo(nn.Module):
    def __init__(self):
        super(LiteAnyStereo, self).__init__()
        self.fnet = FeatureNet()

        self.cost_agg_2d = Aggregation2D(in_channels=48,
                                    left_att=True,
                                    blocks=[4, 8, 16],
                                    expanse_ratio=4,
                                    backbone_channels=[24, 32, 96, 160])

        self.cost_stem_3d = nn.Sequential(BasicConv(1, 4, is_3d=True, kernel_size=3, stride=1, padding=1),
                                          BasicConv(4, 4, is_3d=True, kernel_size=3, stride=1, padding=1),
                                          BasicConv(4, 1, is_3d=True, kernel_size=3, stride=1, padding=1),
                                          )

        # disp refine
        self.refine_1 = nn.Sequential(
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(24, 24, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU))

        self.stem_2 = nn.Sequential(
            BasicConv2d(3, 16, kernel_size=3, stride=2, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.LeakyReLU),
            BasicConv2d(16, 16, kernel_size=3, stride=1, padding=1,
                        norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU))
        self.refine_2 = FPNLayer(24, 16)

        self.refine_3 = BasicDeconv2d(16, 9, kernel_size=4, stride=2, padding=1)

    def upsample_disp(self, disp, mask, scale=4):
        """ Upsample disp field [H//4, W//4] -> [H, W] using convex combination """
        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(scale * disp, [3, 3], padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, scale * H, scale * W)

    def forward(self, left, right, max_disp=192, test_mode=False, kd_mode=False):
        left = (2 * (left / 255.0) - 1.0).contiguous()
        right = (2 * (right / 255.0) - 1.0).contiguous()

        features_left = self.fnet(left)
        features_right = self.fnet(right)
        cost_volume = build_correlation_volume(features_left[0], features_right[0], max_disp // 4)

        cv_3d = self.cost_stem_3d(cost_volume[:,None]).squeeze(1)

        cv = self.cost_agg_2d(cv_3d, features_left)

        prob = F.softmax(cv, dim=1)
        disp = disparity_regression(prob, max_disp // 4)

        xspx = self.refine_1(features_left[0])
        xspx = self.refine_2(xspx, self.stem_2(left))
        xspx = self.refine_3(xspx)
        spx_pred = F.softmax(xspx, 1)
        disp_up = context_upsample(disp * 4., spx_pred.float())

        if test_mode:
            return disp_up
        elif kd_mode:
            disp_linear = F.interpolate(disp, left.shape[2:], mode='bilinear', align_corners=False)
            return [disp_up, disp_linear * 4.], features_left, features_right
        else:
            disp_linear = F.interpolate(disp, left.shape[2:], mode='bilinear', align_corners=False)
            return [disp_up, disp_linear * 4.]
