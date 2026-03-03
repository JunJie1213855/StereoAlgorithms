import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler
from core.submodule import get_warped_feats

class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, radius=4):  #init_corr, 
        self.radius = radius

        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)
        b, h, w, _, w2 = init_corr.shape
        self.init_corr = init_corr.reshape(b*h*w, 1, 1, w2)

        b, c, d, h, w = geo_volume.shape
        self.geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)



    def __call__(self, disp, coords, ndisps=None):
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []

        dx = torch.linspace(-r, r, 2*r+1)
        dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)

        if ndisps is not None:  # For local geo_encoding_volume[B, 8, ndisps, H, W], use the center of the disparity dim (ndisps // 2) as reference
            center = torch.tensor([ndisps//2], dtype=torch.float32).to(disp.device).unsqueeze(0).repeat(b*h*w, 1)
            x0 = dx + center.reshape(b*h*w, 1, 1, 1)
        else:  # For full geo_encoding_volume[B, 8, D, H, W], directly use the ground-truth disparity value as reference 
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1)
        y0 = torch.zeros_like(x0)

        disp_lvl = torch.cat([x0,y0], dim=-1)
        geo_volume = bilinear_sampler(self.geo_volume, disp_lvl)
        geo_volume = geo_volume.view(b, h, w, -1)
        out_pyramid.append(geo_volume)

        init_x0 = coords.reshape(b*h*w, 1, 1, 1) - disp.reshape(b*h*w, 1, 1, 1) + dx
        init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
        init_corr = bilinear_sampler(self.init_corr, init_coords_lvl).view(b, h, w, -1)

        out_pyramid.append(init_corr)
        out = torch.cat(out_pyramid, dim=-1)

        return out.permute(0, 3, 1, 2).contiguous().float()

    
    @staticmethod
    def corr(fmap1, fmap2):
        B, D, H, W1 = fmap1.shape
        _, _, _, W2 = fmap2.shape
        fmap1 = fmap1.view(B, D, H, W1)
        fmap2 = fmap2.view(B, D, H, W2)
        corr = torch.einsum('aijk,aijh->ajkh', fmap1, fmap2)
        corr = corr.reshape(B, H, W1, 1, W2).contiguous()
        return corr

    @staticmethod
    def corr_selective(ref_fea, tgt_fea, disp_samples):
        """
        ref_fea: (B, C, H, W)
        tgt_fea: (B, C, H, W)
        disp_samples: (B, D, H, W)  每个像素的可行视差采样，单位像素
        Return:
            cost_volume: (B, D, H, W)
        """
        B, C, H, W = ref_fea.shape
        D = disp_samples.shape[1]

        x_warped, y_warped = get_warped_feats(ref_fea, tgt_fea, disp_samples, D)  # (B, C, D, H, W)

        cost = (x_warped * y_warped).sum(dim=1)  # 点乘后在通道维求和，(B, D, H, W)

        return cost.contiguous()