import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler
# def grid_sample(img: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
#     """Wrapper for grid_sample, uses pixel coordinates

#     Args:
#         img (torch.Tensor): Input image with shape (N, C, Hi, Wi).
#         grid (torch.Tensor): Coordinates with shape (N, Ho, Wo, 2).
#     Returns:
#         torch.Tensor: Output image with shape (N, C, Ho, Wo).
#     """

#     return F.grid_sample(
#         img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
#     )


# def bilinear_sampler(img: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
#     """Grid sample for 4D tensors with bilinear interpolation, zero padding and alignment corners.
#     Ref: https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/GridSampler.cpp

#     Args:
#         img (torch.Tensor): Input image with shape (N, C, Hi, Wi).
#         grid (torch.Tensor): Grid with shape (N, Ho, Wo, 2).
#     Returns:
#         torch.Tensor: Output image with shape (N, C, Ho, Wo).

#     Sanity check:
#         grid = torch.rand([1000, 12, 34, 2])
#         img = torch.randn([1000, 8, 100, 200])
#         out1 = torch.nn.functional.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
#         out2 = bilinear_sampler(img, grid)
#         print(torch.abs(out1 - out2).max())
#     """
#     N, C, Hi, Wi = img.shape
#     Ho, Wo = grid.shape[1:3]
#     img = img.view(N, C, -1)
#     grid = grid.view(N, -1, 2)

#     # align_corners=True and unormalize grid [-1, 1] to [0, W-1] and [0, H-1]
#     grid = (grid + 1) * torch.tensor([Wi - 1, Hi - 1]) / 2
#     delta = grid - grid.floor()  # bilinear interpolation

#     # (N, Ho, Wo, 4) => (x0, y0, x1, y1)
#     grid = grid.floor().long()
#     grid = torch.cat([grid, grid + 1], dim=-1)
#     # (N, Ho, Wo, 4) => (dx0, dy0, dx1, dy1)
#     delta = torch.cat([delta, 1 - delta], dim=-1)
#     # 0 => dx1 * dy1, 1 => dx0 * dy1
#     # 3 => dx1 * dy0, 2 => dx0 * dy0
#     # (dy1, dx0, dy0, dx1) * (dx1, dy1, dx0, dy0) = (dx1 * dy1, dx0 * dy1, dx0 * dy0, dx1 * dy0)
#     delta = delta.roll(1, dims=-1) * delta.roll(2, dims=-1)

#     is_inside = torch.logical_and(grid >= 0, grid < torch.tensor([Wi, Hi, Wi, Hi]))
#     is_inside = torch.logical_and(is_inside, is_inside.roll(-1, dims=-1))
#     delta.mul_(is_inside)  # if outside, delta = 0
#     # delta = torch.where(is_inside, delta, 0)
#     grid.clamp_(torch.tensor([0, 0, 0, 0]), torch.tensor([Wi, Hi, Wi, Hi]) - 1)

#     # Expand dimension to (N, C, Ho, Wo, 4)
#     delta = delta.unsqueeze(1)  # .expand(-1, C, -1, -1)

#     # Get the 4 corners
#     x = torch.tensor([0, 2, 2, 0]).view(1, 1, 4).expand(N, Ho * Wo, -1)
#     y = torch.tensor([1, 1, 3, 3]).view(1, 1, 4).expand(N, Ho * Wo, -1)
#     indices = grid.gather(-1, y) * Wi + grid.gather(-1, x)
#     indices = indices.view(N, 1, -1).expand(-1, C, -1)
#     return (
#         img.gather(-1, indices).view(N, C, -1, 4).mul(delta).sum(-1).view(N, C, Ho, Wo)
#     )


class Combined_Geo_Encoding_Volume:
    def __init__(self, init_fmap1, init_fmap2, geo_volume, num_levels=2, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.geo_volume_pyramid = []
        self.init_corr_pyramid = []

        # all pairs correlation
        init_corr = Combined_Geo_Encoding_Volume.corr(init_fmap1, init_fmap2)

        b, h, w, _, w2 = init_corr.shape
        b, c, d, h, w = geo_volume.shape
        geo_volume = geo_volume.permute(0, 3, 4, 1, 2).reshape(b*h*w, c, 1, d)

        init_corr = init_corr.reshape(b*h*w, 1, 1, w2)
        self.geo_volume_pyramid.append(geo_volume)
        self.init_corr_pyramid.append(init_corr)
        for i in range(self.num_levels-1):
            geo_volume = F.avg_pool2d(geo_volume, [1,2], stride=[1,2])
            self.geo_volume_pyramid.append(geo_volume)

        for i in range(self.num_levels-1):
            init_corr = F.avg_pool2d(init_corr, [1,2], stride=[1,2])
            self.init_corr_pyramid.append(init_corr)




    def __call__(self, disp, coords):
        r = self.radius
        b, _, h, w = disp.shape
        out_pyramid = []
        for i in range(self.num_levels):
            geo_volume = self.geo_volume_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dx = dx.view(1, 1, 2*r+1, 1).to(disp.device)
            x0 = dx + disp.reshape(b*h*w, 1, 1, 1) / 2**i
            y0 = torch.zeros_like(x0)

            disp_lvl = torch.cat([x0,y0], dim=-1)
            geo_volume = bilinear_sampler(geo_volume, disp_lvl)
            geo_volume = geo_volume.view(b, h, w, -1)

            init_corr = self.init_corr_pyramid[i]
            init_x0 = coords.reshape(b*h*w, 1, 1, 1)/2**i - disp.reshape(b*h*w, 1, 1, 1) / 2**i + dx
            init_coords_lvl = torch.cat([init_x0,y0], dim=-1)
            init_corr = bilinear_sampler(init_corr, init_coords_lvl)
            init_corr = init_corr.view(b, h, w, -1)

            out_pyramid.append(geo_volume)
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