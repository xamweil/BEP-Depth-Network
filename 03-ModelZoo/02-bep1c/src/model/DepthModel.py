import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torchvision.ops  # for deform_conv2d

class CostVolume3Patch(nn.Module):
    """
    Patch-based cost-volume for three rotationally-offset frames (xl, xm, xr).

    Output: cost_vol ∈ [B, 3, D, H, W]
        channel 0 : left  → mid  similarity
        channel 1 : right → mid  similarity
        channel 2 : left  ↔ right similarity
    """

    def __init__(self, max_disp: int, patch_size: int, offset: int = 0):
        super().__init__()
        assert patch_size % 2 == 1, "patch_size must be odd"
        self.max_disp   = max_disp
        self.patch_size = patch_size
        self.offset     = offset
        self.padding    = patch_size // 2      # pad so every (h,w) has a K×K patch

    # ------------------------------------------------------------------ #
    def forward(self, xl: torch.Tensor, xm: torch.Tensor, xr: torch.Tensor) -> torch.Tensor:

        B, C, H, W = xl.shape
        K2 = self.patch_size ** 2
        D  = self.max_disp

        # 1) unfold left & right into K×K patches .........................
        patches_l = F.unfold(xl, kernel_size=self.patch_size, padding=self.padding) \
                      .view(B, C, K2, H, W)
        patches_r = F.unfold(xr, kernel_size=self.patch_size, padding=self.padding) \
                      .view(B, C, K2, H, W)

        # 2) L2-normalise full (C×K²) vector for every location ..........
        def l2_norm(patch):
            patch = patch.reshape(B, C * K2, H, W)
            patch = F.normalize(patch, dim=1, eps=1e-3)
            return patch.reshape(B, C, K2, H, W)

        patches_l = l2_norm(patches_l)
        patches_r = l2_norm(patches_r)

        # 3) prepare middle-frame pixel vectors (unit-norm) .............
        fl = F.normalize(xm, dim=1, eps=1e-3)          # [B, C, H, W]
        fl = fl.unsqueeze(2).expand(-1, -1, K2, -1, -1)  # [B, C, K², H, W]

        # 4) sweep disparities ...........................................
        cost_L, cost_R, cost_LR = [], [], []
        scale = (C * K2) ** 0.5

        for d in range(self.offset, self.offset + D):
            if d > 0:
                shifted_l = F.pad(patches_l[..., d:], (0, d))   # shift left → right
                shifted_r = F.pad(patches_r[..., :-d], (d, 0))  # shift right → left
            else:
                shifted_l = patches_l
                shifted_r = patches_r

            # mid-frame vs left / right
            sim_L  = (fl * shifted_l).sum(dim=(1, 2)) / scale    # [B, H, W]
            sim_R  = (fl * shifted_r).sum(dim=(1, 2)) / scale

            # left vs right (patch-to-patch)
            sim_LR = (shifted_l * shifted_r).sum(dim=(1, 2)) / scale

            cost_L.append(sim_L)
            cost_R.append(sim_R)
            cost_LR.append(sim_LR)

        # 5) stack per-disparity results .................................
        cost_L   = torch.stack(cost_L,   dim=1)   # [B, D, H, W]
        cost_R   = torch.stack(cost_R,   dim=1)
        cost_LR  = torch.stack(cost_LR,  dim=1)

        # 6) pack into 3-channel volume and clamp / cast back to fp16 ....
        cost_vol = torch.stack([cost_L, cost_R, cost_LR], dim=1)   # [B, 3, D, H, W]
        cost_vol = torch.clamp(cost_vol, -10.0, 10.0).half()

        return cost_vol




class DownsampleBlock(nn.Module):
    """Learnable 2× downsampling via stride-2 conv."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3,
                              stride=2, padding=1, bias=False)
        G = max(1, min(32, channels // 8))
        self.norm = nn.GroupNorm(num_groups=G, num_channels=channels)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        # → [B, C, H/2, W/2]
        return self.act(self.norm(self.conv(x)))

class ChannelResBlock(nn.Module):
    """
    A simple residual block that *changes* the channel count.
    (Does *not* touch spatial dims.)
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False)
        G = max(1, min(32, out_ch // 8))
        self.norm1 = nn.GroupNorm(num_groups=G, num_channels=out_ch)
        self.act1  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=G, num_channels=out_ch)

        # Shortcut to match dimensions if needed
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act1(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act2(h + self.skip(x))


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3,
                              padding=1, bias=False)
        self.norm = nn.GroupNorm(max(1, channels // 8), channels)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        # 2× enlarge first
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=False)
        # then refine with a 3×3 conv
        return self.act(self.norm(self.conv(x)))


class DownSample3D(nn.Module):
    def __init__(self, channels: int, stride=(2, 2, 2), padding=(1, 1, 1)):
        super().__init__()
        self.conv3d = nn.Conv3d(channels, channels, kernel_size=3, stride=stride, padding=padding)
        G = max(1, min(32, channels // 8))
        self.norm = nn.GroupNorm(num_groups=G, num_channels=channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv3d(x)))

class ChannelRes3D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv3d1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        G = max(1, min(32, out_ch // 8))
        self.norm1 = nn.GroupNorm(num_groups=G, num_channels=out_ch)
        self.act1 = nn.ReLU(inplace=True)

        self.conv3d2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups=G, num_channels=out_ch)

        # Shortcut to match dimensions if needed
        self.skip = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act2 = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act1(self.norm1(self.conv3d1(x)))
        h = self.norm2(self.conv3d2(h))
        return self.act2(h + self.skip(x))

class UpSample3D(nn.Module):
    def __init__(self, channels, stride=(2, 2, 2), padding=(1, 1, 1)):
        super().__init__()
        self.deconv3d = nn.ConvTranspose3d(channels, channels, kernel_size=4, stride=stride, padding=padding, bias=False)

        G = max(1, min(32, channels // 8))
        self.norm = nn.GroupNorm(num_groups=G, num_channels=channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # → [B, C, 2H, 2W]
        return self.act(self.norm(self.deconv3d(x)))


class DisparityRegression(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, cost_volume):

        B, C, D, H, W = cost_volume.shape
        cost_volume = cost_volume.squeeze(1)  # [B, D, H, W]
        prob_volume = torch.softmax(-cost_volume, dim=1)  # [B, D, H, W]
        disparities = torch.arange(D, device=cost_volume.device).view(1, D, 1, 1)
        disp_map = torch.sum(prob_volume * disparities, dim=1)  # [B, H, W]
        return disp_map.unsqueeze(1)

class FixedSobel(nn.Module):
    """
    Edge map = σ( (‖∇depth‖ − threshold) · sharpness )

    • threshold, sharpness are hyper-parameters (not learned)
    • kernels are registered as buffers => require_grad = False
    """

    def __init__(self, threshold: float = 0.05, sharpness: float = 10.0):
        super().__init__()
        self.threshold = threshold
        self.sharpness = sharpness

        kx = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        # register as buffers so they move with .to(device) but are NOT updated
        self.register_buffer("kx", kx, persistent=False)
        self.register_buffer("ky", ky, persistent=False)

    @torch.no_grad()  # no gradient wrt the kernels
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B,1,H,W] depth (linear or log) in metres
        returns edge map in [0,1] with same spatial size
        """
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

        # soft edge mask
        edge = torch.sigmoid((grad_mag - self.threshold) * self.sharpness)
        return edge

class DropMask(nn.Module):
    def __init__(self, p=0.15):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        mask = torch.rand_like(x[:, :1]) > self.p   # [B,1,H,W]
        return x * mask
class BEPDepthNetwork(nn.Module):
    def __init__(self, base_ch: int = 32):

        super(BEPDepthNetwork, self).__init__()

        ####################
        # Drop outs
        ####################
        self.drop_out = DropMask(p=0.05)
        ####################
        # Encoder
        ####################
        # in_ch →     base_ch
        self.down1_spatial = DownsampleBlock(1)
        self.down1_chan = ChannelResBlock(1, base_ch)

        # base_ch → base_ch*2
        self.down2_spatial = DownsampleBlock(base_ch)
        self.down2_chan = ChannelResBlock(base_ch, base_ch * 2)

        # base_ch*2 → base_ch*4
        self.down3_spatial = DownsampleBlock(base_ch * 2)
        self.down3_chan = ChannelResBlock(base_ch * 2, base_ch * 4)

        ####################
        # Cost volume
        ####################
        self.cost_vol = CostVolume3Patch(max_disp=64, offset=50, patch_size=3)

        ####################
        # 3D encoder
        ####################
        self.down1_spatial3D = DownSample3D(3)
        self.down1_chan3D = ChannelRes3D(3, base_ch)
        self.down2_spatial3D = DownSample3D(base_ch)
        self.down2_chan3D = ChannelRes3D(base_ch, base_ch * 2)
        self.down3_spatial3D = DownSample3D(base_ch * 2, stride=(2, 2, 2), padding=(1, 1, 1))
        self.down3_chan3D = ChannelRes3D(base_ch * 2, base_ch * 4)

        ####################
        # 3D decoder
        ####################
        self.up3_3D = UpSample3D(base_ch * 4, stride=(2, 2, 2), padding=(1, 1, 1))
        self.dec3_3D = ChannelRes3D(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up2_3D = UpSample3D(base_ch * 2)
        self.dec2_3D = ChannelRes3D(base_ch * 2 + base_ch, base_ch)
        self.up1_3D = UpSample3D(base_ch)
        self.dec1_3D = ChannelRes3D(base_ch, 1)

        ####################
        # Disparity regression
        ####################
        self.disp_reg = DisparityRegression()

        ####################
        # Prepare skip connections and auxillery losses
        ####################
        self.proj_aux32 = nn.Conv3d(base_ch*2, 1, 1, bias=False)
        self.proj_aux16 = nn.Conv3d(base_ch, 1, 1, bias=False)
        self.skip1_proj = ChannelResBlock(in_ch=base_ch * 3, out_ch=base_ch * 2)
        self.skip2_proj = ChannelResBlock(in_ch=base_ch * 6, out_ch=base_ch * 3)
        self.bottleneck_proj = nn.Sequential(
                        nn.Conv2d(2,   base_ch * 4, 3, padding=1, groups=1, bias=False),  # depth-wise = False here (1→128)
                        nn.GroupNorm(32, base_ch * 4),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(base_ch * 4, base_ch * 8, 1, bias=False)
                    )

        ####################
        # Decoder
        ####################

        self.up3 = UpsampleBlock(base_ch * 8)
        self.dec3 = ChannelResBlock(base_ch * 8 + base_ch * 3, base_ch * 4)
        self.up2 = UpsampleBlock(base_ch * 4)
        self.dec2 = ChannelResBlock(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.up1 = UpsampleBlock(base_ch * 2)
        self.dec1 = ChannelResBlock(base_ch * 2, base_ch)
        ####################
        # Refinement
        ####################
        self.refine = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1, bias=False),
            nn.GroupNorm(32, base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 3, padding=1, bias=False),
            nn.GroupNorm(32, base_ch), nn.ReLU(inplace=True),
        )

        ####################
        # Final conv
        ####################
        # final base_ch → 1 depth map

        self.final_conv = nn.Conv2d(base_ch, 1, kernel_size=1)  # [B, base_ch, H, W]
        nn.init.zeros_(self.final_conv.bias)

        self.sobel = FixedSobel()

    def encoder(self, x):
        x = self.down1_spatial(x)  # [B, 2, H/2, W/2]
        x1 = self.down1_chan(x)  # [B, base_ch, H/2, W/2]
        x = self.down2_spatial(x1)  # [B, base_ch, H/4, W/4]
        x2 = self.down2_chan(x)  # [B, base_ch*2, H/4, W/4]
        # Dropouts
        x2 = self.drop_out(x2)
        x = self.down3_spatial(x2)  # [B, base_ch*2, H/8, W/8]
        x3 = self.down3_chan(x)  # [B, base_ch*4, H/8, W/8]

        return x3, x1, x2

    def u_net3D(self, x):
        x0 = x
        x = self.down1_spatial3D(x)     # [B, 3, D/2, H/16, W/16]
        x1 = self.down1_chan3D(x)       # [B, base_ch, D/2, H/16, W/16]
        x = self.down2_spatial3D(x1)    # [B, base_ch, D/4, H/32, W/32]
        x2 = self.down2_chan3D(x)       # [B, base_ch * 2, D/4, H/32, W/32]
        x = self.down3_spatial3D(x2)    # [B, base_ch * 2, D/4, H/64, W/64]
        x = self.down3_chan3D(x)        # [B, base_ch * 4, D/8, H/64, W/64]

        x = self.up3_3D(x)              # [B, base_ch * 4, D/8, H/32, W/32]
        x = F.interpolate(x, size=x2.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)   # [B, base_ch * 4 + base_ch * 2, D/4, H/32, W/32]
        x = self.dec3_3D(x)             # [B, base_ch * 2, D/4, H/32, W/32]
        aux32 = x
        x = self.up2_3D(x)              # [B, base_ch * 2, D/2, H/16, W/16]
        x = F.interpolate(x, size=x1.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)   # [B, base_ch * 2 + base_ch, D/2, H/16, W/16]
        x = self.dec2_3D(x)             # [B, base_ch, D/2, H/16, W/16]
        aux16 = x
        x = self.up1_3D(x)              # [B, base_ch, D, H/8, W/8]
        x = self.dec1_3D(x)             # [B, 1, D, H/8, W/8]

        return x, aux32, aux16

    def bottleNeck(self, xl, xm, xr):
        x = self.cost_vol(xl, xm, xr)  # [B, 3, D, H/8, W/8]
        x, aux32, aux16 = self.u_net3D(x)  # [B, 1, D, H/8, W/8], [B, base_ch*2, D, H/32, W/32], [B, base_ch, D, H/16, W/16]
        disp32, disp16 = self.disp_reg(self.proj_aux32(aux32)), self.disp_reg(self.proj_aux16(aux16)) # [B, 1, D, H/32, W/32], [B, 1, D, H/16, W/16]
        disp_map = self.disp_reg(x)  # [B, 1, H/8, W/8]
        return disp_map, disp16, disp32, x


    def decoder(self, x, skip1, skip2):
        x = self.up3(x)  # [B, base_ch*8, H/4, W/4]
        # force to match skip dim (prevent 'off by one')

        x = F.interpolate(x, size=skip2.shape[2:],
                          mode='bilinear', align_corners=False)
        x = torch.cat([x, skip2], dim=1)  # [B, base_ch*8 + base_ch*3, H/4, W/4]
        x = self.dec3(x)  # [B, base_ch*4, H/4, W/4]

        x = self.up2(x)  # [B, base_ch*4, H/2, W/2]
        x = F.interpolate(x, size=skip1.shape[2:],
                          mode='bilinear', align_corners=False)
        x = torch.cat([x, skip1], dim=1)  # [B, base_ch*4 + base_ch*2, H/2, W/2]
        x = self.dec2(x)  # [B, base_ch*2, H/2, W/2]

        x = self.up1(x)  # [B, base_ch*2, H, W]
        x = self.dec1(x)  # [B, base_ch, H, W]

        return x

    def encoder_bottle_checkpoint(self, xl, xm, xr):
        # siamese  encoding
        xl, skip1l, skip2l = self.encoder(xl)
        xm, skip1m, skip2m = self.encoder(xm)
        xr, skip1r, skip2r = self.encoder(xr)

        # Cost-volume 3D-Unet and disparity regression in bottleneck
        disp1_8, disp1_16, disp1_32, x_cost = self.bottleNeck(xl, xm, xr)
        # Fuse features and disparity
        feat8 = x_cost.mean(dim=2, keepdim=False) # Mean over disparity
        dec_in = torch.cat([disp1_8, feat8], dim=1)
        # Project low res disparity to 8 * base_ch
        x = self.bottleneck_proj(dec_in)

        # Prepare skip connections
        skip1 = torch.cat([skip1l, skip1m, skip1r], dim=1)
        skip2 = torch.cat([skip2l, skip2m, skip2r], dim=1)

        skip1 = self.skip1_proj(skip1)
        skip2 = self.skip2_proj(skip2)

        return x, disp1_8, disp1_16, disp1_32, skip1, skip2


    def decoder_checkpoint(self, x, skip1, skip2):
        # Decode with skip connections.
        x = self.decoder(x, skip1, skip2)
        return x

    def forward(self, x):
        B, T, C, H, W = x.shape
        xl, xm, xr = x[:, 0], x[:, 1], x[:, 2]

        # encoder + bottleneck
        x, disp1_8, disp1_16, disp1_32, skip1, skip2 = checkpoint(self.encoder_bottle_checkpoint, xl, xm, xr, use_reentrant=False)

        # Decoder
        x = self.decoder(x, skip1, skip2)

        x = x + self.refine(x)  # residual sharpen
        # Collapse channels
        x = self.final_conv(x)
        depth_map = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        edge = self.sobel(depth_map)

        if self.training:
            return depth_map, edge, disp1_8, disp1_16, disp1_32
        else:
            return depth_map, edge





