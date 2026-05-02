import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.distributions.normal import Normal
import einops
from functools import reduce
from operator import mul
from GAgnn_Github import GNN

def window_partition(x_in, window_size, mode='conv'):
    x = einops.rearrange(x_in, 'b c d h w -> b d h w c')
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    if mode == 'mlp':
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    elif mode == 'conv':
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
        windows = windows.permute(0, 4, 1, 2, 3)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.mode = mode
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)

class VecInt(nn.Module):
    def __init__(self, inshape, nsteps=7):
        super().__init__()
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)
    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class ResizeTransform(nn.Module):
    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode
    def forward(self, x):
        if self.factor < 1:
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            x = self.factor * x
            x = F.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
        return x

class ConvInsBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1):
        super(ConvInsBlock, self).__init__()
        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)
    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, channel, alpha=0.1):
        super(ResBlock, self).__init__()
        self.channel = channel
        c = self.channel
        self.block = nn.Sequential(
            nn.InstanceNorm3d(c),
            nn.LeakyReLU(alpha),
            nn.Conv3d(c, c, 3, 1, 1)
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(c),
            nn.LeakyReLU(alpha),
        )
    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)

class Encoder(nn.Module):
    def __init__(self, in_channel, channel):
        super(Encoder, self).__init__()
        self.channel = channel
        c = self.channel
        self.conv0 = ConvInsBlock(in_channel, c, 3, 1, 1, alpha=0.1)
        self.conv1 = nn.Sequential(
            nn.Conv3d(c, 2*c, 3, 2, 1),
            ResBlock(2*c, alpha=0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(2*c, 3*c, 3, 2, 1),
            ResBlock(3*c, alpha=0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(3*c, 4*c, 3, 2, 1),
            ResBlock(4*c, alpha=0.1)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(4*c, 5*c, 3, 2, 1),
            ResBlock(5*c, alpha=0.1)
        )
    def forward(self, x):
        out0 = self.conv0(x)     # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)  # 1/16
        return out0, out1, out2, out3, out4

class DefBlock(nn.Module):
    def __init__(self, in_channels):
        super(DefBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, 3, 3, 1, 1)
        self.conv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv.weight.shape))
        self.conv.bias = nn.Parameter(torch.zeros(self.conv.bias.shape))
    def forward(self, x):
        x = self.conv(x)
        return x

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, alpha=0.1):
        super(UpConvBlock, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(alpha)
        )
    def forward(self, x):
        x = self.upconv(x)
        return self.actout(x)

class CConv(nn.Module):
    def __init__(self, channel):
        super(CConv, self).__init__()
        self.conv = nn.Sequential(
            ConvInsBlock(int(channel + 1) * 3, channel, 3, 1),
            ConvInsBlock(channel, channel, 3, 1)
        )
    def forward(self, float_fm, fixed_fm, d_fm):
        concat_fm = torch.cat([float_fm, fixed_fm, d_fm], dim=1)
        x = self.conv(concat_fm)
        return x

class FFTPool3d(nn.Module):
    def __init__(self, output_ratio):
        super(FFTPool3d, self).__init__()
        self.output_ratio = output_ratio
    def forward(self, x):
        X_temp = x.squeeze().squeeze()
        X_temp_fourier_all = torch.fft.fftn(X_temp)
        if X_temp.shape == (160, 192, 224):
            if self.output_ratio == 1/2:
                X_temp_fourier_low = torch.fft.fftshift(X_temp_fourier_all)[40:120, 48:144, 56:168]
            elif self.output_ratio == 1/4:
                X_temp_fourier_low = torch.fft.fftshift(X_temp_fourier_all)[20:60, 24:72, 28:84]
            elif self.output_ratio == 1/8:
                X_temp_fourier_low = torch.fft.fftshift(X_temp_fourier_all)[10:30, 12:36, 14:42]
            elif self.output_ratio == 1/16:
                X_temp_fourier_low = torch.fft.fftshift(X_temp_fourier_all)[5:15, 6:18, 7:21]
        else:
            print("Unknown size of image!")
            sys.exit(-1)
        X_temp_low_spatial_low = torch.real(torch.fft.ifftn(torch.fft.ifftshift(X_temp_fourier_low)).unsqueeze(0).unsqueeze(0))
        return X_temp_low_spatial_low

class DownPoolingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample1_avg = nn.AvgPool3d(kernel_size=2)
        self.downsample1_max = nn.MaxPool3d(kernel_size=2)
        self.downsample1_min = nn.MaxPool3d(kernel_size=2, ceil_mode=True)
        self.downsample1_fft = FFTPool3d(output_ratio=1/2)
        self.downsample2_avg = nn.AvgPool3d(kernel_size=4)
        self.downsample2_max = nn.MaxPool3d(kernel_size=4)
        self.downsample2_min = nn.MaxPool3d(kernel_size=4, ceil_mode=True)
        self.downsample2_fft = FFTPool3d(output_ratio=1/4)
        self.downsample3_avg = nn.AvgPool3d(kernel_size=8)
        self.downsample3_max = nn.MaxPool3d(kernel_size=8)
        self.downsample3_min = nn.MaxPool3d(kernel_size=8, ceil_mode=True)
        self.downsample3_fft = FFTPool3d(output_ratio=1/8)
        self.downsample4_avg = nn.AvgPool3d(kernel_size=16)
        self.downsample4_max = nn.MaxPool3d(kernel_size=16)
        self.downsample4_min = nn.MaxPool3d(kernel_size=16, ceil_mode=True)
        self.downsample4_fft = FFTPool3d(output_ratio=1/16)
    def forward(self, Img):
        return Img,\
               torch.cat([self.downsample1_avg(Img), self.downsample1_max(Img), self.downsample1_min(Img), self.downsample1_fft(Img)], dim=1),\
               torch.cat([self.downsample2_avg(Img), self.downsample2_max(Img), self.downsample2_min(Img), self.downsample2_fft(Img)], dim=1),\
               torch.cat([self.downsample3_avg(Img), self.downsample3_max(Img), self.downsample3_min(Img), self.downsample3_fft(Img)], dim=1),\
               torch.cat([self.downsample4_avg(Img), self.downsample4_max(Img), self.downsample4_min(Img), self.downsample4_fft(Img)], dim=1)

class Model(nn.Module):
    def __init__(self, inshape=(160, 192, 224), in_channel=1, channel=16, G8_channel=56, dropout=0.2):
        super(Model, self).__init__()
        self.inshape = inshape
        self.channel = channel
        c = self.channel
        self.encoder_mov = Encoder(in_channel, c)
        self.encoder_fix = Encoder(in_channel, c)
        self.encoder_pooling = DownPoolingBlock()
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.warp = nn.ModuleList()
        self.diff = nn.ModuleList()
        for i in range(5):
            self.warp.append(SpatialTransformer([s // 2**i for s in inshape]))
            self.diff.append(VecInt([s // 2**i for s in inshape]))

        self.GNN5 = GNN(int(5 * c + 4) * 2, int(5 * c + 4) // 2, G8_channel=G8_channel, G_ratio=2, W_ratio=[1, 1, 1, 1], dropout=dropout)
        self.Def5 = DefBlock(int(5 * c + 4) * 2)
        self.Dconv5 = nn.Sequential(
            ConvInsBlock(int(5 * c + 4) * 4, int(5 * c + 4) * 2),
            ConvInsBlock(int(5 * c + 4) * 2, int(5 * c + 4) * 2)
        )

        self.upconv4 = UpConvBlock(int(5 * c + 4) * 2, int(4 * c + 4), 4, 2)
        self.GNN4 = GNN(int(4 * c + 4) * 2, int(4 * c + 4) // 2, G8_channel=G8_channel, G_ratio=2, W_ratio=[1, 1, 1, 1], dropout=dropout)
        self.Def4 = DefBlock(int(4 * c + 4) * 2)
        self.Dconv4 = nn.Sequential(
            ConvInsBlock(int(4 * c + 4) * 4, int(4 * c + 4) * 2),
            ConvInsBlock(int(4 * c + 4) * 2, int(4 * c + 4) * 2)
        )
        self.conv4 = ConvInsBlock(int(4 * c + 4) * 3, int(4 * c + 4) * 2, kernal_size=2, stride=2, padding=0)
        self.conv4_trans = nn.ConvTranspose3d(int(4 * c + 4) * 2, int(4 * c + 4) * 2, kernel_size=2, stride=2)

        self.upconv3 = UpConvBlock(int(4 * c + 4) * 2, int(3 * c + 4), 4, 2)
        self.GNN3 = GNN(int(3 * c + 4) * 2, int(3 * c + 4) // 2, G8_channel=G8_channel, G_ratio=2, W_ratio=[1, 1, 1, 1], dropout=dropout)
        self.Def3 = DefBlock(int(3 * c + 4) * 2)
        self.Dconv3 = nn.Sequential(
            ConvInsBlock(int(3 * c + 4) * 4, int(3 * c + 4) * 2),
            ConvInsBlock(int(3 * c + 4) * 2, int(3 * c + 4) * 2)
        )
        self.conv3 = ConvInsBlock(int(3 * c + 4) * 3, int(3 * c + 4) * 2, kernal_size=4, stride=4, padding=0)
        self.conv3_trans = nn.ConvTranspose3d(int(3 * c + 4) * 2, int(3 * c + 4) * 2, kernel_size=4, stride=4)

        self.upconv2 = UpConvBlock(int(3 * c + 4) * 2, int(2 * c + 4), 4, 2)
        self.GNN2 = GNN(int(2 * c + 4) * 2, int(2 * c + 4) // 2, G8_channel=G8_channel, G_ratio=2, W_ratio=[1, 1, 1, 1], dropout=dropout)
        self.Def2 = DefBlock(int(2 * c + 4) * 2)
        self.Dconv2 = nn.Sequential(
            ConvInsBlock(int(2 * c + 4) * 4, int(2 * c + 4) * 2),
            ConvInsBlock(int(2 * c + 4) * 2, int(2 * c + 4) * 2)
        )
        self.conv2 = ConvInsBlock(int(2 * c + 4) * 3, int(2 * c + 4) * 2, kernal_size=8, stride=8, padding=0)
        self.conv2_trans = nn.ConvTranspose3d(int(2 * c + 4) * 2, int(2 * c + 4) * 2, kernel_size=8, stride=8)

        self.upconv1 = UpConvBlock(int(2 * c + 4) * 2, int(1 * c + 1), 4, 2)
        self.cconv1 = CConv(c)
        self.Def1 = DefBlock(c)
    def forward(self, mov, fix):
        M1, M2, M3, M4, M5 = self.encoder_mov(mov)  # 1 after convolution, 1/2, 1/4, 1/8, 1/16
        F1, F2, F3, F4, F5 = self.encoder_fix(fix)  # 1 after convolution, 1/2, 1/4, 1/8, 1/16
        m1, m2, m3, m4, m5 = self.encoder_pooling(mov)  # original 1, 1/2, 1/4, 1/8, 1/16
        f1, f2, f3, f4, f5 = self.encoder_pooling(fix)  # original 1, 1/2, 1/4, 1/8, 1/16

        Mm5 = window_partition(torch.cat([M5, m5], dim=1), (2, 2, 2), 'mlp')  # (210,8,5c+4)
        Ff5 = window_partition(torch.cat([F5, f5], dim=1), (2, 2, 2), 'mlp')  # (210,8,5c+4)
        C5 = torch.cat([Ff5, Mm5], dim=2)
        C5 = self.GNN5(C5)
        C5 = window_reverse(C5, (2, 2, 2), B=1, D=10, H=12, W=14)
        C5 = einops.rearrange(C5, 'b d h w c -> b c d h w')
        flow = self.Def5(C5)
        flow = self.diff[4](flow)

        warped = self.warp[4](torch.cat([M5, m5], dim=1), flow)
        C5 = self.Dconv5(torch.cat([F5, f5, warped, C5], dim=1))
        v = self.Def5(C5)
        v = self.diff[4](v)

        D4 = self.upconv4(C5)
        D4 = window_partition(D4, (4, 4, 4), 'conv')
        flow = self.upsample_trilin(2 * (self.warp[4](flow, v) + v))

        warped = self.warp[3](torch.cat([M4, m4], dim=1), flow)
        warped = window_partition(warped, (4, 4, 4), 'conv')
        Ff4 = window_partition(torch.cat([F4, f4], dim=1), (4, 4, 4), 'conv')
        C4 = torch.cat([Ff4, warped, D4], dim=1)
        C4 = self.conv4(C4)
        C4 = einops.rearrange(C4, 'b c d h w -> b (d h w) c')
        C4 = self.GNN4(C4)
        C4 = einops.rearrange(C4, 'b (d h w) c -> b c d h w', d=2, h=2, w=2)
        C4 = self.conv4_trans(C4)
        C4 = window_reverse(C4.flatten(start_dim=2, end_dim=4).permute(0, 2, 1), (4, 4, 4), B=1, D=20, H=24, W=28)
        C4 = einops.rearrange(C4, 'b d h w c -> b c d h w')
        flow = self.Def4(C4)
        flow = self.diff[3](flow)

        warped = self.warp[3](torch.cat([M4, m4], dim=1), flow)
        C4 = self.Dconv4(torch.cat([F4, f4, warped, C4], dim=1))
        v = self.Def4(C4)
        v = self.diff[3](v)
        D3 = self.upconv3(C4)
        D3 = window_partition(D3, (8, 8, 8), 'conv')
        flow = self.upsample_trilin(2 * (self.warp[3](flow, v) + v))

        warped = self.warp[2](torch.cat([M3, m3], dim=1), flow)
        warped = window_partition(warped, (8, 8, 8), 'conv')
        Ff3 = window_partition(torch.cat([F3, f3], dim=1), (8, 8, 8), 'conv')
        C3 = torch.cat([Ff3, warped, D3], dim=1)
        C3 = self.conv3(C3)
        C3 = einops.rearrange(C3, 'b c d h w -> b (d h w) c')
        C3 = self.GNN3(C3)
        C3 = einops.rearrange(C3, 'b (d h w) c -> b c d h w', d=2, h=2, w=2)
        C3 = self.conv3_trans(C3)
        C3 = window_reverse(C3.flatten(start_dim=2, end_dim=4).permute(0, 2, 1), (8, 8, 8), B=1, D=40, H=48, W=56)
        C3 = einops.rearrange(C3, 'b d h w c -> b c d h w')
        flow = self.Def3(C3)
        flow = self.diff[2](flow)

        warped = self.warp[2](torch.cat([M3, m3], dim=1), flow)
        C3 = self.Dconv3(torch.cat([F3, f3, warped, C3], dim=1))
        v = self.Def3(C3)
        v = self.diff[2](v)

        D2 = self.upconv2(C3)
        D2 = window_partition(D2, (16, 16, 16), 'conv')
        flow = self.upsample_trilin(2 * (self.warp[2](flow, v) + v))

        warped = self.warp[1](torch.cat([M2, m2], dim=1), flow)
        warped = window_partition(warped, (16, 16, 16), 'conv')
        Ff2 = window_partition(torch.cat([F2, f2], dim=1), (16, 16, 16), 'conv')
        C2 = torch.cat([Ff2, warped, D2], dim=1)
        C2 = self.conv2(C2)
        C2 = einops.rearrange(C2, 'b c d h w -> b (d h w) c')
        C2 = self.GNN2(C2)
        C2 = einops.rearrange(C2, 'b (d h w) c -> b c d h w', d=2, h=2, w=2)
        C2 = self.conv2_trans(C2)
        C2 = window_reverse(C2.flatten(start_dim=2, end_dim=4).permute(0, 2, 1), (16, 16, 16), B=1, D=80, H=96, W=112)
        C2 = einops.rearrange(C2, 'b d h w c -> b c d h w')
        flow = self.Def2(C2)
        flow = self.diff[1](flow)

        warped = self.warp[1](torch.cat([M2, m2], dim=1), flow)
        C2 = self.Dconv2(torch.cat([F2, f2, warped, C2], dim=1))
        v = self.Def2(C2)
        v = self.diff[1](v)

        D1 = self.upconv1(C2)
        flow = self.upsample_trilin(2 * (self.warp[1](flow, v) + v))

        warped = self.warp[0](torch.cat([M1, m1], dim=1), flow)
        C1 = self.cconv1(torch.cat([F1, f1], dim=1), warped, D1)
        v = self.Def1(C1)
        v = self.diff[0](v)

        flow = self.warp[0](flow, v) + v

        wapred_mov = self.warp[0](mov, flow)
        return wapred_mov, flow