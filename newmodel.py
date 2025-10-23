
import torch.nn as nn
import torch.nn.functional as F
import torch
from math import sqrt
from einops import rearrange
import numbers
from sobel import SobelOperator
import numpy as np



class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(TripletAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.SpatialGate(x)
            x_out = (1 / 3) * (x_out + x_out11 + x_out21)
        else:
            x_out = (1 / 2) * (x_out11 + x_out21)
        return x_out


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.avg_pool(x).view(b, c)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out).view(b, c, 1, 1)
        out = x * out.expand_as(x)
        return out


class SENet(nn.Module):
    def __init__(self, inp_channels):
        super(SENet, self).__init__()

        self.conv1 = nn.Conv2d(inp_channels, inp_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inp_channels)
        self.se_block = SEBlock(inp_channels)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(inp_channels , inp_channels)

    def forward(self, x):

        out = self.conv1(x)

        out = self.bn1(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.se_block(out)
        return out



class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##############################
class CAnet(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(CAnet, self).__init__()
        self.TCAB = TripletAttention(inchannel)
        self.CAM = SENet(inchannel)

    def forward(self, x):
        x = self.TCAB(x)
        x = self.CAM(x)

        return x
#噪
class MultiFramePromptEncoder(nn.Module):
    """多帧提示词编码器 - 提取多帧上下文信息作为引导"""
    def __init__(self, num_frames=3, base_channels=16):
        super(MultiFramePromptEncoder, self).__init__()
        self.num_frames = num_frames
        
        # 多帧特征提取
        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_frames, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # 多尺度特征提取
        self.enc1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(inplace=True)
        )
        
        # 多尺度提示特征融合
        self.prompt_fusion = nn.ModuleDict({
            'level1': nn.Sequential(
                nn.Conv2d(base_channels, base_channels, 1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            ),
            'level2': nn.Sequential(
                nn.Conv2d(base_channels*2, base_channels, 1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            ),
            'level3': nn.Sequential(
                nn.Conv2d(base_channels*4, base_channels, 1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            )
        })

    def forward(self, x):
        # x shape: (batch_size, num_frames, h, w)
        x1 = self.initial_conv(x)
        x1 = self.enc1(x1)
        
        x2 = self.down1(x1)
        x2 = self.enc2(x2)
        
        x3 = self.down2(x2)
        
        # 提取多尺度提示特征
        prompts = {
            'level1': self.prompt_fusion['level1'](x1),
            'level2': self.prompt_fusion['level2'](x2),
            'level3': self.prompt_fusion['level3'](x3)
        }
        
        return prompts

class GuidedAttentionBlock(nn.Module):
    """引导注意力块 - 结合多帧提示词和梯度信息"""
    def __init__(self, in_channels, prompt_channels):
        super(GuidedAttentionBlock, self).__init__()
        
        # 主特征处理
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 提示词条件注入
        self.prompt_condition = nn.Conv2d(prompt_channels, in_channels, 1)
        self.gradient_condition = nn.Conv2d(1, in_channels, 1)
        
        # 条件融合权重
        self.condition_weight = nn.Parameter(torch.tensor(0.1))
        
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x, prompt, gradient_map):
        # 主特征处理
        main_feat = self.conv1(x)
        main_feat = self.bn1(main_feat)
        main_feat = self.relu(main_feat)
        
        # 引导信息作为条件
        if prompt.size()[-2:] != x.size()[-2:]:
            prompt = F.interpolate(prompt, size=x.size()[-2:], mode='bilinear', align_corners=True)
        prompt_cond = self.prompt_condition(prompt)
        
        if gradient_map.size()[-2:] != x.size()[-2:]:
            gradient_map = F.interpolate(gradient_map, size=x.size()[-2:], mode='bilinear', align_corners=True)
        gradient_cond = self.gradient_condition(gradient_map)
        
        # 条件融合
        conditioned_feat = main_feat + self.condition_weight * (prompt_cond + gradient_cond)
        
        # 后处理
        conditioned_feat = self.conv2(conditioned_feat)
        conditioned_feat = self.bn2(conditioned_feat)
        conditioned_feat = self.relu(conditioned_feat + x)  # 残差连接
        
        return conditioned_feat

class ImageGenerator(nn.Module):
    def __init__(self,
                 inp_channels=3,  # 改为3通道以匹配多帧输入
                 out_channels=1,
                 dim = 16,
                 num_frames=3  # 添加多帧参数
                 ):

        super(ImageGenerator, self).__init__()

        self.num_frames = num_frames
        self.prompt_encoder = MultiFramePromptEncoder(num_frames, dim)
        
        # 修改patch_embed以处理当前帧（1通道）
        self.patch_embed = OverlapPatchEmbed(1, dim)  # 当前帧单独处理
        self.sobel = SobelOperator()
        
        # 添加引导注意力块
        self.guided_attn1 = GuidedAttentionBlock(dim*2, dim)
        self.guided_attn2 = GuidedAttentionBlock(int(dim*4), dim)
        self.guided_attn3 = GuidedAttentionBlock(int(dim*8), dim)
        self.guided_attn4 = GuidedAttentionBlock(int(dim*16), dim)
        
        # 原有的CAnet模块保持不变
        self.CAM1 = CAnet(dim*2)
        self.down1_2 = Downsample(dim*2)  ## From Level 1 to Level 2
        self.CAM2 = CAnet(int(dim*4))
        self.down2_3 = Downsample(int(dim * 4 ))  ## From Level 2 to Level 3
        self.CAM3 = CAnet(int(dim*8))
        self.down3_4 = Downsample(int(dim * 8))  ## From Level 3 to Level 4
        self.CAM4 = CAnet(int(dim*16))
        self.up4_3 = Upsample(int(dim *16))  ## From Level 4 to Level 3
        self.CAM5 = CAnet(int(dim*16))
        self.up3_2 = Upsample(int(dim * 16))  ## From Level 3 to Level 2
        self.CAM6 = CAnet(int(dim*12))
        self.up2_1 = Upsample(int(dim * 12))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.CAM7 = CAnet(int(dim*8))
        self.output = nn.Conv2d(int(dim*8) , out_channels, kernel_size=3, stride=1, padding=1)

        self.grad_cleaner = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)

    def weights_init_normal(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=0.01)
            if m.bias is not None:
               nn.init.constant_(m.bias, 0)


    def forward(self, inp_img):
        # inp_img shape: (batch_size, num_frames, h, w) - 多帧输入
        batch_size, num_frames, h, w = inp_img.shape
        
        # 提取当前帧（中间帧）
        current_frame = inp_img[:, 1:2, :, :]  # (batch_size, 1, h, w)
        
        # 提取多帧提示词特征
        prompts = self.prompt_encoder(inp_img)
        
        # 计算当前帧的梯度
        grad = self.sobel(current_frame)
        grad1 = self.patch_embed(grad)
        image = self.patch_embed(current_frame)  # 只处理当前帧
        
        # 拼接当前帧特征和梯度特征
        input = torch.cat([image, grad1], 1)
        
        # 编码路径 - 集成多帧提示词引导
        image1 = self.CAM1(input)  # 第一次交叉注意力
        image1 = self.guided_attn1(image1, prompts['level1'], grad)  # 添加提示词引导
        
        image2 = self.down1_2(image1)  # 图像下采样
        image2 = self.guided_attn2(image2, prompts['level2'], grad)  # 添加提示词引导
        
        image3 = self.CAM2(image2)  # 第二次交叉注意力
        image3 = self.guided_attn2(image3, prompts['level2'], grad)  # 添加提示词引导
        
        image4 = self.down2_3(image3)
        image4 = self.guided_attn3(image4, prompts['level3'], grad)  # 添加提示词引导
        
        image5 = self.CAM3(image4)  # 第三次交叉注意力
        image5 = self.guided_attn3(image5, prompts['level3'], grad)  # 添加提示词引导
        
        image6 = self.down3_4(image5)
        image6 = self.guided_attn4(image6, prompts['level3'], grad)  # 添加提示词引导
        
        latent = self.CAM4(image6)
        latent = self.guided_attn4(latent, prompts['level3'], grad)  # 添加提示词引导
        
        # 解码路径
        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, image5], 1)
        out_dec_level3 = self.CAM5(inp_dec_level3)
        

        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, image3], 1)
        out_dec_level2 = self.CAM6(inp_dec_level2)
        


        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, image1], 1)
        out_dec_level1 = self.CAM7(inp_dec_level1)
        
        # 输出去噪后的当前帧
        denoised_image = self.output(out_dec_level1) + current_frame

        cleargrad = self.grad_cleaner(grad1)
        return grad, cleargrad, denoised_image
