import torch
import torch.nn as nn
import torch.nn.functional as F
from sobel import SobelOperator

class DoubleConv(nn.Module):
    """U-Net中的双卷积块"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """U-Net下采样层"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """U-Net上采样层"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 上采样后与跳跃连接拼接，通道数变为 in_channels + out_channels
            self.conv = DoubleConv(in_channels + out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # 转置卷积后与跳跃连接拼接，通道数变为 in_channels//2 + out_channels
            self.conv = DoubleConv(in_channels // 2 + out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 输入是 CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class MultiFramePromptEncoder(nn.Module):
    """多帧提示词编码器"""
    def __init__(self, num_frames=3, base_channels=64):
        super(MultiFramePromptEncoder, self).__init__()
        self.num_frames = num_frames
        
        # 多帧特征提取
        self.initial_conv = nn.Conv2d(num_frames, base_channels, 3, padding=1)
        self.enc1 = DoubleConv(base_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.enc2 = DoubleConv(base_channels * 2, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.enc3 = DoubleConv(base_channels * 4, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.enc4 = DoubleConv(base_channels * 8, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        self.bottleneck = DoubleConv(base_channels * 16, base_channels * 16)
        
        # 多尺度特征融合
        self.prompt_fusion = nn.ModuleDict({
            'level1': nn.Sequential(
                nn.Conv2d(base_channels, base_channels // 4, 1),
                nn.ReLU(inplace=True)
            ),
            'level2': nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels // 2, 1),
                nn.ReLU(inplace=True)
            ),
            'level3': nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels, 1),
                nn.ReLU(inplace=True)
            ),
            'level4': nn.Sequential(
                nn.Conv2d(base_channels * 8, base_channels * 2, 1),
                nn.ReLU(inplace=True)
            ),
            'bottleneck': nn.Sequential(
                nn.Conv2d(base_channels * 16, base_channels * 4, 1),
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
        x3 = self.enc3(x3)
        
        x4 = self.down3(x3)
        x4 = self.enc4(x4)
        
        x5 = self.down4(x4)
        x5 = self.bottleneck(x5)
        
        # 提取多尺度提示特征
        prompts = {
            'level1': self.prompt_fusion['level1'](x1),
            'level2': self.prompt_fusion['level2'](x2),
            'level3': self.prompt_fusion['level3'](x3),
            'level4': self.prompt_fusion['level4'](x4),
            'bottleneck': self.prompt_fusion['bottleneck'](x5)
        }
        
        return prompts

class ChannelAttention(nn.Module):
    """通道注意力机制"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class SpatialAttention(nn.Module):
    """空间注意力机制"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return self.sigmoid(attention) * x

class PromptGuidedBlock(nn.Module):
    """提示词引导块"""
    def __init__(self, in_channels, prompt_channels):
        super(PromptGuidedBlock, self).__init__()
        self.main_conv = DoubleConv(in_channels, in_channels)
        
        # 提示词融合
        self.prompt_conv = nn.Conv2d(prompt_channels, in_channels, 1)
        self.prompt_bn = nn.BatchNorm2d(in_channels)
        
        # 梯度特征融合
        self.gradient_conv = nn.Conv2d(1, in_channels, 3, padding=1)
        self.gradient_bn = nn.BatchNorm2d(in_channels)
        
        # 注意力机制
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prompt, gradient_map):
        # 主特征处理
        main_feat = self.main_conv(x)
        
        # 提示词融合
        prompt_feat = self.prompt_conv(prompt)
        prompt_feat = self.prompt_bn(prompt_feat)
        prompt_feat = self.relu(prompt_feat)
        
        # 梯度特征融合 - 必须融合
        # 调整梯度图尺寸以匹配当前特征图
        if gradient_map.size()[-2:] != x.size()[-2:]:
            gradient_map = F.interpolate(gradient_map, size=x.size()[-2:], mode='bilinear', align_corners=True)
        gradient_feat = self.gradient_conv(gradient_map)
        gradient_feat = self.gradient_bn(gradient_feat)
        gradient_feat = self.relu(gradient_feat)
        
        # 将梯度特征与主特征和提示特征相加
        fused_feat = main_feat + prompt_feat + gradient_feat
        
        # 应用注意力机制
        fused_feat = self.channel_attention(fused_feat)
        fused_feat = self.spatial_attention(fused_feat)
        
        return fused_feat

class MultiFrameCTDenoiser(nn.Module):
    """基于完整U-Net和多帧提示词引导的低剂量CT去噪模型"""
    
    def __init__(self, num_frames=3, in_channels=1, out_channels=1, base_channels=64):
        super(MultiFrameCTDenoiser, self).__init__()
        self.num_frames = num_frames
        self.sobel = SobelOperator()
        # 多帧提示词编码器
        self.prompt_encoder = MultiFramePromptEncoder(num_frames, base_channels)
        # 主U-Net编码器 (处理当前帧)
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        
        # 提示词引导块
        self.prompt_guided1 = PromptGuidedBlock(base_channels, base_channels // 4)
        self.prompt_guided2 = PromptGuidedBlock(base_channels * 2, base_channels // 2)
        self.prompt_guided3 = PromptGuidedBlock(base_channels * 4, base_channels)
        self.prompt_guided4 = PromptGuidedBlock(base_channels * 8, base_channels * 2)
        self.prompt_guided_bottleneck = PromptGuidedBlock(base_channels * 16, base_channels * 4)
        
        # 解码器
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)
        
        # 输出层
        self.outc = nn.Conv2d(base_channels, out_channels, 1)
        
        # 权重初始化
        self.apply(self.weights_init_normal)
    
    @staticmethod
    def weights_init_normal(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)
    
    def forward(self, x):
        """
        Args:
            x: 多帧输入 (batch_size, num_frames, h, w)
        """
        batch_size, num_frames, h, w = x.shape
        
        current_frame = x[:, 1:2, :, :]  # (batch_size, 1, h, w)
        
        gradient_map = self.sobel(current_frame)
        
        prompts = self.prompt_encoder(x)
        
        x1 = self.inc(current_frame)
        x1 = self.prompt_guided1(x1, prompts['level1'], gradient_map)
        
        x2 = self.down1(x1)
        x2 = self.prompt_guided2(x2, prompts['level2'], gradient_map)
        
        x3 = self.down2(x2)
        x3 = self.prompt_guided3(x3, prompts['level3'], gradient_map)
        
        x4 = self.down3(x3)
        x4 = self.prompt_guided4(x4, prompts['level4'], gradient_map)
        
        x5 = self.down4(x4)
        x5 = self.prompt_guided_bottleneck(x5, prompts['bottleneck'], gradient_map)
        
        # 解码路径 with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 输出
        output = self.outc(x)
        return output
