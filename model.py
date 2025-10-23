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
    """多帧提示词编码器 - 专注提取多帧上下文信息"""
    def __init__(self, num_frames=3, base_channels=64):
        super(MultiFramePromptEncoder, self).__init__()
        self.num_frames = num_frames
        
        # 多帧特征提取
        self.initial_conv = nn.Sequential(
            nn.Conv2d(num_frames, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
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
                nn.BatchNorm2d(base_channels // 4),
                nn.ReLU(inplace=True)
            ),
            'level2': nn.Sequential(
                nn.Conv2d(base_channels * 2, base_channels // 2, 1),
                nn.BatchNorm2d(base_channels // 2),
                nn.ReLU(inplace=True)
            ),
            'level3': nn.Sequential(
                nn.Conv2d(base_channels * 4, base_channels, 1),
                nn.BatchNorm2d(base_channels),
                nn.ReLU(inplace=True)
            ),
            'level4': nn.Sequential(
                nn.Conv2d(base_channels * 8, base_channels * 2, 1),
                nn.BatchNorm2d(base_channels * 2),
                nn.ReLU(inplace=True)
            ),
            'bottleneck': nn.Sequential(
                nn.Conv2d(base_channels * 16, base_channels * 4, 1),
                nn.BatchNorm2d(base_channels * 4),
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

class EnhancedChannelAttention(nn.Module):
    """增强的通道注意力机制 - 提取更多信息"""
    def __init__(self, in_channels, reduction_ratio=8):
        super(EnhancedChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 更深的网络提取更多信息
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels // (reduction_ratio//2), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // (reduction_ratio//2), in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x

class EnhancedSpatialAttention(nn.Module):
    """增强的空间注意力机制 - 提取更多信息"""
    def __init__(self, kernel_size=7):
        super(EnhancedSpatialAttention, self).__init__()
        # 更复杂的空间注意力
        self.conv1 = nn.Conv2d(2, 32, kernel_size, padding=kernel_size//2, bias=False)
        self.conv2 = nn.Conv2d(32, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.relu(self.conv1(attention))
        attention = self.conv2(attention)
        return self.sigmoid(attention) * x

class MultiScaleAttention(nn.Module):
    """多尺度注意力模块 - 像示例中的PGR模块"""
    def __init__(self, in_channels):
        super(MultiScaleAttention, self).__init__()
        # 多尺度特征提取
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3)
        
        # 注意力融合
        self.channel_attention = EnhancedChannelAttention(in_channels * 3)
        self.spatial_attention = EnhancedSpatialAttention()
        
        self.fusion_conv = nn.Conv2d(in_channels * 3, in_channels, 1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 多尺度特征
        scale1 = self.conv1(x)
        scale2 = self.conv2(x)
        scale3 = self.conv3(x)
        
        # 拼接多尺度特征
        multi_scale = torch.cat([scale1, scale2, scale3], dim=1)
        
        # 应用注意力
        attended = self.channel_attention(multi_scale)
        attended = self.spatial_attention(attended)
        
        # 融合
        fused = self.fusion_conv(attended)
        fused = self.bn(fused)
        fused = self.relu(fused + x)  # 残差连接
        
        return fused

class GuidedAttentionBlock(nn.Module):
    """引导注意力块 - 专注信息提取，引导信息只做条件"""
    def __init__(self, in_channels, prompt_channels):
        super(GuidedAttentionBlock, self).__init__()
        # 主特征处理 - 专注提取信息
        self.multi_scale_attention = MultiScaleAttention(in_channels)
        
        # 引导信息融合 - 简单条件注入
        self.prompt_condition = nn.Conv2d(prompt_channels, in_channels, 1)
        self.gradient_condition = nn.Conv2d(1, in_channels, 1)
        
        # 条件融合权重
        self.condition_weight = nn.Parameter(torch.tensor(0.1))
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prompt, gradient_map):
        # 主特征信息提取
        main_feat = self.multi_scale_attention(x)
        
        # 引导信息作为条件（轻量级融合）
        if prompt.size()[-2:] != x.size()[-2:]:
            prompt = F.interpolate(prompt, size=x.size()[-2:], mode='bilinear', align_corners=True)
        prompt_cond = self.prompt_condition(prompt)
        
        if gradient_map.size()[-2:] != x.size()[-2:]:
            gradient_map = F.interpolate(gradient_map, size=x.size()[-2:], mode='bilinear', align_corners=True)
        gradient_cond = self.gradient_condition(gradient_map)
        
        # 条件融合 - 保持主特征主导
        conditioned_feat = main_feat + self.condition_weight * (prompt_cond + gradient_cond)
        
        # 后处理
        conditioned_feat = self.bn(conditioned_feat)
        conditioned_feat = self.relu(conditioned_feat)
        
        return conditioned_feat

class AdaptiveWeightedFusion(nn.Module):
    """自适应权重融合模块"""
    def __init__(self, in_channels):
        super(AdaptiveWeightedFusion, self).__init__()
        # 权重学习网络
        self.weight_net = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 3, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, main_feat, prompt_feat, gradient_feat):
        # 拼接特征
        concat_feat = torch.cat([main_feat, prompt_feat, gradient_feat], dim=1)
        # 学习权重 (batch, 3, h, w)
        weights = self.weight_net(concat_feat)
        # 加权融合
        weighted_main = main_feat * weights[:, 0:1]
        weighted_prompt = prompt_feat * weights[:, 1:2] 
        weighted_gradient = gradient_feat * weights[:, 2:3]
        
        return weighted_main + weighted_prompt + weighted_gradient

class MultiModalInteraction(nn.Module):
    """多模态交互模块 - 平衡计算量和精度的卷积注意力版本"""
    def __init__(self, in_channels, prompt_channels):
        super(MultiModalInteraction, self).__init__()
        
        # 投影层
        self.prompt_proj = nn.Conv2d(prompt_channels, in_channels, 1)
        self.gradient_proj = nn.Conv2d(1, in_channels, 1)
        
        # 卷积注意力机制 - 替代MultiheadAttention
        self.conv_attention = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 特征融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, main_feat, prompt_feat, gradient_feat):
        # 投影到相同维度
        prompt_proj = self.prompt_proj(prompt_feat)
        gradient_proj = self.gradient_proj(gradient_feat)
        
        # 调整尺寸匹配
        if prompt_proj.size()[-2:] != main_feat.size()[-2:]:
            prompt_proj = F.interpolate(prompt_proj, size=main_feat.size()[-2:], 
                                      mode='bilinear', align_corners=True)
        if gradient_proj.size()[-2:] != main_feat.size()[-2:]:
            gradient_proj = F.interpolate(gradient_proj, size=main_feat.size()[-2:], 
                                        mode='bilinear', align_corners=True)
        
        # 拼接特征用于注意力计算
        concat_feat = torch.cat([main_feat, prompt_proj, gradient_proj], dim=1)
        
        # 卷积注意力权重
        attention_weights = self.conv_attention(concat_feat)
        
        # 应用注意力权重到主特征
        attended_main = main_feat * attention_weights
        
        # 特征融合
        fused_feat = torch.cat([attended_main, prompt_proj, gradient_proj], dim=1)
        fused_feat = self.fusion_conv(fused_feat)
        
        # 残差连接
        final_feat = fused_feat + main_feat
        
        # 后处理
        final_feat = self.bn(final_feat)
        final_feat = self.relu(final_feat)
        
        return final_feat

class EnhancedFusionBlock(nn.Module):
    """增强融合块 - 结合自适应权重和多模态交互"""
    def __init__(self, in_channels, prompt_channels):
        super(EnhancedFusionBlock, self).__init__()
        # 多模态交互
        self.multi_modal_interaction = MultiModalInteraction(in_channels, prompt_channels)
        
        # 自适应权重融合
        self.adaptive_fusion = AdaptiveWeightedFusion(in_channels)
        
        # 多尺度注意力
        self.multi_scale_attention = MultiScaleAttention(in_channels)
        
        # 投影层 - 修正通道数
        self.prompt_proj = nn.Conv2d(prompt_channels, in_channels, 1)
        self.gradient_proj = nn.Conv2d(1, in_channels, 1)
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prompt, gradient_map):
        # 投影特征
        prompt_proj = self.prompt_proj(prompt)
        
        # 调整梯度图尺寸
        if gradient_map.size()[-2:] != x.size()[-2:]:
            gradient_map = F.interpolate(gradient_map, size=x.size()[-2:], 
                                       mode='bilinear', align_corners=True)
        gradient_proj = self.gradient_proj(gradient_map)
        
        # 多模态交互 - 使用原始提示特征，让MultiModalInteraction内部处理投影
        interactive_feat = self.multi_modal_interaction(x, prompt, gradient_map)
        
        # 自适应权重融合 - 使用投影后的特征
        fused_feat = self.adaptive_fusion(x, prompt_proj, gradient_proj)
        
        # 结合交互特征和融合特征
        combined_feat = interactive_feat + fused_feat
        
        # 多尺度注意力
        attended_feat = self.multi_scale_attention(combined_feat)
        
        # 后处理
        attended_feat = self.bn(attended_feat)
        attended_feat = self.relu(attended_feat)
        
        return attended_feat

class MultiFrameCTDenoiser(nn.Module):
    """基于完整U-Net和多帧引导的低剂量CT去噪模型 - 增强版"""
    
    def __init__(self, num_frames=3, in_channels=1, out_channels=1, base_channels=64):
        super(MultiFrameCTDenoiser, self).__init__()
        self.num_frames = num_frames
        self.sobel = SobelOperator()
        
        # 多帧提示词编码器
        self.prompt_encoder = MultiFramePromptEncoder(num_frames, base_channels)
        
        # 初始卷积 - 输入为当前帧和梯度图拼接 (2通道)
        self.inc = DoubleConv(2, base_channels)

        # 下采样
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        
        # 增强融合块 - 结合自适应权重和多模态交互
        self.enhanced_fusion1 = EnhancedFusionBlock(base_channels, base_channels // 4)
        self.enhanced_fusion2 = EnhancedFusionBlock(base_channels * 2, base_channels // 2)
        self.enhanced_fusion3 = EnhancedFusionBlock(base_channels * 4, base_channels)
        self.enhanced_fusion4 = EnhancedFusionBlock(base_channels * 8, base_channels * 2)
        self.enhanced_fusion_bottleneck = EnhancedFusionBlock(base_channels * 16, base_channels * 4)
        
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
        
        # 直接拼接当前帧和梯度图，就像示例代码一样
        input_concat = torch.cat([current_frame, gradient_map], dim=1)  # (batch_size, 2, h, w)
        
        # 编码路径 with 增强融合
        x1 = self.inc(input_concat)
        x1 = self.enhanced_fusion1(x1, prompts['level1'], gradient_map)
        
        x2 = self.down1(x1)
        x2 = self.enhanced_fusion2(x2, prompts['level2'], gradient_map)
        
        x3 = self.down2(x2)
        x3 = self.enhanced_fusion3(x3, prompts['level3'], gradient_map)
        
        x4 = self.down3(x3)
        x4 = self.enhanced_fusion4(x4, prompts['level4'], gradient_map)
        
        x5 = self.down4(x4)
        x5 = self.enhanced_fusion_bottleneck(x5, prompts['bottleneck'], gradient_map)
        
        # 解码路径 with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 输出
        output = self.outc(x)
        return output
