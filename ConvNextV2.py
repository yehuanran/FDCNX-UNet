import torch
import torch.nn as nn
import torch.nn.functional as F
"""
各种尺寸的都适用，不管是base还是large
## ConvNeXtV2
base是	[128, 256, 512, 1024]
large是 [192, 384, 768, 1536]
"""
class ConvBlockResPath(nn.Module):
    def __init__(self, num_filters, kernel_size, padding="same", act=True):
        super(ConvBlockResPath, self).__init__()
        self.num_filters = num_filters
        self.kernel_size = 3
        self.padding = padding
        self.act = act
        self.depthwise_conv = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=self.kernel_size, #改过
            stride=1,
            padding=1, #改过
            groups=num_filters,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.pointwise_conv = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_filters)

        if act:
            self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        if self.act:
            x = self.activation(x)

        x = self.pointwise_conv(x)
        x = self.bn2(x)
        if self.act:
            x = self.activation(x)
        return x


class ResPath(nn.Module):
    def __init__(self, num_filters, length):
        super(ResPath, self).__init__()
        self.num_filters = num_filters
        self.length = length

        self.res_layers = nn.ModuleList()
        for i in range(length):
            self.res_layers.append(nn.ModuleList([
                ConvBlockResPath(num_filters, 3, padding="same", act=False),
                ConvBlockResPath(num_filters, 1, padding="same", act=False),
                nn.LeakyReLU(inplace=True),
                nn.BatchNorm2d(num_filters)
            ]))

    def forward(self, x):
        for i in range(self.length):
            x0 = x
            conv_3x3, conv_1x1, leaky_relu, bn = self.res_layers[i]

            x1 = conv_3x3(x0)
            sc = conv_1x1(x0)

            x = x1 + sc
            x = leaky_relu(x)
            x = bn(x)

        return x
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, rate=1):
        super(ConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=rate,
                                   dilation=rate, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
class LinearAttention(nn.Module):
    def __init__(self, in_channels):
        super(LinearAttention, self).__init__()
        self.keys = nn.Linear(in_channels, in_channels)
        self.queries = nn.Linear(in_channels, in_channels)
        self.values = nn.Linear(in_channels, in_channels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        keys = self.keys(x)
        queries = self.queries(x)
        values = self.values(x)

        # 使用线性注意力近似，避免计算 (H*W, H*W) 的大矩阵
        # 标准注意力：softmax(QK^T)V -> 复杂度 O((HW)^2)
        # 线性注意力：Q(K^TV) -> 复杂度 O(HW*C^2)，其中 C<<HW

        # 对 queries 和 keys 应用核函数映射（这里用 ReLU 作为核函数）
        keys = F.relu(keys)
        queries = F.relu(queries)

        # 归一化
        keys_sum = keys.sum(dim=1, keepdim=True) + 1e-6
        queries_sum = queries.sum(dim=-1, keepdim=True) + 1e-6

        # 计算 K^T V (C, C) - 这是一个很小的矩阵
        kv = torch.bmm(keys.transpose(-2, -1), values)

        # 计算 Q (K^T V)
        out = torch.bmm(queries, kv)

        # 归一化因子
        normalizer = torch.bmm(queries, keys_sum.transpose(-2, -1))
        out = out / (normalizer + 1e-6)

        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

class ConvNeXtV2(nn.Module):
    def __init__(self, backbone, num_classes=21):
        super(ConvNeXtV2, self).__init__()

        # 加载预训练的模型
        self.backbone = backbone

        # 获取特征提取层
        self.stem = backbone.stem
        self.stages = backbone.stages

        # 通过实际前向传播确定各层的输出通道数
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)

            # 获取stem输出
            stem_out = self.stem(dummy_input)

            # 获取各stage输出
            stage1_out = self.stages[0](stem_out)
            stage2_out = self.stages[1](stage1_out)
            stage3_out = self.stages[2](stage2_out)
            stage4_out = self.stages[3](stage3_out)

            # 获取各层的通道数,以base为例
            self.feature_channels = [
                int(stage1_out.size(1)/4), # 32
                int(stage1_out.size(1)/2), # 64
                stage1_out.size(1),  # stage1 输出通道数 128
                stage2_out.size(1),  # stage2 输出通道数 256
                stage3_out.size(1),  # stage3 输出通道数 512
                stage4_out.size(1)  # stage4 输出通道数 1024
            ]

        # 上采样模块 以base为例，upsample_blocks是[1024->512, 512->256, 256->128,128->64,64->32]一共5个
        self.upsample_blocks = nn.ModuleList()
        for i in range(len(self.feature_channels) - 1, 0, -1): # 倒序
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(self.feature_channels[i], self.feature_channels[i - 1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(self.feature_channels[i - 1]),
                    nn.ReLU(inplace=True)
                )
            )
        # RLAB
        self.RLABS = nn.ModuleList()
        for i in range(len(self.feature_channels) - 2, 0, -1):
            self.RLABS.append(
                nn.Sequential(
                    ResPath(self.feature_channels[i],i),
                    ConvBlock(self.feature_channels[i]*2, self.feature_channels[i]),
                    LinearAttention(self.feature_channels[i])
                )
            )

        # 特征融合卷积层 以base为例，fusion_convs[1024->512, 512->256, 256->128,128->64,64->32]一共5个
        self.fusion_convs = nn.ModuleList()
        for i in range(len(self.feature_channels) - 1, 0, -1):
            self.fusion_convs.append(
                nn.Conv2d(self.feature_channels[i], self.feature_channels[i-1], kernel_size=1)
            )

        self.final = nn.Conv2d(self.feature_channels[0], num_classes, kernel_size=1)
    def forward(self, x):
        # 编码器部分 - 提取多尺度特征
        features = []

        # stem 层
        x = self.stem(x)
        features.append(x)

        # stages
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)

        # 解码器部分 - 级联上采样和注意力融合
        decoder_feature = features[-1]  # 使用最深层特征开始解码

        # 逐级上采样和融合,最后一个upsample_blocks在循环外用，而最后一个fusion_convs没用
        for i in range(len(self.upsample_blocks)-1):
            # 先上采样，对最底层特征进行上采样
            decoder_feature = self.upsample_blocks[i](decoder_feature)
            # 跳跃链接从倒数第二层开始
            skip_feat_adjusted = features[len(features) - 2 - i]

            # 拼接特征
            if(i==3): #i=3也就是stem对应的那层，尺度不一致，必须都进行反卷积，然后再拼接
                skip_feat_adjusted = self.upsample_blocks[i](skip_feat_adjusted)
                ResPath_feature = self.RLABS[i][0](skip_feat_adjusted)
                fused_feature = torch.cat([decoder_feature, ResPath_feature], dim=1)
                convblock_feature = self.RLABS[i][1](fused_feature)
                #加上LAB，就是下面两句
                linear_feature = self.RLABS[i][2](convblock_feature)
                RLAB_feature = linear_feature + decoder_feature
                #不加LAB，下面一句
                # RLAB_feature = convblock_feature + decoder_feature
            else:
                ResPath_feature = self.RLABS[i][0](skip_feat_adjusted)
                fused_feature = torch.cat([decoder_feature, ResPath_feature], dim=1)
                convblock_feature = self.RLABS[i][1](fused_feature)
                #加上LAB，就是下面两句
                linear_feature = self.RLABS[i][2](convblock_feature)
                RLAB_feature = linear_feature + decoder_feature
                # RLAB_feature = convblock_feature + decoder_feature
            # # 应用特征融合卷积
            # fused_feature = self.fusion_convs[i](fused_feature)
            # 更新解码器特征
            decoder_feature = RLAB_feature

        # 最终预测,还差两次反卷积，从6*128*128*128->6*64*256*256->6*3*512*512
        output = self.final(self.upsample_blocks[-1](decoder_feature))

        return output

# if __name__ == '__main__':
#     model = timm.create_model('convnextv2_base', pretrained=False)
#     model = ConvNeXtV2(model)
#     print(model)