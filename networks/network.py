import math
import torch
import torch.nn.functional as F

from einops import rearrange
from torch import nn, einsum


def exists(x):
    return x is not None


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim_in, dim_out):
    return nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1)


def Downsample(dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, 4, 2, 1)


class SinusoidalPositionEmbeddings(nn.Module):  # 生成正弦位置编码
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 维度

    def forward(self, time):  # 时间步t生成位置编码
        device = time.device  # 获取时间步t所在的设备
        half_dim = self.dim // 2  # 计算半维度
        embeddings = math.log(10000) / (half_dim - 1)  # 计算每个维度的频率
        embeddings = torch.exp(torch.arange(half_dim,
                                            device=device) * -embeddings)  # 从0到负数递增的序列，乘以之前的embeddings系数。这样做的目的是生成不同频率的波长，每个维度对应不同的频率，从而捕捉不同位置的信息。
        embeddings = time[:, None] * embeddings[None, :]  # 将时间步t乘以每个维度的频率，得到每个维度的嵌入向量。
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)  # 将sin和cos相加，得到每个维度的嵌入向量。
        return embeddings


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)  # 可以试试batchsize会不会效果更好
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ResNetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1)  # if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h  # 将原来的二维张量`(b, c)`转换为四维张量，新增两个维度，结果形状变为`(b, c, 1, 1)`

        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        # 步骤1: 生成Q/K/V并分头
        qkv = self.to_qkv(x).chunk(3, dim=1)  # chunk沿通道维度（dim=1）均匀分割为3部分
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )  # 重排为 [batch, heads, dim_head, seq_len] (seq_len=h*w)
        # 步骤2: 归一化Q/K
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        # 步骤3: 计算注意力上下文
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        # 步骤4: 注意力加权输出
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        # 步骤5: 合并多头并还原空间结构
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        # 步骤6: 输出层处理
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class NetworkConfig:
    """Configuration for the network."""
    # Default configuration
    image_channels = 3
    n_classes = 19
    dim = 32
    dim_mults = (1, 2, 4, 8)
    resnet_block_groups = 8

    # diffusion parameters
    n_timesteps = 10
    n_scales = 3
    max_patch_size = 512
    scale_procedure = "loop"  # "linear" or "loop"

    # ensemble parameters
    built_in_ensemble = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class CAB(nn.Module):
    def __init__(self, poolin_channel, out_channel):
        super(CAB, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 最大赤化还没试过
        self.conv1 = nn.Conv2d(poolin_channel, out_channel, 1, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, globle_pool):
        out1 = x
        out1_1 = torch.cat([x, globle_pool], 1)
        out1_2 = self.pool(out1_1)
        out2 = self.sigmoid(self.conv2(self.bn(self.relu(self.conv1(out1_2)))))
        out3 = out1 * out2
        out = torch.cat([out3, globle_pool], 1)
        # out = out3+globle_pool
        return out


class Network(nn.Module):
    def __init__(
            self,
            network_config=NetworkConfig(),
    ):
        super().__init__()
        self.config = network_config
        image_channels = self.config.image_channels
        n_classes = self.config.n_classes
        dim = self.config.dim
        dim_mults = self.config.dim_mults
        resnet_block_groups = self.config.resnet_block_groups

        # determine dimensions
        self.image_channels = image_channels
        self.n_classes = n_classes
        self.dims = [c * dim for c in dim_mults]  # [32, 64, 128, 256]

        # time embedding 时间嵌入层模块,为扩散模型的时间步生成嵌入，帮助网络在不同时间步调整行为
        time_dim = dim * 4  # 128,将基础维度扩展4倍
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),  # 生成时间步的位置编码,是将时间步转换为高维向量，这样模型能更好地处理不同时间步的信息
            nn.Linear(dim, time_dim),  # 将高维向量映射到更大的维度
            nn.GELU(),  # 非线性激活函数
            nn.Linear(time_dim, time_dim),  # 映射到更大的维度
        )

        # image initial 图像初始化处理模块
        self.image_initial = nn.ModuleList([
            ResNetBlock(image_channels, self.dims[0], time_emb_dim=time_dim, groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups)
        ])

        # segmentation initial 分割图初始化处理模块
        self.seg_initial = nn.ModuleList([
            ResNetBlock(n_classes, self.dims[0], time_emb_dim=time_dim, groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups),
            ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups)
        ])

        # layers
        self.down = nn.ModuleList([])
        self.up = nn.ModuleList([])
        self.CAB = nn.ModuleList([])
        self.conv = nn.ModuleList([])
        # encoder
        for i in range(len(dim_mults) - 1):  # each dblock
            dim_in = self.dims[i]
            dim_out = self.dims[i + 1]

            self.down.append(
                nn.ModuleList([
                    ResNetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                    ResNetBlock(dim_in, dim_in, groups=resnet_block_groups),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out),

                ])
            )

        # decoder
        for i in range(len(dim_mults) - 1):  # each ublock
            dim_in = self.dims[-i - 1]
            dim_out = self.dims[-i - 2]
            if i == 0:
                dim_in_plus_concat = dim_in
            else:
                dim_in_plus_concat = dim_in * 2

            self.up.append(
                nn.ModuleList([
                    ResNetBlock(dim_in_plus_concat, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                    ResNetBlock(dim_in, dim_in, groups=resnet_block_groups),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Upsample(dim_in, dim_out),
                ])
            )
        # CAB
        for i in range(len(dim_mults) - 1):
            self.CAB.append(CAB(poolin_channel=self.dims[-i - 1], out_channel=self.dims[-i - 2]))
            self.conv.append(nn.Conv2d(self.dims[-i - 2], self.dims[-i - 1], 1))
        # final
        self.final = nn.Sequential(ResNetBlock(self.dims[0] * 2, self.dims[0], groups=resnet_block_groups),
                                   ResNetBlock(self.dims[0], self.dims[0], groups=resnet_block_groups),
                                   nn.Conv2d(self.dims[0], n_classes, 1))

    def forward(self, seg, img, time):
        # time embedding 生成时间嵌入
        t = self.time_mlp(time)

        # segmentation initial
        resnetblock1, resnetblock2, resnetblock3 = self.seg_initial
        seg_emb = resnetblock1(seg, t)
        # seg_emb = resnetblock2(seg_emb)
        # seg_emb = resnetblock3(seg_emb)

        # image initial
        resnetblock1, resnetblock2, resnetblock3 = self.image_initial
        img_emb = resnetblock1(img, t)
        # img_emb = resnetblock2(img_emb)
        # img_emb = resnetblock3(img_emb)

        # add embeddings together
        x = seg_emb + img_emb

        # skip connections
        h = []

        # downsample
        for resnetblock1, resnetblock2, attn, downsample in self.down:
            x = resnetblock1(x, t)
            x = resnetblock2(x)
            # x = attn(x)
            h.append(x)
            x = downsample(x)
        i = 0
        # upsample
        for resnetblock1, resnetblock2, attn, upsample in self.up:
            x = resnetblock1(x, t)
            x = resnetblock2(x)
            # x = attn(x)
            x = upsample(x)
            # x = self.CAB[i](x, h.pop())
            # x = self.conv[i](x)
            # x = torch.cat(x, h.pop(), dim=1)
            i = i + 1
        return self.final(x)