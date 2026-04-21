import torch
from FDConv import FDConv

# 叶焕然：定义替换函数,把编码器中的conv层替换为FDConv层
def replace_conv_with_fdconv(module, kernel_num=64):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Conv2d):
            # 获取原卷积层参数
            in_channels = child.in_channels
            out_channels = child.out_channels
            kernel_size = child.kernel_size[0]  # 假设kernel是方形的
            stride = child.stride
            padding = child.padding
            groups = child.groups
            bias = child.bias is not None

            # 创建FDConv层
            fdconv_layer = FDConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
                kernel_num=kernel_num  # 可根据需要调整
            )

            # 替换原层
            setattr(module, name, fdconv_layer)
        else:
            # 递归处理子模块
            replace_conv_with_fdconv(child, kernel_num)
###示范案例
# import timm
# 执行替换
# 创建原始模型
# model = timm.create_model('convnextv2_base', pretrained=False)
# print(model)
# replace_conv_with_fdconv(model)
# print(model)
# model.load_state_dict(torch.load("convnextv2_base_FDConv.bin"))
# print(model)
