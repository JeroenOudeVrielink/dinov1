import torch
from torch import nn

from torchvision.models import resnet50
import torch
from torch import nn

# model = resnet50()
# # model = nn.Sequential(*list(model.children())[:-1])
# model = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
# # model.fc = nn.Identity()
# # model.head = nn.Identity()
# # model.avgpool = nn.Identity()
# x = torch.randn(1, 3, 224, 224)
# out = model(x)
# print(out.size())

# for name, child in model.named_modules():
#     print(name)

# channels = 2048
# x = torch.randn(1, 2048, 7, 7)

# m1 = nn.ConvTranspose2d(channels, channels // 2, 2, stride=2)
# m2 = nn.ConvTranspose2d(channels // 2, channels // 4, 2, stride=2)
# m3 = nn.ConvTranspose2d(channels // 4, channels // 8, 2, stride=2)


# x = m1(x)
# x = m2(x)
# x = m3(x)

# print(x.size())

# o = nn.ConvTranspose2d(channels // 8, 1, 4, stride=4)
# x = o(x)

# print(x.size())

# # Assuming x is the input tensor of size (batch, channels, 3, 3)
# x = torch.randn(1, 3, 3, 3)

# # Upsample the input tensor to size (batch, channels, 7, 7)
# upsample = nn.Upsample(size=(7, 7), mode="bilinear", align_corners=False)
# x = upsample(x)

# print(x.size()[-2:] == (7, 7))

test = nn.Conv2d(2048, 1, kernel_size=1, stride=1, bias=False)
x = torch.randn(1, 2048, 7, 7)
x = test(x)
# x = torch.flatten(x, 1)
print(x.size())


in_dim = 2048
channel_reduction_factor = 8

layers = []
layers.append(
    nn.ConvTranspose2d(
        in_dim,
        in_dim // (channel_reduction_factor**1),
        kernel_size=2,
        stride=2,
    )
)
layers.append(nn.GELU())
# layers.append(
#     nn.Conv2d(in_dim, in_dim // (channel_reduction_factor**1), kernel_size=1, stride=1)
# )
# layers.append(nn.GELU())

# Layer 2
layers.append(
    nn.ConvTranspose2d(
        in_dim // (channel_reduction_factor**1),
        in_dim // (channel_reduction_factor**2),
        kernel_size=2,
        stride=2,
    )
)
layers.append(nn.GELU())
# layers.append(
#     nn.Conv2d(
#         in_dim // (channel_reduction_factor**1),
#         in_dim // (channel_reduction_factor**2),
#         kernel_size=1,
#         stride=1,
#     )
# )
# layers.append(nn.GELU())

# Layer 3
layers.append(
    nn.ConvTranspose2d(
        in_dim // (channel_reduction_factor**2),
        in_dim // (channel_reduction_factor**3),
        kernel_size=2,
        stride=2,
    )
)
layers.append(nn.GELU())
# layers.append(
#     nn.Conv2d(
#         in_dim // (channel_reduction_factor**2),
#         in_dim // (channel_reduction_factor**3),
#         kernel_size=1,
#         stride=1,
#     )
# )
# layers.append(nn.GELU())

# layer 4
layers.append(
    nn.ConvTranspose2d(
        in_dim // (channel_reduction_factor**3),
        1,
        kernel_size=2,
        stride=2,
    )
)
layers.append(nn.GELU())
upsample = nn.Sequential(*layers)
# output layer
last_layer = nn.utils.weight_norm(
    nn.Conv2d(
        1,
        1,
        kernel_size=1,
        stride=1,
        bias=False,
    )
)

x = torch.randn(1, 2048, 7, 7)
x = upsample(x)
x = last_layer(x)
x = torch.flatten(x, 1)
print(x.size())
