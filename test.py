import torch
from torch import nn

from torchvision.models import resnet50
import torch
from torch import nn

model = resnet50()
# model = nn.Sequential(*list(model.children())[:-1])
model = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
# model.fc = nn.Identity()
# model.head = nn.Identity()
# model.avgpool = nn.Identity()
x = torch.randn(1, 3, 224, 224)
out = model(x)
print(out.size())

for name, child in model.named_modules():
    print(name)

channels = 2048
x = torch.randn(1, 2048, 7, 7)

m1 = nn.ConvTranspose2d(channels, channels // 2, 2, stride=2)
m2 = nn.ConvTranspose2d(channels // 2, channels // 4, 2, stride=2)
m3 = nn.ConvTranspose2d(channels // 4, channels // 8, 2, stride=2)


x = m1(x)
x = m2(x)
x = m3(x)

print(x.size())

o = nn.ConvTranspose2d(channels // 8, 1, 4, stride=4)
x = o(x)

print(x.size())

# Assuming x is the input tensor of size (batch, channels, 3, 3)
x = torch.randn(1, 3, 3, 3)

# Upsample the input tensor to size (batch, channels, 7, 7)
upsample = nn.Upsample(size=(7, 7), mode="bilinear", align_corners=False)
x = upsample(x)

print(x.size()[-2:] == (7, 7))
