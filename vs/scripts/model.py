import torch
import torch.nn as nn

class ResNet18Clone(nn.Module):
  def __init__(self, num_classes = 2):
    super().__init__()

    self.stem = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size = 7, stride =2, padding = 3, bias = False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace = True),
        nn.MaxPool2d(kernel_size =3, stride =2, padding = 1)
    )

    ## defining out blocks

    self.layer1_block1 = self.block(64,64, stride = 1, downsample = False)
    self.layer1_block2 = self.block(64,64, stride = 1, downsample = False)

    self.layer2_block1 = self.block(64,128, stride = 2, downsample = True)
    self.layer2_block2 = self.block(128,128, stride = 1, downsample = False)

    self.layer3_block1 = self.block(128,256, stride = 2, downsample = True)
    self.layer3_block2 = self.block(256,256, stride = 1,downsample = False)

    self.layer4_block1 = self.block(256,512, stride = 2, downsample = True)
    self.layer4_block2 = self.block(512,512, stride = 1, downsample = False)

    self.avgpool = nn.AdaptiveAvgPool2d((1,1)) #pooling is used to reduce dimensions oposite to interpolation
    self.fc = nn.Linear(512, num_classes)

  def block(self, in_channels, out_channels, stride, downsample):
    layers = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size =3 , stride = stride, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace = True),
        nn.Conv2d(out_channels, out_channels, kernel_size =3 , stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(out_channels)
    )

    if downsample:
      down = nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
          nn.BatchNorm2d(out_channels)
      )
    else:
      down = nn.Identity()

    return nn.ModuleDict({"layers" : layers, "downsample" : down})

  def forward_block(self, x , block):
    identity = x
    out  = block["layers"](x)
    identity = block["downsample"](identity)

    return nn.ReLU(inplace = True)(out + identity)

  def forward(self, x):
    x = self.stem(x)
    x = self.forward_block(x, self.layer1_block1)
    x = self.forward_block(x, self.layer1_block2)
    x = self.forward_block(x, self.layer2_block1)
    x = self.forward_block(x, self.layer2_block2)
    x = self.forward_block(x, self.layer3_block1)
    x = self.forward_block(x, self.layer3_block2)
    x = self.forward_block(x, self.layer4_block1)
    x = self.forward_block(x, self.layer4_block2)

    x = self.avgpool(x)
    x= torch.flatten(x,1)
    x=self.fc(x)

    return x