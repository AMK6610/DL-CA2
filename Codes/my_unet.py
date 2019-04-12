import torch
from torch import nn
import torch.nn.functional as F
import gc

gc.enable()


class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=50, depth=5, wf=6, padding=1,
                 batch_norm=False, kernel_size=3, max_pool_or_stride=False):
        super(UNet, self).__init__()
        self.padding = padding
        self.depth = depth
        self.kernel_size = kernel_size
        self.max_pool_or_stride = max_pool_or_stride
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            block = []
            block.append(nn.Conv2d(prev_channels, 2**(wf+i), kernel_size=self.kernel_size,
                                padding=int(padding)))
            if batch_norm:
                block.append(nn.BatchNorm2d(2**(wf+i)))
            block.append(nn.ReLU())
            
            if(self.kernel_size == 3):
                block.append(nn.Conv2d(2**(wf+i), 2**(wf+i), kernel_size=self.kernel_size,
                                    padding=int(padding)))
                if batch_norm:
                    block.append(nn.BatchNorm2d(2**(wf+i)))
                block.append(nn.ReLU())

            self.down_path.append(nn.Sequential(*block))

            prev_channels = 2**(wf+i)

        self.up_path = nn.ModuleList()
        self.conv_block = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
                                    nn.Conv2d(prev_channels, 2**(wf+i), kernel_size=1)))
            
            block = []
            block.append(nn.Conv2d(prev_channels, 2**(wf+i), kernel_size=self.kernel_size,
                                padding=int(padding)))
            block.append(nn.ReLU())
            stride = 1
            if(self.kernel_size == 3):
                if(self.max_pool_or_stride):
                    stride = 2
                block.append(nn.Conv2d(2**(wf+i), 2**(wf+i), kernel_size=self.kernel_size,
                                    padding=int(padding), stride=stride))
                block.append(nn.ReLU())
            

            self.conv_block.append(nn.Sequential(*block))

            prev_channels = 2**(wf+i)

        self.last = nn.Conv2d(prev_channels, n_classes, kernel_size=1)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                blocks.append(x)
                if(not self.max_pool_or_stride):
                    x = F.max_pool2d(x, 2)
            gc.collect()

        for i, up in enumerate(self.up_path):
            up = up(x)
            crop1 = self.center_crop(blocks[-i-1], up.shape[2:])
            x = torch.cat([up, crop1], 1)
            x = self.conv_block[i](x)
            gc.collect()

        return self.last(x)