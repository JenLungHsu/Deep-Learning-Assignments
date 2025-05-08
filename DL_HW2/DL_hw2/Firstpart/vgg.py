import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile

# from .utils import load_state_dict_from_url
from dynamic_conv import Dynamic_conv2d

class VGG(nn.Module):

    def __init__(self, test_channel, features, num_classes=50, init_weights=True):
        super(VGG, self).__init__()
        self.test_channel = test_channel
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, num_classes),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, 4096),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        if self.test_channel is not None:
            if len(self.test_channel)==1:
                x = x.repeat(1,3,1,1) # 使用 repeat 方法複製通道
            elif len(self.test_channel)==2:
                mean_channel = x.mean(dim=1, keepdim=True)
                x = torch.cat((x, mean_channel), dim=1)
        x = self.features(x) 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, Dynamic_conv2d):
                m.update_temperature()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, is_dynamic_conv=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if is_dynamic_conv:
                conv2d = Dynamic_conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(test_channel, arch, cfg, batch_norm, is_dynamic_conv, pretrained, progress, **kwargs):
    model = VGG(test_channel, make_layers(cfgs[cfg], batch_norm=batch_norm, is_dynamic_conv=is_dynamic_conv), **kwargs)
    return model

def raw_vgg19(test_channel=None, pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(test_channel, 'raw_vgg19', 'E', True, False, pretrained, progress, **kwargs)

def dy_vgg19(test_channel=None, pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg(test_channel, 'dy_vgg19', 'E', True, True, pretrained, progress, **kwargs)


if __name__ == '__main__':
    model = raw_vgg19()
    # model = dy_vgg19()
    model.to('cpu')
    summary(model, (3, 256, 256), device='cpu')

    input = torch.randn(1, 3, 256, 256)  # 假設輸入形狀為 [batch_size, channels, height, width]
    # 計算FLOPS
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, input)
    print(f"Total FLOPS: {flops.total()}")  # 獲取總的FLOPS

    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops/1e6}M, Parameters: {params/1e6}M")

