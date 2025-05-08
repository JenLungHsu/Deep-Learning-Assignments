import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# RRDB
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate=32):
        super(DenseBlock, self).__init__()
        self.layer1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(in_channels + growth_rate, growth_rate, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(in_channels + 2 * growth_rate, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out1 = F.relu(self.layer1(x))
        out2 = F.relu(self.layer2(torch.cat([x, out1], 1)))
        out3 = self.layer3(torch.cat([x, out1, out2], 1))
        return out3 * 0.2 + x  # 殘差連接

class RRDB(nn.Module):
    def __init__(self, in_channels, growth_rate=32, num_layers=3):
        super(RRDB, self).__init__()
        self.blocks = nn.Sequential(
            *[DenseBlock(in_channels, growth_rate) for _ in range(num_layers)]
        )

    def forward(self, x):
        return self.blocks(x) * 0.2 + x  # 殘差連接


# Attention technique
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        attention = torch.bmm(proj_query, proj_key)
        attention = F.softmax(attention, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out


class AdvancedTwoLayerCNN(nn.Module):
    def __init__(self):
        super(AdvancedTwoLayerCNN, self).__init__()

        # 第一層卷積
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # RRDB模塊
        self.rrdb = RRDB(in_channels=64)

        # 自注意力機制
        self.attention = SelfAttention(in_channels=64)

        # 第二層卷積
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全局平均池化層
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全連接層
        self.fc = nn.Linear(128, 50)  # 假設輸入圖像大小是64x64


    def forward(self, x):

        #第一層
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        # 第二層
        rrdb_out = self.rrdb(x)
        attention_out = self.attention(x)
        x = rrdb_out + attention_out # 將兩個模塊的輸出進行融合
        
        # 第三層
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        # 輸出層
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    



class AdvancedTwoLayerCNN_NoAll(nn.Module):
    def __init__(self):
        super(AdvancedTwoLayerCNN_NoAll, self).__init__()

        # 第一層卷積
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # RRDB模塊
        # self.rrdb = RRDB(in_channels=64)

        # 自注意力機制
        # self.attention = SelfAttention(in_channels=64)

        # 第二層卷積
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全局平均池化層
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全連接層
        self.fc = nn.Linear(128, 50)  # 假設輸入圖像大小是64x64


    def forward(self, x):

        #第一層
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        # 第二層
        # rrdb_out = self.rrdb(x)
        # attention_out = self.attention(x)
        # x = rrdb_out + attention_out # 將兩個模塊的輸出進行融合
        
        # 第三層
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        # 輸出層
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



class AdvancedTwoLayerCNN_RRDB(nn.Module):
    def __init__(self):
        super(AdvancedTwoLayerCNN_RRDB, self).__init__()

        # 第一層卷積
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # RRDB模塊
        self.rrdb = RRDB(in_channels=64)

        # 自注意力機制
        # self.attention = SelfAttention(in_channels=64)

        # 第二層卷積
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全局平均池化層
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全連接層
        self.fc = nn.Linear(128, 50)  # 假設輸入圖像大小是64x64


    def forward(self, x):

        #第一層
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        # 第二層
        rrdb_out = self.rrdb(x)
        # attention_out = self.attention(x)
        x = rrdb_out # + attention_out # 將兩個模塊的輸出進行融合
        
        # 第三層
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        # 輸出層
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

class AdvancedTwoLayerCNN_Atte(nn.Module):
    def __init__(self):
        super(AdvancedTwoLayerCNN_Atte, self).__init__()

        # 第一層卷積
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # RRDB模塊
        # self.rrdb = RRDB(in_channels=64)

        # 自注意力機制
        self.attention = SelfAttention(in_channels=64)

        # 第二層卷積
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全局平均池化層
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 全連接層
        self.fc = nn.Linear(128, 50)  # 假設輸入圖像大小是64x64


    def forward(self, x):

        #第一層
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        
        # 第二層
        # rrdb_out = self.rrdb(x)
        attention_out = self.attention(x)
        x = attention_out # 將兩個模塊的輸出進行融合
        
        # 第三層
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)

        # 輸出層
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# class AdvancedTwoLayerCNN(nn.Module):
#     def __init__(self):
#         super(AdvancedTwoLayerCNN, self).__init__()
#         # 第一層卷積
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU()
#         # RRDB模塊
#         self.rrdb = RRDB(in_channels=64)
#         # 自注意力機制
#         self.attention = SelfAttention(in_channels=64)
#         # 第二層卷積
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2)
#         self.bn2 = nn.BatchNorm2d(128)
#         # 平均池化層
#         self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
#         # 全連接層
#         self.fc = nn.Linear(128 * 32 * 32, 50)  # 假設輸入圖像大小是256x256

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.rrdb(x)
#         x = self.attention(x)
#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.fc(x)
#         return x



class SmallTwoLayerCNN(nn.Module):
    def __init__(self):
        super(SmallTwoLayerCNN, self).__init__()
        # 第一層卷積
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=0)  # padding改為0
        self.bn1 = nn.BatchNorm2d(64)
        # RRDB模塊
        self.rrdb = RRDB(in_channels=64)
        # 自注意力機制
        self.attention = SelfAttention(in_channels=64)
        # 第二層卷積
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0)  # padding改為0
        self.bn2 = nn.BatchNorm2d(128)
        # 平均池化層
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        # 全連接層
        self.fc = nn.Linear(128 * 14 * 14, 50)  # 調整全連接層的大小

    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool(x)
        x = self.bn1(x)
        x = self.rrdb(x)
        x = self.attention(x)
        x = self.conv2(x)
        x = self.avgpool(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


from torchsummary import summary
from thop import profile
import torchvision.models as models

if __name__ == '__main__':
    # model = AdvancedTwoLayerCNN()
    model = models.resnet34(pretrained=False)
    model.to('cpu')
    summary(model, (3, 256, 256), device='cpu')

    input = torch.randn(1, 3, 256, 256)  # 假設輸入形狀為 [batch_size, channels, height, width]
    # 計算FLOPS
    from fvcore.nn import FlopCountAnalysis
    flops = FlopCountAnalysis(model, input)
    print(f"Total FLOPS: {flops.total()}")  # 獲取總的FLOPS

    flops, params = profile(model, inputs=(input,))
    print(f"FLOPs: {flops/1e6}M, Parameters: {params/1e6}M")

