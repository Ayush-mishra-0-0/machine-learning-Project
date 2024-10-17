import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class C3Block(nn.Module):
    def __init__(self, channels):
        super(C3Block, self).__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
        self.conv3 = ConvBlock(channels, channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        return out + residual

class SPPBlock(nn.Module):
    def __init__(self, channels):
        super(SPPBlock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.conv = ConvBlock(channels * 4, channels)

    def forward(self, x):
        pool1 = self.pool1(x)
        pool2 = self.pool2(x)
        pool3 = self.pool3(x)
        concat = torch.cat([x, pool1, pool2, pool3], dim=1)
        return self.conv(concat)

class PSAAttention(nn.Module):
    def __init__(self, channels):
        super(PSAAttention, self).__init__()
        self.conv3x3 = nn.Conv2d(channels, channels // 4, kernel_size=3, padding=1, groups=1)
        self.conv5x5 = nn.Conv2d(channels, channels // 4, kernel_size=5, padding=2, groups=4)
        self.conv7x7 = nn.Conv2d(channels, channels // 4, kernel_size=7, padding=3, groups=8)
        self.conv9x9 = nn.Conv2d(channels, channels // 4, kernel_size=9, padding=4, groups=16)
        
        self.se_blocks = nn.ModuleList([SEBlock(channels // 4) for _ in range(4)])
        self.concat_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(x)
        conv7x7 = self.conv7x7(x)
        conv9x9 = self.conv9x9(x)
        
        conv3x3 = self.se_blocks[0](conv3x3)
        conv5x5 = self.se_blocks[1](conv5x5)
        conv7x7 = self.se_blocks[2](conv7x7)
        conv9x9 = self.se_blocks[3](conv9x9)
        
        feats = torch.cat([conv3x3, conv5x5, conv7x7, conv9x9], dim=1)
        feats = self.concat_conv(feats)
        attention = self.softmax(feats)
        
        out = x * attention + x
        return out

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class PSAN(nn.Module):
    def __init__(self, num_classes):
        super(PSAN, self).__init__()
        
        # Backbone
        self.backbone = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            C3Block(64),
            ConvBlock(64, 128),
            C3Block(128),
            ConvBlock(128, 256),
            C3Block(256),
            ConvBlock(256, 512),
            C3Block(512),
            SPPBlock(512)
        )
        
        # Neck
        self.neck = nn.Sequential(
            ConvBlock(512, 256),
            C3Block(256),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(256, 128),
            C3Block(128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ConvBlock(128, 64),
            C3Block(64)
        )
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, num_classes)
        )
        
        # PSA Attention
        self.psa_attention = PSAAttention(64)
        
    def forward(self, x):
        # Backbone
        x = self.backbone(x)
        
        # Neck
        x = self.neck(x)
        
        # PSA Attention
        x = self.psa_attention(x)
        
        # Head
        x = self.head(x)
        
        return x

# Example usage
if __name__ == "__main__":
    model = PSAN(num_classes=12)
    input_tensor = torch.randn(1, 3, 224, 224)
    output = model(input_tensor)
    print(output)
    print(f"Output shape: {output.shape}")