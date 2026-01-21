# models/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Остаточный блок для глубоких сетей"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out + residual

class AttentionBlock(nn.Module):
    """Блок внимания для фокуса на важных деталях"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, C, H, W = x.shape
        
        # Query, Key, Value
        q = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, H * W)
        v = self.value(x).view(batch_size, -1, H * W)
        
        # Attention matrix
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x

class Generator(nn.Module):
    """U-Net генератор с остаточными блоками и вниманием"""
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1),
            nn.PReLU()
        )
        self.encoder2 = self._make_encoder_layer(features, features*2)
        self.encoder3 = self._make_encoder_layer(features*2, features*4)
        self.encoder4 = self._make_encoder_layer(features*4, features*8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(features*8) for _ in range(6)],
            AttentionBlock(features*8)
        )
        
        # Decoder
        self.decoder4 = self._make_decoder_layer(features*8, features*4)
        self.decoder3 = self._make_decoder_layer(features*4, features*2)
        self.decoder2 = self._make_decoder_layer(features*2, features)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(features*2, features, 3, padding=1),
            nn.PReLU()
        )
        
        # Final layer
        self.final = nn.Sequential(
            nn.Conv2d(features, out_channels, 3, padding=1),
            nn.Tanh()
        )
        
    def _make_encoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        
    def _make_decoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )
        
    def forward(self, x):
        # Encoding
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Bottleneck
        b = self.bottleneck(e4)
        
        # Decoding with skip connections
        d4 = self.decoder4(b)
        d4 = torch.cat([d4, e3], dim=1)
        
        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        
        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        
        d1 = self.decoder1(d2)
        
        return self.final(d1)
