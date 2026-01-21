# models/discriminator.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """PatchGAN дискриминатор"""
    def __init__(self, in_channels=6, features=64):
        super().__init__()
        
        self.model = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 2
            nn.Conv2d(features, features*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 3
            nn.Conv2d(features*2, features*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 4
            nn.Conv2d(features*4, features*8, 4, stride=1, padding=1),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final layer
            nn.Conv2d(features*8, 1, 4, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, x, y):
        # Конкатенируем реальное и сгенерированное изображение
        combined = torch.cat([x, y], dim=1)
        return self.model(combined)
