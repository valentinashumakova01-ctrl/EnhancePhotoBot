import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class StrongGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.final = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, padding=1)
        )

    def forward(self, x):
        identity = x
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.final(x)
        return identity + 0.3 * x


class WeakDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.4),

            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.4),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.4),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)


# Дополнительные функции
def load_model(checkpoint_path, device='cpu'):
    """Загружает модель из чекпоинта"""
    model = StrongGenerator().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'generator' in checkpoint:
        model.load_state_dict(checkpoint['generator'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def enhance_image(model, image_tensor):
    """Улучшает одно изображение"""
    with torch.no_grad():
        return model(image_tensor)
