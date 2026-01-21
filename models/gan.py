# models/gan.py
import torch
import torch.nn as nn
from .generator import Generator
from .discriminator import Discriminator

class PortraitEnhancerGAN(nn.Module):
    """Полная GAN модель для улучшения портретов"""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        
        self.generator = Generator(in_channels, out_channels)
        self.discriminator = Discriminator()
        
        # Функции потерь
        self.criterion_gan = nn.BCELoss()
        self.criterion_l1 = nn.L1Loss()
        self.criterion_perceptual = None  # Можно добавить VGG loss
        
    def forward(self, low_res, high_res=None, training=False):
        if training:
            # Тренировка
            generated = self.generator(low_res)
            
            # Потери
            g_loss, d_loss = self.compute_losses(
                low_res, high_res, generated
            )
            
            return generated, g_loss, d_loss
            
        else:
            # Инференс
            return self.generator(low_res)
            
    def compute_losses(self, low_res, high_res, generated):
        # Потери генератора
        real_labels = torch.ones(low_res.size(0), 1, 30, 30).to(low_res.device)
        fake_labels = torch.zeros(low_res.size(0), 1, 30, 30).to(low_res.device)
        
        # GAN loss
        fake_pred = self.discriminator(low_res, generated)
        g_gan_loss = self.criterion_gan(fake_pred, real_labels)
        
        # L1 loss (content loss)
        g_l1_loss = self.criterion_l1(generated, high_res) * 100
        
        # Total generator loss
        g_loss = g_gan_loss + g_l1_loss
        
        # Discriminator loss
        real_pred = self.discriminator(low_res, high_res)
        d_real_loss = self.criterion_gan(real_pred, real_labels)
        
        fake_pred = self.discriminator(low_res, generated.detach())
        d_fake_loss = self.criterion_gan(fake_pred, fake_labels)
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        
        return g_loss, d_loss
