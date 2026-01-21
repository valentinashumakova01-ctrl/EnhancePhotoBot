# training/train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from tqdm import tqdm

from models.gan import PortraitEnhancerGAN
from training.dataset import PortraitDataset

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Модель
        self.model = PortraitEnhancerGAN().to(self.device)
        
        # Оптимизаторы
        self.optimizer_g = optim.Adam(
            self.model.generator.parameters(),
            lr=config['lr_g'],
            betas=(0.5, 0.999)
        )
        self.optimizer_d = optim.Adam(
            self.model.discriminator.parameters(),
            lr=config['lr_d'],
            betas=(0.5, 0.999)
        )
        
        # Планировщик скорости обучения
        self.scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_g, T_max=config['epochs']
        )
        self.scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer_d, T_max=config['epochs']
        )
        
        # Датасеты
        self.train_dataset = PortraitDataset(
            config['train_low_res_paths'],
            config['train_high_res_paths'],
            train=True
        )
        
        self.val_dataset = PortraitDataset(
            config['val_low_res_paths'],
            config['val_high_res_paths'],
            train=False
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        # TensorBoard
        self.writer = SummaryWriter(config['log_dir'])
        
        # Лучшие веса
        self.best_psnr = 0
        
    def train_epoch(self, epoch):
        self.model.train()
        total_g_loss = 0
        total_d_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (low_res, high_res) in enumerate(pbar):
            low_res = low_res.to(self.device)
            high_res = high_res.to(self.device)
            
            # Тренировка дискриминатора
            self.optimizer_d.zero_grad()
            generated = self.model.generator(low_res)
            
            # Отдельно вычисляем потери D
            real_labels = torch.ones(low_res.size(0), 1, 30, 30).to(self.device)
            fake_labels = torch.zeros(low_res.size(0), 1, 30, 30).to(self.device)
            
            real_pred = self.model.discriminator(low_res, high_res)
            d_real_loss = self.model.criterion_gan(real_pred, real_labels)
            
            fake_pred = self.model.discriminator(low_res, generated.detach())
            d_fake_loss = self.model.criterion_gan(fake_pred, fake_labels)
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            self.optimizer_d.step()
            
            # Тренировка генератора
            self.optimizer_g.zero_grad()
            generated = self.model.generator(low_res)
            
            fake_pred = self.model.discriminator(low_res, generated)
            g_gan_loss = self.model.criterion_gan(fake_pred, real_labels)
            
            g_l1_loss = self.model.criterion_l1(generated, high_res) * 100
            g_loss = g_gan_loss + g_l1_loss
            
            g_loss.backward()
            self.optimizer_g.step()
            
            # Обновление прогресс-бара
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            
            pbar.set_postfix({
                'G Loss': f'{g_loss.item():.4f}',
                'D Loss': f'{d_loss.item():.4f}',
                'L1 Loss': f'{g_l1_loss.item():.4f}'
            })
            
            # Логирование в TensorBoard
            if batch_idx % 100 == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/Generator', g_loss.item(), global_step)
                self.writer.add_scalar('Loss/Discriminator', d_loss.item(), global_step)
                self.writer.add_scalar('Loss/L1', g_l1_loss.item(), global_step)
                
                # Визуализация изображений
                if batch_idx == 0:
                    self.writer.add_images('Input', low_res[:4], global_step)
                    self.writer.add_images('Generated', generated[:4], global_step)
                    self.writer.add_images('Target', high_res[:4], global_step)
        
        return total_g_loss / len(self.train_loader), total_d_loss / len(self.train_loader)
    
    def validate(self, epoch):
        self.model.eval()
        total_psnr = 0
        total_ssim = 0
        
        with torch.no_grad():
            for low_res, high_res in self.val_loader:
                low_res = low_res.to(self.device)
                high_res = high_res.to(self.device)
                
                generated = self.model.generator(low_res)
                
                # Вычисление метрик качества
                for i in range(generated.size(0)):
                    psnr = self.calculate_psnr(generated[i], high_res[i])
                    ssim = self.calculate_ssim(generated[i], high_res[i])
                    total_psnr += psnr
                    total_ssim += ssim
        
        avg_psnr = total_psnr / len(self.val_dataset)
        avg_ssim = total_ssim / len(self.val_dataset)
        
        # Логирование
        self.writer.add_scalar('Metrics/PSNR', avg_psnr, epoch)
        self.writer.add_scalar('Metrics/SSIM', avg_ssim, epoch)
        
        # Сохранение лучшей модели
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
            self.save_checkpoint(epoch, 'best_model.pth')
            
        return avg_psnr, avg_ssim
    
    def calculate_psnr(self, img1, img2):
        """Вычисление PSNR"""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()
    
    def calculate_ssim(self, img1, img2):
        """Вычисление SSIM"""
        # Упрощенная версия SSIM
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu1 = torch.mean(img1)
        mu2 = torch.mean(img2)
        
        sigma1 = torch.std(img1)
        sigma2 = torch.std(img2)
        sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 ** 2 + sigma2 ** 2 + C2))
        
        return ssim.item()
    
    def save_checkpoint(self, epoch, filename):
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.model.generator.state_dict(),
            'discriminator_state_dict': self.model.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'best_psnr': self.best_psnr,
            'config': self.config
        }
        
        torch.save(checkpoint, os.path.join(self.config['checkpoint_dir'], filename))
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.best_psnr = checkpoint['best_psnr']
        
        return checkpoint['epoch']
    
    def train(self):
        for epoch in range(self.config['epochs']):
            # Тренировка
            avg_g_loss, avg_d_loss = self.train_epoch(epoch)
            
            # Валидация
            avg_psnr, avg_ssim = self.validate(epoch)
            
            # Планировщик
            self.scheduler_g.step()
            self.scheduler_d.step()
            
            # Логирование
            print(f'Epoch {epoch}: '
                  f'G Loss: {avg_g_loss:.4f}, '
                  f'D Loss: {avg_d_loss:.4f}, '
                  f'PSNR: {avg_psnr:.2f} dB, '
                  f'SSIM: {avg_ssim:.4f}')
            
            # Сохранение чекпоинта
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')
