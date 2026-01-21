# training/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PortraitDataset(Dataset):
    """Датасет пар низкого/высокого качества"""
    def __init__(self, low_res_paths, high_res_paths, transform=None, train=True):
        self.low_res_paths = low_res_paths
        self.high_res_paths = high_res_paths
        self.train = train
        
        if transform is None:
            self.transform = A.Compose([
                A.RandomCrop(256, 256),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, 
                            saturation=0.2, hue=0.1, p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ]) if train else A.Compose([
                A.CenterCrop(256, 256),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.low_res_paths)
        
    def __getitem__(self, idx):
        # Загрузка изображений
        low_res = cv2.imread(self.low_res_paths[idx])
        high_res = cv2.imread(self.high_res_paths[idx])
        
        # Конвертация BGR -> RGB
        low_res = cv2.cvtColor(low_res, cv2.COLOR_BGR2RGB)
        high_res = cv2.cvtColor(high_res, cv2.COLOR_BGR2RGB)
        
        # Аугментация
        if self.transform:
            augmented = self.transform(image=low_res, target=high_res)
            low_res = augmented['image']
            high_res = augmented['target']
            
        return low_res, high_res

def create_degraded_image(high_res_image):
    """Создание искусственных изображений низкого качества"""
    # Добавляем различные виды деградации
    degraded = high_res_image.copy()
    
    # Размытие
    kernel_size = np.random.randint(1, 4) * 2 + 1
    degraded = cv2.GaussianBlur(degraded, (kernel_size, kernel_size), 0)
    
    # Добавление шума
    noise = np.random.normal(0, 0.05, degraded.shape).astype(np.float32)
    degraded = degraded.astype(np.float32) / 255.0
    degraded = degraded + noise
    degraded = np.clip(degraded, 0, 1)
    degraded = (degraded * 255).astype(np.uint8)
    
    # Уменьшение разрешения
    scale = np.random.uniform(0.25, 0.5)
    h, w = degraded.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    degraded = cv2.resize(degraded, (new_w, new_h), 
                         interpolation=cv2.INTER_CUBIC)
    
    # Увеличение обратно (теряем детали)
    degraded = cv2.resize(degraded, (w, h), 
                         interpolation=cv2.INTER_CUBIC)
    
    return degraded
