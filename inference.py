# inference.py
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from models.generator import Generator

class PortraitEnhancer:
    def __init__(self, model_path='models/best_model.pth', device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        # Загрузка модели
        self.model = Generator().to(self.device)
        self.load_weights(model_path)
        self.model.eval()
        
        # Предобработка
        self.mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(self.device)
        
    def load_weights(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'generator_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['generator_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
            
    def preprocess(self, image):
        """Предобработка изображения"""
        # Конвертация в RGB
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            image = np.array(image.convert('RGB'))
        
        # Нормализация
        image = image.astype(np.float32) / 255.0
        
        # Преобразование в тензор
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image = image.to(self.device)
        
        # Стандартизация
        image = (image - self.mean) / self.std
        
        return image
        
    def postprocess(self, tensor):
        """Постобработка результата"""
        # Денормализация
        tensor = tensor * self.std + self.mean
        tensor = torch.clamp(tensor, 0, 1)
        
        # Конвертация в numpy
        image = tensor.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
        image = (image * 255).astype(np.uint8)
        
        return image
        
    def enhance(self, image, scale=4):
        """Улучшение портрета"""
        with torch.no_grad():
            # Предобработка
            input_tensor = self.preprocess(image)
            
            # Улучшение
            output_tensor = self.model(input_tensor)
            
            # Постобработка
            result = self.postprocess(output_tensor)
            
            # Увеличение размера если нужно
            if scale > 1:
                h, w = result.shape[:2]
                result = cv2.resize(result, (w*scale, h*scale), 
                                  interpolation=cv2.INTER_CUBIC)
            
            return result
            
    def enhance_face(self, image):
        """Специализированное улучшение лица"""
        # Детекция лица (можно добавить OpenCV face detection)
        # Пока просто улучшаем все изображение
        enhanced = self.enhance(image)
        
        # Дополнительная постобработка для лица
        # 1. Улучшение кожи (билатеральный фильтр)
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # 2. Улучшение глаз
        enhanced = self.enhance_eyes(enhanced)
        
        # 3. Улучшение губ
        enhanced = self.enhance_lips(enhanced)
        
        return enhanced
        
    def enhance_eyes(self, image):
        """Улучшение глаз"""
        # Можно использовать детекцию глаз и локальное улучшение
        # Упрощенная версия - увеличение резкости в области глаз
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]]) / 9.0
        
        # Предполагаем, что глаза в верхней половине изображения
        h, w = image.shape[:2]
        eye_region = image[:h//2, :]
        enhanced_eyes = cv2.filter2D(eye_region, -1, kernel)
        
        # Смешивание с оригиналом
        alpha = 0.3
        eye_region = cv2.addWeighted(eye_region, 1-alpha, 
                                    enhanced_eyes, alpha, 0)
        image[:h//2, :] = eye_region
        
        return image
        
    def enhance_lips(self, image):
        """Улучшение губ"""
        # Усиление красного канала в нижней части лица
        h, w = image.shape[:2]
        lip_region = image[h//2:, :]
        
        # Разделение каналов
        b, g, r = cv2.split(lip_region)
        
        # Усиление красного
        r = cv2.addWeighted(r, 1.2, r, 0, 0)
        r = np.clip(r, 0, 255).astype(np.uint8)
        
        # Объединение обратно
        enhanced_lips = cv2.merge([b, g, r])
        image[h//2:, :] = enhanced_lips
        
        return image

# Интеграция с вашим ботом
def enhance_image(image, enhancement_type='портрет'):
    """Обертка для использования в боте"""
    if enhancement_type == 'портрет':
        # Используем специализированную модель для портретов
        enhancer = PortraitEnhancer('models/portrait_enhancer.pth')
        result = enhancer.enhance_face(image)
    else:
        # Базовая модель для других типов
        enhancer = PortraitEnhancer('models/general_enhancer.pth')
        result = enhancer.enhance(image)
    
    return Image.fromarray(result)
