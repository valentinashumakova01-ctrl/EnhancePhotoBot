# test_model_load.py
import torch
import os
import numpy as np

print("=== Тестирование загрузки модели ===")

# Проверяем файл
weights_path = "models/enhanced_epoch_28_ratio_1.23.pth"
print(f"1. Файл существует: {os.path.exists(weights_path)}")
if os.path.exists(weights_path):
    print(f"   Размер: {os.path.getsize(weights_path) / 1024 / 1024:.2f} MB")

print(f"\n2. Версии библиотек:")
print(f"   PyTorch: {torch.__version__}")
print(f"   NumPy: {np.__version__}")

print(f"\n3. Пробуем загрузить модель...")

# Класс модели
class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(channels, channels, 3, padding=1),
        )
    def forward(self, x):
        return x + self.block(x)

class StrongGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = torch.nn.Sequential(
            torch.nn.Conv2d(3, 128, 3, padding=1),
            torch.nn.ReLU(inplace=True)
        )
        self.res_blocks = torch.nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(128, 64, 3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 3, 3, padding=1)
        )
    def forward(self, x):
        identity = x
        x = self.initial(x)
        x = self.res_blocks(x)
        x = self.final(x)
        return identity + 0.3 * x

try:
    # Способ 1: С safe_globals
    import torch.serialization
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    print(f"   ✅ Загрузка успешна (способ 1)!")
    print(f"   Ключи: {list(checkpoint.keys())}")
    
except Exception as e1:
    print(f"   ❌ Способ 1 не сработал: {e1}")
    
    try:
        # Способ 2: Через pickle
        import pickle
        with open(weights_path, 'rb') as f:
            checkpoint = pickle.load(f)
        print(f"   ✅ Загрузка успешна (способ 2 - pickle)!")
        print(f"   Тип данных: {type(checkpoint)}")
        if isinstance(checkpoint, dict):
            print(f"   Ключи: {list(checkpoint.keys())}")
            
    except Exception as e2:
        print(f"   ❌ Способ 2 не сработал: {e2}")
        
        try:
            # Способ 3: Старый способ
            checkpoint = torch.load(weights_path, map_location='cpu')
            print(f"   ✅ Загрузка успешна (способ 3 - старый)!")
            if isinstance(checkpoint, dict):
                print(f"   Ключи: {list(checkpoint.keys())}")
                
        except Exception as e3:
            print(f"   ❌ Способ 3 не сработал: {e3}")
