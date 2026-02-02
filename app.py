import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import io
import urllib.request
from pathlib import Path
import time
import math
import gc

st.set_page_config(
    page_title="RealESRGAN - Стабильная версия",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ RealESRGAN - Стабильная обработка")

# Остальной код архитектуры такой же...

def safe_process_with_tiling(model, image, tile_size=128, overlap=16):
    """Безопасная обработка с очисткой памяти"""
    try:
        # Ограничиваем максимальный размер
        max_input_size = 1024
        if max(image.size) > max_input_size:
            ratio = max_input_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            st.warning(f"⚠️ Изображение уменьшено до {new_size} для стабильности")
        
        # Конвертируем
        img_np = np.array(image).astype(np.float32) / 255.0
        h, w = img_np.shape[:2]
        
        # Вычисляем выходной размер
        out_h, out_w = h * 4, w * 4
        
        # Создаем выходной массив
        output = np.zeros((out_h, out_w, 3), dtype=np.float32)
        
        # Маленький размер тайла для экономии памяти
        tile_size = min(tile_size, 128)
        overlap = min(overlap, 16)
        
        tiles_x = math.ceil(w / tile_size)
        tiles_y = math.ceil(h / tile_size)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(tiles_y):
            for j in range(tiles_x):
                # Координаты тайла
                x1 = j * tile_size
                y1 = i * tile_size
                x2 = min(x1 + tile_size, w)
                y2 = min(y1 + tile_size, h)
                
                status_text.text(f"Тайл {i*tiles_x + j + 1}/{tiles_x * tiles_y}")
                
                if (x2 - x1) > 0 and (y2 - y1) > 0:
                    # Вырезаем тайл
                    tile = img_np[y1:y2, x1:x2]
                    
                    # Минимальный padding
                    tile_padded = np.pad(tile, 
                                       ((overlap, overlap), 
                                        (overlap, overlap), 
                                        (0, 0)), 
                                       mode='reflect')
                    
                    # Конвертируем
                    tile_tensor = torch.from_numpy(tile_padded).permute(2, 0, 1).unsqueeze(0).float()
                    
                    # Обрабатываем
                    with torch.no_grad():
                        tile_output = model(tile_tensor)
                    
                    # Конвертируем обратно
                    tile_output_np = tile_output.squeeze().permute(1, 2, 0).cpu().numpy()
                    
                    # Убираем overlap
                    overlap_scaled = overlap * 4
                    if tile_output_np.shape[0] > overlap_scaled * 2:
                        tile_output_cropped = tile_output_np[overlap_scaled:-overlap_scaled, 
                                                           overlap_scaled:-overlap_scaled]
                    else:
                        tile_output_cropped = tile_output_np
                    
                    # Копируем
                    output[y1*4:y2*4, x1*4:x2*4] = tile_output_cropped
                    
                    # Очищаем память
                    del tile_tensor, tile_output, tile_output_np
                    gc.collect()
                
                # Прогресс
                progress = (i * tiles_x + j + 1) / (tiles_x * tiles_y)
                progress_bar.progress(progress)
        
        progress_bar.empty()
        status_text.empty()
        
        # Конвертируем в PIL
        output = np.clip(output, 0, 1) * 255
        return Image.fromarray(output.astype(np.uint8))
        
    except Exception as e:
        st.error(f"Ошибка обработки: {str(e)[:200]}")
        return None

# В интерфейсе используем safe_process_with_tiling вместо process_with_tiling
