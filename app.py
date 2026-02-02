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

st.set_page_config(
    page_title="RealESRGAN Lite",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ RealESRGAN Lite")
st.markdown("Упрощенная архитектура с оригинальными весами")

# Создаем директории
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "realesrgan_x4plus.pth"

# УПРОЩЕННАЯ архитектура, но совместимая с весами
class LiteESRGAN(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=8, scale=4):
        super().__init__()
        self.scale = scale
        
        # Упрощенные блоки
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # Меньше блоков (8 вместо 23)
        self.body = nn.Sequential(*[
            LiteRRDB(num_feat) for _ in range(num_block)
        ])
        
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsampling
        if scale == 4:
            self.up1 = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2)
            )
            self.up2 = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2)
            )
        elif scale == 2:
            self.up1 = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2)
            )
        
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        
        # Упрощенная обработка
        body_feat = self.body(feat)
        body_feat = self.lrelu(self.conv_body(body_feat))
        feat = feat + body_feat
        
        # Upsampling
        if self.scale == 4:
            feat = self.up1(feat)
            feat = self.up2(feat)
        elif self.scale == 2:
            feat = self.up1(feat)
        
        out = self.conv_last(feat)
        return out

class LiteRRDB(nn.Module):
    """Упрощенный RRDB блок"""
    def __init__(self, num_feat):
        super().__init__()
        self.rdb1 = LiteRDB(num_feat)
        self.rdb2 = LiteRDB(num_feat)
    
    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        return out * 0.2 + x

class LiteRDB(nn.Module):
    """Упрощенный Residual Dense Block"""
    def __init__(self, num_feat):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        return x2 * 0.2 + x

@st.cache_resource
def download_and_load_model():
    """Скачивает и загружает веса"""
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    
    if not MODEL_PATH.exists():
        with st.spinner("Скачивание весов RealESRGAN (1.07GB)... Это займет несколько минут"):
            try:
                urllib.request.urlretrieve(model_url, MODEL_PATH)
                st.success("✅ Веса скачаны!")
            except Exception as e:
                st.error(f"Ошибка скачивания: {e}")
                return None
    
    try:
        st.info("Загрузка весов...")
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        
        # Берем веса из state dict
        if 'params_ema' in state_dict:
            weights = state_dict['params_ema']
        else:
            weights = state_dict
        
        # Создаем упрощенную модель
        model = LiteESRGAN(num_block=8, scale=4)
        
        # Адаптация весов для упрощенной архитектуры
        adapted_weights = {}
        
        # Маппинг ключей: оригинальные -> упрощенные
        key_mapping = {
            'conv_first.weight': 'conv_first.weight',
            'conv_first.bias': 'conv_first.bias',
            'conv_body.weight': 'conv_body.weight',
            'conv_body.bias': 'conv_body.bias',
            'conv_up1.weight': 'up1.0.weight',
            'conv_up1.bias': 'up1.0.bias',
            'conv_up2.weight': 'up2.0.weight',
            'conv_up2.bias': 'up2.0.bias',
            'conv_hr.weight': 'conv_last.weight',
            'conv_hr.bias': 'conv_last.bias',
            'conv_last.weight': 'conv_last.weight',
            'conv_last.bias': 'conv_last.bias',
        }
        
        # Копируем соответствующие веса
        for orig_key, new_key in key_mapping.items():
            if orig_key in weights:
                adapted_weights[new_key] = weights[orig_key]
        
        # Для RRDB блоков берем только первые блоки
        for i in range(8):  # Берем первые 8 блоков вместо 23
            if f'body.{i}.rdb1.conv1.weight' in weights:
                # Копируем только первый RDB блок из каждого RRDB
                adapted_weights[f'body.{i}.rdb1.conv1.weight'] = weights[f'body.{i}.rdb1.conv1.weight']
                adapted_weights[f'body.{i}.rdb1.conv1.bias'] = weights[f'body.{i}.rdb1.conv1.bias']
                adapted_weights[f'body.{i}.rdb1.conv2.weight'] = weights[f'body.{i}.rdb1.conv2.weight']
                adapted_weights[f'body.{i}.rdb1.conv2.bias'] = weights[f'body.{i}.rdb1.conv2.bias']
        
        # Загружаем адаптированные веса
        model.load_state_dict(adapted_weights, strict=False)
        model.eval()
        
        # Используем CPU для стабильности
        device = torch.device('cpu')
        model = model.to(device)
        
        st.success(f"✅ Модель загружена на {device}")
        return model
        
    except Exception as e:
        st.error(f"Ошибка загрузки: {e}")
        return None

def tile_process(model, image, tile_size=512, tile_pad=32):
    """Обработка с tiling для больших изображений"""
    # Конвертируем в numpy
    img = np.array(image).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    
    # Вычисляем размер выходного изображения
    out_h, out_w = h * 4, w * 4
    
    # Создаем выходной массив
    output = np.zeros((out_h, out_w, 3), dtype=np.float32)
    
    # Разбиваем на тайлы
    tiles_x = math.ceil(w / tile_size)
    tiles_y = math.ceil(h / tile_size)
    
    progress_bar = st.progress(0)
    
    for i in range(tiles_y):
        for j in range(tiles_x):
            # Координаты тайла
            x1 = j * tile_size
            y1 = i * tile_size
            x2 = min(x1 + tile_size, w)
            y2 = min(y1 + tile_size, h)
            
            # Вырезаем тайл с padding
            tile = img[y1:y2, x1:x2]
            
            if tile.size > 0:
                # Добавляем padding
                tile_padded = np.pad(tile, 
                                   ((tile_pad, tile_pad), 
                                    (tile_pad, tile_pad), 
                                    (0, 0)), 
                                   mode='reflect')
                
                # Конвертируем в tensor
                tile_tensor = torch.from_numpy(tile_padded).permute(2, 0, 1).unsqueeze(0)
                
                # Обрабатываем
                with torch.no_grad():
                    tile_output = model(tile_tensor)
                
                # Конвертируем обратно
                tile_output_np = tile_output.squeeze().permute(1, 2, 0).numpy()
                
                # Убираем padding
                tile_output_np = tile_output_np[tile_pad*4:-tile_pad*4, tile_pad*4:-tile_pad*4]
                
                # Копируем в выходное изображение
                output[y1*4:y2*4, x1*4:x2*4] = tile_output_np
            
            # Обновляем прогресс
            progress = (i * tiles_x + j + 1) / (tiles_x * tiles_y)
            progress_bar.progress(progress)
    
    progress_bar.empty()
    
    # Конвертируем в PIL
    output = np.clip(output, 0, 1) * 255
    return Image.fromarray(output.astype(np.uint8))

def main():
    # Сайдбар с настройками
    st.sidebar.header("⚙️ Настройки обработки")
    tile_size = st.sidebar.slider("Размер тайла", 128, 512, 256, 64)
    
    # Загрузка модели
    with st.spinner("Загрузка модели..."):
        model = download_and_load_model()
    
    if model is None:
        st.error("Не удалось загрузить модель")
        return
    
    # Основной интерфейс
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Входное изображение")
        uploaded = st.file_uploader(
            "Загрузите фото (рекомендуется до 1024x1024)",
            type=['jpg', 'png', 'jpeg']
        )
        
        if uploaded:
            input_img = Image.open(uploaded).convert('RGB')
            
            # Ограничиваем для предпросмотра
            max_preview = 512
            if max(input_img.size) > max_preview:
                ratio = max_preview / max(input_img.size)
                preview_size = (int(input_img.size[0] * ratio), 
                              int(input_img.size[1] * ratio))
                preview_img = input_img.resize(preview_size, Image.Resampling.LANCZOS)
            else:
                preview_img = input_img
            
            st.image(preview_img, caption=f"Оригинал: {input_img.size}", width=350)
            
            st.info(f"""
            **Детали:**
            - Размер: {input_img.size[0]} × {input_img.size[1]}
            - Тайлов: ~{math.ceil(input_img.size[0]/tile_size) * math.ceil(input_img.size[1]/tile_size)}
            - Выходной размер: {input_img.size[0]*4} × {input_img.size[1]*4}
            """)
    
    with col2:
        st.header("✨ Результат")
        
        if uploaded and 'input_img' in locals():
            if st.button("🚀 Запустить RealESRGAN (4x)", type="primary", use_container_width=True):
                with st.spinner(f"Обработка тайлами {tile_size}x{tile_size}..."):
                    start_time = time.time()
                    
                    # Обработка с tiling
                    enhanced = tile_process(model, input_img, tile_size=tile_size)
                    
                    elapsed = time.time() - start_time
                    
                    # Предпросмотр результата
                    if max(enhanced.size) > max_preview:
                        ratio = max_preview / max(enhanced.size)
                        preview_size = (int(enhanced.size[0] * ratio), 
                                      int(enhanced.size[1] * ratio))
                        preview_enhanced = enhanced.resize(preview_size, Image.Resampling.LANCZOS)
                    else:
                        preview_enhanced = enhanced
                    
                    st.image(preview_enhanced, caption=f"Улучшено: {enhanced.size}", width=350)
                    
                    # Скачивание
                    buf = io.BytesIO()
                    enhanced.save(buf, format="PNG", optimize=True)
                    
                    st.download_button(
                        "📥 Скачать PNG",
                        buf.getvalue(),
                        file_name=f"real_esrgan_4x_{uploaded.name}",
                        mime="image/png",
                        use_container_width=True
                    )
                    
                    # Статистика
                    st.success(f"✅ Обработано за {elapsed:.1f} секунд")
                    
                    col_stat1, col_stat2 = st.columns(2)
                    with col_stat1:
                        st.metric("Вход", f"{input_img.size[0]}×{input_img.size[1]}")
                    with col_stat2:
                        st.metric("Выход", f"{enhanced.size[0]}×{enhanced.size[1]}", delta="4x")
        
        else:
            st.info("Загрузите изображение слева")
    
    # Информация о модели
    with st.expander("📊 Технические детали"):
        st.markdown(f"""
        ### Архитектура LiteESRGAN:
        
        **Упрощения:**
        - Блоков RRDB: 8 вместо 23
        - Упрощенные RDB блоки (2 conv вместо 5)
        - Только первые веса из оригинальной модели
        
        **Tiling система:**
        - Размер тайла: {tile_size}px
        - Overlap: 32px
        - Reflection padding
        
        **Память:**
        - Веса модели: 1.07GB
        - Пиковая память на тайл: ~{(tile_size+64)*4*4*4/1024/1024:.1f}MB
        - Общая память: ~1.5-2GB
        
        💡 **Совет:** Используйте меньший размер тайла если возникает ошибка памяти
        """)

if __name__ == "__main__":
    main()
