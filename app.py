import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import io
import urllib.request
import os
from pathlib import Path
import json

st.set_page_config(
    page_title="ESRGAN с реальными весами",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 ESRGAN с предобученными весами")
st.markdown("Загружает настоящие веса модели RealESRGAN")

# Создаем директории
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "realesrgan_x4plus.pth"
MODEL_CONFIG_PATH = MODEL_DIR / "model_config.json"

# Определяем точную архитектуру из оригинальной модели
class ResidualDenseBlock(nn.Module):
    """Residual Dense Block"""
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RealESRGAN(nn.Module):
    """Архитектура RealESRGAN как в оригинале"""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super(RealESRGAN, self).__init__()
        self.scale = scale
        
        # Первая конволюция
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # Основные блоки
        self.body = nn.ModuleList()
        for _ in range(num_block):
            self.body.append(RRDB(num_feat, num_grow_ch))
        
        # Конволюция тела
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsampling
        if scale == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
            )
        elif scale == 3:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 9, 3, 1, 1),
                nn.PixelShuffle(3),
            )
        elif scale == 4:
            self.upsample = nn.Sequential(
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                nn.PixelShuffle(2),
            )
        
        # Финальные слои
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        
        # Основные блоки
        body_feat = feat.clone()
        for block in self.body:
            body_feat = block(body_feat)
        body_feat = self.lrelu(self.conv_body(body_feat))
        feat = feat + body_feat
        
        # Upsampling
        feat = self.upsample(feat)
        
        # Финальные слои
        feat = self.lrelu(self.conv_hr(feat))
        out = self.conv_last(feat)
        return out

@st.cache_resource
def download_and_load_model():
    """Скачивает и загружает настоящие веса"""
    
    # URL оригинальных весов
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    
    # Скачиваем модель если нужно
    if not MODEL_PATH.exists():
        with st.spinner("Скачивание модели RealESRGAN (1.07GB)..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                progress_bar.progress(percent)
                status_text.text(f"Загрузка: {percent}%")
            
            try:
                urllib.request.urlretrieve(model_url, MODEL_PATH, reporthook=update_progress)
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Модель скачана!")
            except Exception as e:
                st.error(f"Ошибка скачивания: {e}")
                return None
    
    try:
        # Загружаем веса
        st.info("Загрузка весов модели...")
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        
        # Создаем модель с правильными параметрами
        model = RealESRGAN(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
        
        # Загружаем веса
        model.load_state_dict(state_dict['params_ema'] if 'params_ema' in state_dict else state_dict, strict=True)
        model.eval()
        
        # Сохраняем конфиг
        config = {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 23,
            'num_grow_ch': 32,
            'scale': 4
        }
        with open(MODEL_CONFIG_PATH, 'w') as f:
            json.dump(config, f)
        
        st.success("✅ Веса успешно загружены!")
        return model
        
    except Exception as e:
        st.error(f"Ошибка загрузки весов: {e}")
        st.exception(e)
        return None

def process_image(model, image, scale=4):
    """Обрабатывает изображение через модель"""
    # Конвертируем PIL в numpy
    img_np = np.array(image).astype(np.float32) / 255.0
    
    # Конвертируем в tensor
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
    
    # Обработка
    with torch.no_grad():
        output = model(img_tensor)
    
    # Конвертируем обратно
    output_np = output.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    output_np = (output_np * 255.0).astype(np.uint8)
    
    return Image.fromarray(output_np)

# Основной интерфейс
def main():
    # Загрузка модели
    with st.spinner("Инициализация модели..."):
        model = download_and_load_model()
    
    if model is None:
        st.error("Не удалось загрузить модель. Проверьте интернет соединение.")
        return
    
    # Интерфейс
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Загрузите фото")
        uploaded = st.file_uploader("Выберите изображение", type=['jpg', 'png', 'jpeg'])
        
        if uploaded:
            input_img = Image.open(uploaded).convert('RGB')
            st.image(input_img, caption=f"Оригинал: {input_img.size}", use_column_width=True)
            
            if st.button("🚀 Улучшить качество (4x)", type="primary"):
                with st.spinner("Обработка..."):
                    try:
                        enhanced = process_image(model, input_img, scale=4)
                        
                        with col2:
                            st.header("✨ Результат")
                            st.image(enhanced, caption=f"Улучшено: {enhanced.size}", use_column_width=True)
                            
                            # Скачивание
                            buf = io.BytesIO()
                            enhanced.save(buf, format="PNG", quality=95)
                            
                            st.download_button(
                                "📥 Скачать",
                                buf.getvalue(),
                                file_name=f"enhanced_{uploaded.name}",
                                mime="image/png"
                            )
                            
                            # Статистика
                            st.metric("Увеличение", "4x", 
                                     f"{enhanced.size[0]//input_img.size[0]}×")
                            
                    except Exception as e:
                        st.error(f"Ошибка обработки: {e}")
    
    # Информация о модели
    st.sidebar.header("ℹ️ Информация о модели")
    st.sidebar.info("""
    **RealESRGAN_x4plus**
    
    Параметры:
    - Блоков: 23 RRDB
    - Фичей: 64
    - Scale: 4x
    
    Веса скачаны с GitHub:
    https://github.com/xinntao/Real-ESRGAN
    """)

if __name__ == "__main__":
    main()
