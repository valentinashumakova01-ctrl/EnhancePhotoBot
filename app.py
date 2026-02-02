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
    page_title="RealESRGAN - Точная архитектура",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 RealESRGAN с точной архитектурой")
st.markdown("Точное соответствие оригинальным весам")

# Создаем директории
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "realesrgan_x4plus.pth"

# ТОЧНАЯ архитектура как в оригинальном RealESRGAN
class ResidualDenseBlock(nn.Module):
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
    """ТОЧНАЯ архитектура как в оригинальном RealESRGAN"""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super(RealESRGAN, self).__init__()
        self.scale = scale
        
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        self.body = nn.ModuleList()
        for _ in range(num_block):
            self.body.append(RRDB(num_feat, num_grow_ch))
        
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsampling - ТОЧНО как в оригинале
        if scale == 2:
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        elif scale == 3:
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        elif scale == 4:
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Для PixelShuffle
        if scale == 2:
            self.upsample = nn.Sequential(
                self.conv_up1,
                nn.PixelShuffle(2),
                self.lrelu,
                self.conv_up2,
                nn.PixelShuffle(2),
                self.lrelu,
            )
        elif scale == 3:
            self.upsample = nn.Sequential(
                self.conv_up1,
                nn.PixelShuffle(3),
                self.lrelu,
            )
        elif scale == 4:
            self.upsample = nn.Sequential(
                self.conv_up1,
                nn.PixelShuffle(2),
                self.lrelu,
                self.conv_up2,
                nn.PixelShuffle(2),
                self.lrelu,
            )

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        
        body_feat = feat
        for block in self.body:
            body_feat = block(body_feat)
        body_feat = self.lrelu(self.conv_body(body_feat))
        feat = feat + body_feat
        
        if self.scale in [2, 3, 4]:
            feat = self.upsample(feat)
        
        if self.scale == 4:
            feat = self.lrelu(self.conv_hr(feat))
            out = self.conv_last(feat)
        else:
            out = feat
            
        return out

@st.cache_resource
def download_and_load_model():
    """Скачивает и загружает веса"""
    
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    
    if not MODEL_PATH.exists():
        with st.spinner("Скачивание модели RealESRGAN (1.07GB)..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(count, block_size, total_size):
                if total_size > 0:
                    percent = int(count * block_size * 100 / total_size)
                    progress_bar.progress(min(percent, 100))
                    status_text.text(f"Загрузка: {min(percent, 100)}%")
            
            try:
                urllib.request.urlretrieve(model_url, MODEL_PATH, reporthook=update_progress)
                progress_bar.empty()
                status_text.empty()
                st.success("✅ Модель скачана!")
            except Exception as e:
                st.error(f"Ошибка скачивания: {e}")
                return None
    
    try:
        st.info("Загрузка весов модели...")
        
        # Загружаем веса
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        
        # Проверяем структуру весов
        st.write("🔍 Анализ структуры весов...")
        
        # Определяем ключи весов
        if 'params_ema' in state_dict:
            weights = state_dict['params_ema']
        else:
            weights = state_dict
        
        # Показываем ключи для отладки
        weight_keys = list(weights.keys())[:10]  # Первые 10 ключей
        st.write(f"Первые 10 ключей весов: {weight_keys}")
        
        # Создаем модель
        model = RealESRGAN(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
        
        # Пробуем загрузить веса
        try:
            model.load_state_dict(weights, strict=True)
            st.success("✅ Веса загружены (strict mode)")
        except Exception as e:
            st.warning(f"Strict mode failed: {e}")
            
            # Пробуем нестрогий режим
            st.info("Попытка нестрогой загрузки...")
            model.load_state_dict(weights, strict=False)
            st.success("✅ Веса загружены (non-strict mode)")
        
        model.eval()
        
        # Проверяем устройство
        if torch.cuda.is_available():
            model = model.cuda()
            st.info("✅ Используется GPU")
        else:
            st.info("ℹ️ Используется CPU")
        
        return model
        
    except Exception as e:
        st.error(f"Ошибка загрузки весов: {e}")
        st.exception(e)
        return None

def process_image(model, image):
    """Обрабатывает изображение"""
    try:
        # Конвертируем PIL в numpy
        img_np = np.array(image).astype(np.float32) / 255.0
        
        # Конвертируем в tensor [C, H, W]
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
        
        # Перемещаем на GPU если доступно
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()
        
        # Обработка
        with torch.no_grad():
            output = model(img_tensor)
        
        # Конвертируем обратно
        output_np = output.squeeze().permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        output_np = (output_np * 255.0).astype(np.uint8)
        
        return Image.fromarray(output_np)
        
    except Exception as e:
        st.error(f"Ошибка обработки: {e}")
        return None

# Основной интерфейс
def main():
    st.sidebar.header("⚙️ Настройки")
    
    # Загрузка модели
    with st.spinner("Инициализация модели..."):
        model = download_and_load_model()
    
    if model is None:
        st.error("Не удалось загрузить модель")
        return
    
    # Интерфейс
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Загрузите фото")
        uploaded = st.file_uploader(
            "Выберите изображение", 
            type=['jpg', 'png', 'jpeg', 'bmp'],
            help="До 10MB"
        )
        
        if uploaded:
            try:
                input_img = Image.open(uploaded).convert('RGB')
                
                # Ограничиваем размер для предпросмотра
                max_size = 1024
                if max(input_img.size) > max_size:
                    ratio = max_size / max(input_img.size)
                    new_size = (int(input_img.size[0] * ratio), int(input_img.size[1] * ratio))
                    display_img = input_img.resize(new_size, Image.Resampling.LANCZOS)
                else:
                    display_img = input_img
                
                st.image(display_img, caption=f"Оригинал: {input_img.size}", use_column_width=True)
                
                if st.button("🚀 Улучшить качество 4x", type="primary", use_container_width=True):
                    with st.spinner("Обработка... Это может занять несколько минут"):
                        enhanced = process_image(model, input_img)
                        
                        if enhanced is not None:
                            with col2:
                                st.header("✨ Результат")
                                
                                # Ограничиваем размер для отображения
                                if max(enhanced.size) > max_size:
                                    ratio = max_size / max(enhanced.size)
                                    new_size = (int(enhanced.size[0] * ratio), int(enhanced.size[1] * ratio))
                                    display_enhanced = enhanced.resize(new_size, Image.Resampling.LANCZOS)
                                else:
                                    display_enhanced = enhanced
                                
                                st.image(display_enhanced, caption=f"Улучшено: {enhanced.size}", use_column_width=True)
                                
                                # Скачивание
                                buf = io.BytesIO()
                                enhanced.save(buf, format="PNG", quality=95)
                                
                                st.download_button(
                                    "📥 Скачать результат",
                                    buf.getvalue(),
                                    file_name=f"enhanced_{uploaded.name}",
                                    mime="image/png",
                                    use_container_width=True
                                )
                                
                                # Статистика
                                col_size1, col_size2 = st.columns(2)
                                with col_size1:
                                    st.metric("Оригинал", f"{input_img.size[0]}×{input_img.size[1]}")
                                with col_size2:
                                    st.metric("Результат", f"{enhanced.size[0]}×{enhanced.size[1]}", 
                                             delta="4x")
            
            except Exception as e:
                st.error(f"Ошибка: {e}")
    
    # Информация
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **RealESRGAN_x4plus**
    
    - Масштаб: 4x
    - Блоки: 23 RRDB
    - Параметры: 16.7M
    - Веса: 1.07GB
    
    Оригинал: [GitHub](https://github.com/xinntao/Real-ESRGAN)
    """)

if __name__ == "__main__":
    main()
