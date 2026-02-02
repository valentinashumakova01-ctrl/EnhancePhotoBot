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
    page_title="RealESRGAN - Точная архитектура",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 RealESRGAN - Полная совместимость")
st.markdown("Точная архитектура с оригинальными весами")

# Создаем директории
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "realesrgan_x4plus.pth"

# ТОЧНАЯ архитектура как в оригинале
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
    """ТОЧНАЯ архитектура как в оригинале"""
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super(RealESRGAN, self).__init__()
        self.scale = scale
        
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        self.body = nn.ModuleList()
        for _ in range(num_block):
            self.body.append(RRDB(num_feat, num_grow_ch))
        
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Конволюции для upsampling (точные имена как в весах)
        if scale == 4:
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif scale == 2:
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        
        body_feat = feat
        for block in self.body:
            body_feat = block(body_feat)
        body_feat = self.lrelu(self.conv_body(body_feat))
        feat = feat + body_feat
        
        if self.scale == 4:
            # Первый upsampling
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
            # Второй upsampling
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
            feat = self.lrelu(self.conv_hr(feat))
            out = self.conv_last(feat)
        elif self.scale == 2:
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
            out = self.conv_last(feat)
        else:
            out = feat
            
        return out

@st.cache_resource
def download_and_load_model():
    """Скачивает и загружает веса"""
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    
    if not MODEL_PATH.exists():
        with st.spinner("Скачивание весов RealESRGAN (1.07GB)..."):
            try:
                urllib.request.urlretrieve(model_url, MODEL_PATH)
                st.success("✅ Веса скачаны!")
            except Exception as e:
                st.error(f"Ошибка скачивания: {e}")
                return None
    
    try:
        st.info("Загрузка весов...")
        state_dict = torch.load(MODEL_PATH, map_location='cpu')
        
        # Проверяем доступные ключи
        if 'params_ema' in state_dict:
            weights = state_dict['params_ema']
        else:
            weights = state_dict
        
        # Создаем модель с ТОЧНЫМИ параметрами как в оригинале
        model = RealESRGAN(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4
        )
        
        # Загружаем веса
        model.load_state_dict(weights, strict=True)
        model.eval()
        
        st.success("✅ Модель успешно загружена!")
        return model
        
    except Exception as e:
        st.error(f"Ошибка загрузки весов: {e}")
        
        # Пробуем нестрогую загрузку
        try:
            st.info("Попытка нестрогой загрузки...")
            model.load_state_dict(weights, strict=False)
            st.success("✅ Модель загружена (нестрогий режим)")
            return model
        except:
            return None

def process_with_tiling(model, image, tile_size=256, overlap=32):
    """Обработка с tiling для экономии памяти"""
    # Конвертируем в numpy
    img_np = np.array(image).astype(np.float32) / 255.0
    h, w = img.shape[:2]
    
    # Вычисляем выходной размер
    out_h, out_w = h * 4, w * 4
    
    # Создаем выходной массив
    output = np.zeros((out_h, out_w, 3), dtype=np.float32)
    
    # Разбиваем на тайлы
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
            
            status_text.text(f"Обработка тайла {i*tiles_x + j + 1}/{tiles_x * tiles_y}")
            
            if (x2 - x1) > 0 and (y2 - y1) > 0:
                # Вырезаем тайл
                tile = img_np[y1:y2, x1:x2]
                
                # Добавляем overlap
                tile_padded = np.pad(tile, 
                                   ((overlap, overlap), 
                                    (overlap, overlap), 
                                    (0, 0)), 
                                   mode='reflect')
                
                # Конвертируем в tensor
                tile_tensor = torch.from_numpy(tile_padded).permute(2, 0, 1).unsqueeze(0).float()
                
                # Обрабатываем
                with torch.no_grad():
                    tile_output = model(tile_tensor)
                
                # Конвертируем обратно
                tile_output_np = tile_output.squeeze().permute(1, 2, 0).cpu().numpy()
                
                # Убираем overlap (умножаем на 4 т.к. scale=4)
                tile_output_cropped = tile_output_np[overlap*4:-overlap*4, overlap*4:-overlap*4]
                
                # Копируем в выходное изображение
                output[y1*4:y2*4, x1*4:x2*4] = tile_output_cropped
            
            # Обновляем прогресс
            progress = (i * tiles_x + j + 1) / (tiles_x * tiles_y)
            progress_bar.progress(progress)
    
    progress_bar.empty()
    status_text.empty()
    
    # Конвертируем в PIL
    output = np.clip(output, 0, 1) * 255
    return Image.fromarray(output.astype(np.uint8))

def main():
    # Настройки
    st.sidebar.header("⚙️ Настройки")
    tile_size = st.sidebar.selectbox("Размер тайла", [128, 192, 256], index=1)
    use_tiling = st.sidebar.checkbox("Использовать tiling", True, 
                                     help="Обязательно для больших изображений")
    
    # Загрузка модели
    with st.spinner("Инициализация модели..."):
        model = download_and_load_model()
    
    if model is None:
        st.error("Не удалось загрузить модель")
        return
    
    # Основной интерфейс
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Загрузите фото")
        
        uploaded = st.file_uploader(
            "Выберите изображение",
            type=['jpg', 'png', 'jpeg'],
            help="Рекомендуется до 512x512 для быстрой обработки"
        )
        
        if uploaded:
            input_img = Image.open(uploaded).convert('RGB')
            
            # Предпросмотр
            max_preview = 512
            if max(input_img.size) > max_preview:
                ratio = max_preview / max(input_img.size)
                preview_size = (int(input_img.size[0] * ratio), 
                              int(input_img.size[1] * ratio))
                preview_img = input_img.resize(preview_size, Image.Resampling.LANCZOS)
            else:
                preview_img = input_img
            
            st.image(preview_img, caption=f"Оригинал: {input_img.size}", width=300)
            
            # Информация
            st.info(f"""
            **Детали:**
            - Размер: {input_img.size[0]} × {input_img.size[1]}
            - Выход: {input_img.size[0]*4} × {input_img.size[1]*4}
            - Память: ~{(input_img.size[0]*input_img.size[1]*3*4)/1024/1024:.1f} MB
            """)
    
    with col2:
        st.header("✨ Результат")
        
        if uploaded and 'input_img' in locals():
            if st.button("🚀 Улучшить качество 4x", type="primary", use_container_width=True):
                with st.spinner("Обработка RealESRGAN..."):
                    start_time = time.time()
                    
                    try:
                        if use_tiling and (input_img.size[0] > 256 or input_img.size[1] > 256):
                            # Используем tiling для больших изображений
                            enhanced = process_with_tiling(model, input_img, 
                                                         tile_size=tile_size, 
                                                         overlap=32)
                        else:
                            # Прямая обработка для маленьких
                            img_np = np.array(input_img).astype(np.float32) / 255.0
                            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
                            
                            with torch.no_grad():
                                output_tensor = model(img_tensor)
                            
                            output_np = output_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
                            output_np = np.clip(output_np, 0, 1) * 255
                            enhanced = Image.fromarray(output_np.astype(np.uint8))
                        
                        elapsed = time.time() - start_time
                        
                        # Предпросмотр результата
                        if max(enhanced.size) > max_preview:
                            ratio = max_preview / max(enhanced.size)
                            preview_size = (int(enhanced.size[0] * ratio), 
                                          int(enhanced.size[1] * ratio))
                            preview_enhanced = enhanced.resize(preview_size, Image.Resampling.LANCZOS)
                        else:
                            preview_enhanced = enhanced
                        
                        st.image(preview_enhanced, caption=f"Улучшено: {enhanced.size}", width=300)
                        
                        # Скачивание
                        buf = io.BytesIO()
                        enhanced.save(buf, format="PNG", optimize=True)
                        
                        st.download_button(
                            "📥 Скачать PNG",
                            buf.getvalue(),
                            file_name=f"real_esrgan_{uploaded.name}",
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
                        
                    except torch.cuda.OutOfMemoryError:
                        st.error("⚠️ Недостаточно памяти GPU! Увеличьте размер тайла или уменьшите изображение.")
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            st.error("⚠️ Недостаточно памяти! Попробуйте:")
                            st.markdown("""
                            1. Уменьшить размер тайла
                            2. Загрузить меньшее изображение
                            3. Включить tiling
                            """)
                        else:
                            st.error(f"Ошибка: {e}")
        
        else:
            st.info("Загрузите изображение для обработки")
    
    # Техническая информация
    with st.expander("🔧 Технические детали"):
        st.markdown(f"""
        ### RealESRGAN_x4plus
        
        **Архитектура:**
        - Блоки: 23 RRDB
        - Каналы: 64 (num_feat)
        - Рост каналов: 32 (num_grow_ch)
        - Масштаб: 4x
        
        **Tiling система:**
        - Размер тайла: {tile_size}px
        - Overlap: 32px
        - Обработка: по частям
        
        **Память:**
        - Веса модели: 1.07GB
        - Память на обработку: ~500MB-1GB
        - Рекомендация: изображения до 512x512
        """)

if __name__ == "__main__":
    main()
