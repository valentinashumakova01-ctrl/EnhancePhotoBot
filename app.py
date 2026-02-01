import streamlit as st
import os
from pathlib import Path
from PIL import Image
import io
import cv2
import numpy as np
import torch
import torch.nn as nn
import time

# Настройка страницы
st.set_page_config(
    page_title="AI Photo Enhancer",
    page_icon="✨",
    layout="wide"
)

# Конфигурация
MODELS_DIR = Path("models")
MAX_FILE_SIZE_MB = 20

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">✨ AI Photo Enhancer</h1>', unsafe_allow_html=True)

# ================== ПРОСТЫЕ МОДЕЛИ ==================

# Простая архитектура для портретов (inline)
class SimplePortraitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )
    
    def forward(self, x):
        return x + 0.2 * self.net(x)

# ================== ПРОСТАЯ ЗАГРУЗКА МОДЕЛЕЙ ==================

def load_models_simple():
    """Простая загрузка моделей"""
    models = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Проверяем Real-ESRGAN
    realesrgan_path = MODELS_DIR / 'RealESRGAN_x4plus.pth'
    if realesrgan_path.exists():
        try:
            # Пробуем загрузить Real-ESRGAN если есть зависимости
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=23, num_grow_ch=32, scale=4)
                
                upsampler = RealESRGANer(
                    scale=4,
                    model_path=str(realesrgan_path),
                    model=model,
                    tile=400,
                    tile_pad=10,
                    pre_pad=0,
                    half=device.type != 'cpu',
                    device=device
                )
                models['landscape'] = upsampler
                st.success("✅ Real-ESRGAN загружен")
            except ImportError:
                st.warning("⚠️ Real-ESRGAN зависимости не установлены")
        except Exception as e:
            st.error(f"❌ Ошибка загрузки Real-ESRGAN: {e}")
    else:
        st.warning("⚠️ Real-ESRGAN файл не найден")
    
    # 2. Простая модель для портретов
    portrait_path = MODELS_DIR / 'enhanced_epoch_28_ratio_1.23.pth'
    if portrait_path.exists():
        try:
            # Создаем простую модель
            model = SimplePortraitModel()
            
            # Пробуем загрузить веса
            checkpoint = torch.load(str(portrait_path), map_location=device)
            
            # Пробуем разные ключи в checkpoint
            if 'generator' in checkpoint:
                model.load_state_dict(checkpoint['generator'])
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Пробуем загрузить напрямую
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            models['portrait'] = model
            st.success("✅ Модель портретов загружена")
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки модели портретов: {e}")
            st.write(f"Детали ошибки: {str(e)}")
    else:
        st.warning("⚠️ Модель портретов не найдена")
    
    return models, device

# ================== ПРОСТОЙ ИНТЕРФЕЙС ==================

# Сайдбар
with st.sidebar:
    st.title("⚙️ Настройки")
    
    st.subheader("🧠 Модели")
    
    # Кнопка загрузки моделей
    if st.button("🔄 Загрузить модели", use_container_width=True):
        if 'models' not in st.session_state:
            models, device = load_models_simple()
            st.session_state.models = models
            st.session_state.device = device
            
            if models:
                st.success(f"✅ Загружено моделей: {len(models)}")
                for name in models.keys():
                    st.write(f"• {name}")
            else:
                st.error("❌ Не удалось загрузить модели")
    
    # Показываем статус
    if 'models' in st.session_state:
        st.success(f"✅ Модели загружены: {len(st.session_state.models)}")
    else:
        st.warning("⚠️ Модели не загружены")
    
    st.divider()
    
    st.subheader("Тип улучшения")
    model_type = st.radio(
        "Выберите модель:",
        ["🎭 Портрет", "🌄 Пейзаж"],
        index=0
    )
    
    model_type = 'portrait' if "Портрет" in model_type else 'landscape'

# Основной интерфейс
uploaded_file = st.file_uploader(
    "📤 Загрузите фото",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file:
    # Проверка размера
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size > MAX_FILE_SIZE_MB:
        st.error(f"❌ Файл слишком большой ({file_size:.1f}MB)")
        st.stop()
    
    # Загрузка изображения
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Оригинал")
        st.image(image, use_column_width=True)
        st.caption(f"Размер: {image.width}×{image.height}")
    
    # Кнопка улучшения
    if st.button("✨ Улучшить фото", type="primary", use_container_width=True):
        # Проверяем загружены ли модели
        if 'models' not in st.session_state or not st.session_state.models:
            st.error("❌ Сначала загрузите модели (кнопка в сайдбаре)")
            st.stop()
        
        models = st.session_state.models
        device = st.session_state.device
        
        # Проверяем нужную модель
        if model_type not in models:
            st.error(f"❌ Модель '{model_type}' не загружена")
            st.write(f"Доступные модели: {list(models.keys())}")
            st.stop()
        
        model = models[model_type]
        
        with st.spinner("Обработка..."):
            # Конвертируем в numpy
            img_array = np.array(image)
            
            try:
                if model_type == 'landscape' and hasattr(model, 'enhance'):
                    # Real-ESRGAN
                    output, _ = model.enhance(img_array, outscale=4)
                    enhanced_image = Image.fromarray(output)
                    method = "Real-ESRGAN"
                    
                elif model_type == 'portrait':
                    # Простая портретная модель
                    # Подготавливаем изображение
                    h, w = img_array.shape[:2]
                    target_size = 256  # Фиксированный размер для простоты
                    
                    # Ресайзим
                    scale = min(target_size / h, target_size / w)
                    new_h = int(h * scale)
                    new_w = int(w * scale)
                    resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    
                    # Конвертируем в тензор
                    img_tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
                    img_tensor = img_tensor.unsqueeze(0).to(device)
                    
                    # Инференс
                    with torch.no_grad():
                        output_tensor = model(img_tensor)
                    
                    # Обратно в numpy
                    output_tensor = output_tensor.squeeze(0).permute(1, 2, 0)
                    output = (output_tensor.cpu().numpy() * 255.0).astype(np.uint8)
                    
                    # Возвращаем к оригинальному размеру
                    output = cv2.resize(output, (image.width, image.height), interpolation=cv2.INTER_CUBIC)
                    enhanced_image = Image.fromarray(output)
                    method = "Кастомная CNN"
                    
                else:
                    # Fallback - простое увеличение
                    new_size = (image.width * 2, image.height * 2)
                    enhanced_image = image.resize(new_size, Image.Resampling.LANCZOS)
                    method = "Простое увеличение"
                
                # Показываем результат
                with col2:
                    st.subheader("🚀 Улучшенное")
                    st.image(enhanced_image, use_column_width=True)
                    st.caption(f"Размер: {enhanced_image.width}×{enhanced_image.height}")
                    
                    # Кнопка скачивания
                    buf = io.BytesIO()
                    enhanced_image.save(buf, format="PNG")
                    
                    st.download_button(
                        "💾 Скачать",
                        buf.getvalue(),
                        "enhanced.png",
                        "image/png",
                        use_container_width=True
                    )
                
                st.success(f"✅ Фото улучшено с помощью {method}!")
                
            except Exception as e:
                st.error(f"❌ Ошибка обработки: {str(e)}")

else:
    # Инструкция
    st.info("""
    ### 📋 Инструкция:
    1. **Нажмите "Загрузить модели"** в сайдбаре
    2. **Загрузите фото** 
    3. **Выберите тип модели**
    4. **Нажмите "Улучшить фото"**
    
    ### 🧠 Модели:
    - **🎭 Портрет:** Ваша кастомная модель
    - **🌄 Пейзаж:** Real-ESRGAN
    
    ### 📁 Требуемые файлы в папке `models/`:
    1. `RealESRGAN_x4plus.pth`
    2. `enhanced_epoch_28_ratio_1.23.pth`
    """)

# Футер
st.divider()
st.caption("© 2024 AI Photo Enhancer | Простая версия")
