import streamlit as st
import os
from pathlib import Path
import sys
from PIL import Image
import io
import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import time
import requests
from tqdm import tqdm

# Настройка страницы
st.set_page_config(
    page_title="AI Photo Enhancer Pro",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Конфигурация
MODELS_DIR = Path("models")
PORTRAIT_MODEL_SIZE = (128, 128)
PORTRAIT_OUTPUT_SCALE = 2
MAX_FILE_SIZE_MB = 20  # MB

# Создаем директорию для моделей
MODELS_DIR.mkdir(exist_ok=True)

# CSS стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .model-status {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .model-loaded {
        background: #d4edda;
        border-left: 4px solid #28a745;
    }
    .model-missing {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<h1 class="main-header">✨ AI Photo Enhancer Pro</h1>', unsafe_allow_html=True)

# ================== КЛАСС ДЛЯ МОДЕЛЕЙ ==================
class EnhancementModels:
    """Класс для загрузки и управления моделями улучшения"""
    
    def __init__(self):
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.session_state['device'] = self.device
    
    def load_landscape_model(self):
        """Загрузка модели Real-ESRGAN для ландшафтов"""
        try:
            model_path = MODELS_DIR / 'RealESRGAN_x4plus.pth'
            
            if not model_path.exists():
                return None
            
            # Проверяем размер файла
            file_size = os.path.getsize(model_path) / (1024 * 1024)
            if file_size < 60:
                return None
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)
            
            upsampler = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=self.device.type != 'cpu',
                device=self.device
            )
            
            self.models['landscape'] = upsampler
            return upsampler
            
        except Exception as e:
            return None
    
    def load_portrait_model(self):
        """Загрузка кастомной модели для портретов"""
        try:
            model_path = MODELS_DIR / 'enhanced_epoch_28_ratio_1.23.pth'
            
            if not model_path.exists():
                return None
            
            # Определяем архитектуру модели
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
            
            checkpoint = torch.load(str(model_path), map_location=self.device)
            model = StrongGenerator().to(self.device)
            model.load_state_dict(checkpoint['generator'])
            model.eval()
            model.input_size = PORTRAIT_MODEL_SIZE
            
            self.models['portrait'] = model
            return model
            
        except Exception as e:
            return None

# ================== ФУНКЦИИ ИНФЕРЕНСА ==================
def prepare_for_portrait_model(img_array: np.ndarray, target_size: tuple = (128, 128)) -> np.ndarray:
    """Подготавливает изображение для портретной модели"""
    h, w = img_array.shape[:2]
    
    if h > target_size[0] or w > target_size[1]:
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = min(target_size[0] / h, target_size[1] / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    if new_h < target_size[0] or new_w < target_size[1]:
        pad_h = target_size[0] - new_h
        pad_w = target_size[1] - new_w
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        resized = cv2.copyMakeBorder(resized,
                                    pad_top, pad_bottom,
                                    pad_left, pad_right,
                                    cv2.BORDER_REFLECT)
    
    return resized

def run_portrait_model_inference(model, img_array: np.ndarray) -> np.ndarray:
    """Запускает инференс портретной модели"""
    try:
        original_h, original_w = img_array.shape[:2]
        img_prepared = prepare_for_portrait_model(img_array, PORTRAIT_MODEL_SIZE)
        
        img_tensor = torch.from_numpy(img_prepared).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(model.device)
        
        with torch.no_grad():
            output_tensor = model(img_tensor)
        
        output_tensor = output_tensor.squeeze(0).permute(1, 2, 0)
        output = (output_tensor.cpu().numpy() * 255.0).astype(np.uint8)
        
        h, w = output.shape[:2]
        target_h, target_w = PORTRAIT_MODEL_SIZE
        pad_h = h - target_h
        pad_w = w - target_w
        
        if pad_h > 0 or pad_w > 0:
            start_h = pad_h // 2 if pad_h > 0 else 0
            start_w = pad_w // 2 if pad_w > 0 else 0
            end_h = h - (pad_h - start_h) if pad_h > 0 else h
            end_w = w - (pad_w - start_w) if pad_w > 0 else w
            output = output[start_h:end_h, start_w:end_w]
        
        if PORTRAIT_OUTPUT_SCALE > 1:
            scaled_h = output.shape[0] * PORTRAIT_OUTPUT_SCALE
            scaled_w = output.shape[1] * PORTRAIT_OUTPUT_SCALE
            output = cv2.resize(output, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)
        
        result_h, result_w = output.shape[:2]
        if result_h < original_h and result_w < original_w:
            scale_h = original_h / result_h
            scale_w = original_w / result_w
            scale = max(scale_h, scale_w)
            if scale > 1:
                new_h = int(result_h * scale)
                new_w = int(result_w * scale)
                output = cv2.resize(output, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        return output
        
    except Exception as e:
        raise Exception(f"Ошибка в портретной модели: {e}")

def run_landscape_model_inference(model, img_array: np.ndarray) -> np.ndarray:
    """Запускает инференс ландшафтной модели (Real-ESRGAN)"""
    try:
        output, _ = model.enhance(img_array, outscale=4)
        return output
    except Exception as e:
        raise Exception(f"Ошибка в ландшафтной модели: {e}")

# ================== ОСНОВНАЯ ФУНКЦИЯ УЛУЧШЕНИЯ ==================
def enhance_with_model(image: Image.Image, model_type: str, models_manager) -> Image.Image:
    """Улучшает изображение с помощью выбранной модели"""
    
    # Конвертируем PIL в numpy
    img_array = np.array(image)
    
    if model_type == 'portrait':
        model = models_manager.models.get('portrait')
        if model is None:
            raise Exception("Модель для портретов не загружена")
        
        # Запускаем инференс
        output_array = run_portrait_model_inference(model, img_array)
        return Image.fromarray(output_array)
    
    elif model_type == 'landscape':
        model = models_manager.models.get('landscape')
        if model is None:
            raise Exception("Модель для ландшафтов не загружена")
        
        # Запускаем инференс
        output_array = run_landscape_model_inference(model, img_array)
        return Image.fromarray(output_array)
    
    else:  # auto
        # Автоматическое определение
        height, width = img_array.shape[:2]
        aspect_ratio = width / height
        
        if aspect_ratio > 1.3:  # Широкое изображение = ландшафт
            model = models_manager.models.get('landscape')
            if model is None:
                raise Exception("Модель для ландшафтов не загружена")
            output_array = run_landscape_model_inference(model, img_array)
        else:  # Вертикальное или квадратное = портрет
            model = models_manager.models.get('portrait')
            if model is None:
                raise Exception("Модель для портретов не загружена")
            output_array = run_portrait_model_inference(model, img_array)
        
        return Image.fromarray(output_array)

# ================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==================
def save_image(image, format='PNG'):
    """Сохраняет изображение в bytes"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

@st.cache_resource
def init_models():
    """Инициализация и загрузка моделей"""
    models_manager = EnhancementModels()
    
    # Загружаем модели
    with st.spinner("🔄 Загружаю модели..."):
        landscape_loaded = models_manager.load_landscape_model()
        portrait_loaded = models_manager.load_portrait_model()
    
    loaded_models = []
    if landscape_loaded:
        loaded_models.append('landscape')
    if portrait_loaded:
        loaded_models.append('portrait')
    
    return models_manager, loaded_models

# ================== ИНТЕРФЕЙС ==================
# Сайдбар
with st.sidebar:
    st.title("⚙️ Настройки")
    
    st.subheader("🧠 Модели улучшения")
    
    # Проверяем наличие файлов моделей
    realesrgan_exists = (MODELS_DIR / 'RealESRGAN_x4plus.pth').exists()
    portrait_exists = (MODELS_DIR / 'enhanced_epoch_28_ratio_1.23.pth').exists()
    
    if realesrgan_exists:
        size = os.path.getsize(MODELS_DIR / 'RealESRGAN_x4plus.pth') / (1024 * 1024)
        st.markdown(f'<div class="model-status model-loaded">✅ Real-ESRGAN: {size:.1f}MB</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="model-status model-missing">❌ Real-ESRGAN: отсутствует</div>', unsafe_allow_html=True)
    
    if portrait_exists:
        size = os.path.getsize(MODELS_DIR / 'enhanced_epoch_28_ratio_1.23.pth') / (1024 * 1024)
        st.markdown(f'<div class="model-status model-loaded">✅ Модель портретов: {size:.1f}MB</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="model-status model-missing">❌ Модель портретов: отсутствует</div>', unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("Тип улучшения")
    enhancement_type = st.radio(
        "Выберите режим:",
        ["🤖 Автоопределение", "🎭 Только портрет", "🌄 Только пейзаж"],
        index=0,
        help="Автоопределение выберет модель автоматически по формату фото"
    )
    
    # Преобразуем в тип для модели
    if "портрет" in enhancement_type.lower():
        model_type = 'portrait'
    elif "пейзаж" in enhancement_type.lower():
        model_type = 'landscape'
    else:
        model_type = 'auto'
    
    st.divider()
    
    st.subheader("ℹ️ Информация")
    device_name = "GPU 🚀" if torch.cuda.is_available() else "CPU ⚡"
    st.write(f"Устройство: {device_name}")
    st.write(f"Макс. размер файла: {MAX_FILE_SIZE_MB}MB")

# Основной интерфейс
st.markdown('<p class="sub-header">Улучшение фотографий с помощью нейросетевых моделей</p>', unsafe_allow_html=True)

# Загрузка фото
uploaded_file = st.file_uploader(
    "📤 Загрузите фотографию для улучшения",
    type=['jpg', 'jpeg', 'png'],
    help=f"Максимальный размер: {MAX_FILE_SIZE_MB}MB"
)

if uploaded_file is not None:
    # Проверка размера
    file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
    if file_size > MAX_FILE_SIZE_MB:
        st.error(f"❌ Файл слишком большой ({file_size:.1f}MB)")
        st.stop()
    
    # Загрузка изображения
    image = Image.open(uploaded_file).convert('RGB')
    
    # Показываем оригинал
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Оригинал")
        st.image(image, use_column_width=True)
        st.caption(f"Размер: {image.width}×{image.height} пикселей")
    
    # Кнопка улучшения
    if st.button("✨ УЛУЧШИТЬ ФОТО", type="primary", use_container_width=True):
        # Инициализация моделей
        if 'models_manager' not in st.session_state:
            models_manager, loaded_models = init_models()
            st.session_state.models_manager = models_manager
            st.session_state.loaded_models = loaded_models
            
            if not loaded_models:
                st.error("❌ Не удалось загрузить ни одну модель!")
                st.stop()
            else:
                st.success(f"✅ Загружено моделей: {len(loaded_models)}")
        
        models_manager = st.session_state.models_manager
        
        with st.spinner("🧠 Нейросеть обрабатывает изображение..."):
            try:
                # Определяем какую модель использовать
                if model_type == 'auto':
                    # Автоматическое определение
                    if image.width / image.height > 1.3:  # Широкое
                        actual_model = 'landscape'
                        model_name = "Real-ESRGAN (пейзажи)"
                    else:  # Вертикальное или квадратное
                        actual_model = 'portrait'
                        model_name = "Кастомная CNN (портреты)"
                else:
                    actual_model = model_type
                    model_name = "Real-ESRGAN" if model_type == 'landscape' else "Кастомная CNN"
                
                # Проверяем, что модель загружена
                if actual_model not in models_manager.models:
                    available = list(models_manager.models.keys())
                    st.error(f"❌ Модель '{actual_model}' не загружена. Доступны: {available}")
                    st.stop()
                
                # Прогресс бар
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Улучшаем фото с помощью модели
                enhanced_image = enhance_with_model(image, actual_model, models_manager)
                
                progress_bar.empty()
                
                # Показываем результат
                with col2:
                    st.subheader("🚀 Улучшенное")
                    st.image(enhanced_image, use_column_width=True)
                    st.caption(f"Новый размер: {enhanced_image.width}×{enhanced_image.height} пикселей")
                    
                    # Информация об улучшении
                    with st.expander("📊 Детали обработки", expanded=True):
                        st.write(f"**Использованная модель:** {model_name}")
                        st.write(f"**Тип улучшения:** {enhancement_type}")
                        st.write(f"**Исходный размер:** {image.width}×{image.height}")
                        st.write(f"**Финальный размер:** {enhanced_image.width}×{enhanced_image.height}")
                        
                        if actual_model == 'portrait':
                            st.write(f"**Архитектура:** Residual CNN")
                            st.write(f"**Вход модели:** {PORTRAIT_MODEL_SIZE[0]}×{PORTRAIT_MODEL_SIZE[1]}")
                            st.write(f"**Увеличение:** ×{PORTRAIT_OUTPUT_SCALE}")
                        else:
                            st.write(f"**Архитектура:** Real-ESRGAN")
                            st.write(f"**Увеличение:** ×4")
                    
                    # Кнопка скачивания
                    enhanced_bytes = save_image(enhanced_image, 'PNG')
                    
                    st.download_button(
                        label="💾 СКАЧАТЬ УЛУЧШЕННОЕ ФОТО",
                        data=enhanced_bytes,
                        file_name="enhanced_photo.png",
                        mime="image/png",
                        type="primary",
                        use_container_width=True
                    )
                
                st.success(f"✅ Фото успешно улучшено с помощью {model_name}!")
                st.balloons()
                
                # Сравнение
                st.divider()
                st.subheader("🔄 Сравнение результатов")
                
                compare_col1, compare_col2 = st.columns(2)
                with compare_col1:
                    st.image(image, caption="ДО улучшения", use_column_width=True)
                with compare_col2:
                    st.image(enhanced_image, caption="ПОСЛЕ улучшения", use_column_width=True)
                
            except Exception as e:
                st.error(f"❌ Ошибка при улучшении: {str(e)}")
                st.info("Попробуйте другую фотографию или другой тип улучшения")

else:
    # Домашняя страница
    st.info("""
    ### 🎯 Возможности нейросетевых моделей:
    
    **🎭 Кастомная модель для портретов:**
    - Улучшение деталей лица и кожи
    - Сохранение естественных цветов
    - Оптимальная обработка для вертикальных фото
    
    **🌄 Real-ESRGAN для пейзажей:**
    - Увеличение разрешения в 4 раза
    - Улучшение текстур и деталей
    - Идеально для широкоформатных фото
    
    ### 📋 Как использовать:
    1. **Загрузите** фотографию (кнопка выше)
    2. **Выберите** тип улучшения в сайдбаре
    3. **Нажмите** "Улучшить фото"
    4. **Скачайте** результат
    
    ### 💡 Советы:
    - Используйте **"Автоопределение"** для автоматического выбора модели
    - Для **портретов** выбирайте соответствующий режим
    - Для **пейзажей** используйте Real-ESRGAN
    - Модели работают только если файлы загружены в папку `models/`
    """)

# Футер
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>© 2024 AI Photo Enhancer Pro | Используются только нейросетевые модели</p>
    <p>Real-ESRGAN + Кастомная CNN | Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)
