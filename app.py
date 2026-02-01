import streamlit as st
import os
from pathlib import Path
from PIL import Image
import io
import numpy as np
import time
import requests

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
    .status-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
    }
    .enhance-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border-left: 5px solid #667eea;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    .image-container {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .model-download {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    .download-progress {
        background: #e9ecef;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Функция скачивания моделей
def download_model_with_progress(url, output_path):
    """Скачивает модель с прогресс-баром"""
    try:
        st.info(f"📥 Начинаю скачивание модели...")
        
        # Создаем контейнер для прогресса
        progress_container = st.empty()
        status_container = st.empty()
        
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size == 0:
            status_container.warning("Не удалось определить размер файла. Скачиваю...")
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        else:
            # Создаем прогресс-бар
            progress_bar = progress_container.progress(0)
            status_text = status_container.empty()
            
            downloaded = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        downloaded += len(chunk)
                        f.write(chunk)
                        
                        # Обновляем прогресс
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
                        
                        # Обновляем текст статуса
                        downloaded_mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        status_text.text(f"📥 Скачано: {downloaded_mb:.1f}MB / {total_mb:.1f}MB ({progress*100:.1f}%)")
            
            # Очищаем контейнеры
            progress_container.empty()
            status_container.empty()
        
        # Проверяем размер файла
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        if file_size < 10:  # Если файл слишком маленький, вероятно ошибка
            os.remove(output_path)
            raise Exception(f"Файл слишком маленький ({file_size:.1f}MB). Возможно ошибка скачивания.")
        
        st.success(f"✅ Модель успешно скачана: {output_path.name} ({file_size:.1f}MB)")
        return True
        
    except Exception as e:
        st.error(f"❌ Ошибка при скачивании: {str(e)}")
        # Пробуем удалить битый файл если он существует
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return False

def check_and_download_models():
    """Проверяет наличие моделей и предлагает скачать если их нет"""
    
    models_to_download = []
    
    # Проверяем Real-ESRGAN
    realesrgan_path = MODELS_DIR / 'RealESRGAN_x4plus.pth'
    if not realesrgan_path.exists():
        models_to_download.append({
            'name': 'Real-ESRGAN_x4plus.pth',
            'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
            'path': realesrgan_path,
            'description': 'Модель для улучшения пейзажей (x4 увеличение)'
        })
    
    # Проверяем кастомную модель портретов
    portrait_path = MODELS_DIR / 'enhanced_epoch_28_ratio_1.23.pth'
    if not portrait_path.exists():
        models_to_download.append({
            'name': 'enhanced_epoch_28_ratio_1.23.pth',
            'url': None,  # Укажите вашу ссылку если есть
            'path': portrait_path,
            'description': 'Кастомная модель для улучшения портретов'
        })
    
    return models_to_download

# Класс для моделей
class EnhancementModels:
    """Класс для загрузки и управления моделями улучшения"""

    def __init__(self):
        self.models = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.session_state['device'] = self.device

    def load_landscape_model(self):
        """Загрузка модели для улучшения ландшафтов"""
        try:
            model_path = MODELS_DIR / 'RealESRGAN_x4plus.pth'
            
            if not model_path.exists():
                st.warning("Real-ESRGAN модель не найдена")
                return None
        
        # Вместо Real-ESRGAN используем альтернативу
        st.warning("Real-ESRGAN временно недоступен. Используется альтернативное улучшение.")
        return None
        
    except Exception as e:
        st.error(f"Ошибка загрузки модели ландшафтов: {e}")
        return None
            
            # Проверяем размер файла
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            if file_size < 60:  # Real-ESRGAN должен быть около 64MB
                st.warning(f"⚠️ Файл модели слишком маленький ({file_size:.1f}MB). Возможно он поврежден.")
                if st.button("🔄 Перескачать модель", key="redownload_realesrgan"):
                    url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
                    if download_model_with_progress(url, model_path):
                        st.rerun()
                return None

            with st.spinner(f"🔄 Загружаю Real-ESRGAN ({file_size:.1f}MB)..."):
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
                st.success(f"✅ Real-ESRGAN загружен (x4 увеличение)")
                return upsampler

        except Exception as e:
            st.error(f"❌ Ошибка загрузки модели ландшафтов: {e}")
            return None

    def load_portrait_model(self):
        """Загрузка модели для улучшения портретов"""
        try:
            model_path = MODELS_DIR / 'enhanced_epoch_28_ratio_1.23.pth'
            
            if not model_path.exists():
                st.warning("⚠️ Кастомная модель портретов не найдена")
                st.info("Будет использовано базовое улучшение для портретов")
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

            with st.spinner("🔄 Загружаю модель для портретов..."):
                checkpoint = torch.load(str(model_path), map_location=self.device)

                model = StrongGenerator().to(self.device)
                model.load_state_dict(checkpoint['generator'])
                model.eval()
                model.input_size = PORTRAIT_MODEL_SIZE

                self.models['portrait'] = model
                st.success(f"✅ Модель для портретов загружена")
                st.info(f"Вход модели: {PORTRAIT_MODEL_SIZE[0]}x{PORTRAIT_MODEL_SIZE[1]}, Увеличение: x{PORTRAIT_OUTPUT_SCALE}")
                return model

        except Exception as e:
            st.error(f"❌ Ошибка загрузки модели портретов: {e}")
            return None

# Инициализация моделей
@st.cache_resource
def init_models():
    models_manager = EnhancementModels()
    
    # Проверяем наличие моделей
    missing_models = check_and_download_models()
    if missing_models:
        st.warning(f"⚠️ Отсутствует {len(missing_models)} моделей")
        return models_manager, False
    
    # Загружаем модели
    with st.spinner("🔄 Загружаю модели..."):
        models_manager.load_landscape_model()
        models_manager.load_portrait_model()
    
    loaded = list(models_manager.models.keys())
    
    if loaded:
        st.success(f"✅ Загружено {len(loaded)} моделей")
        return models_manager, True
    else:
        st.warning("⚠️ Не удалось загрузить ни одной модели")
        return models_manager, False

# Функции обработки изображений (остаются без изменений)
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

def enhance_portrait_model_inference(model, img_array: np.ndarray) -> np.ndarray:
    """Инференс для портретной модели"""
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
        st.error(f"Ошибка в портретной модели: {e}")
        raise

def enhance_image_basic(img_array: np.ndarray, scale: int = 2, sharpness: float = 1.3) -> Image.Image:
    """Базовое улучшение изображения (fallback)"""
    try:
        h, w = img_array.shape[:2]
        enhanced = cv2.resize(img_array, (w * scale, h * scale),
                             interpolation=cv2.INTER_CUBIC)

        if sharpness > 1.0:
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 3)
            enhanced = cv2.addWeighted(enhanced, sharpness, gaussian, 1 - sharpness, 0)

        lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)

        return Image.fromarray(enhanced)

    except Exception as e:
        st.error(f"Ошибка в базовом улучшении: {e}")
        h, w = img_array.shape[:2]
        enhanced = cv2.resize(img_array, (w * scale, h * scale),
                             interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(enhanced)

def enhance_image_advanced(image: Image.Image, models_manager, enhancement_type: str = 'auto') -> Image.Image:
    """Улучшение изображения"""
    try:
        img_array = np.array(image)
        
        # Всегда используем базовое улучшение
        st.info("Используется базовое улучшение (Real-ESRGAN временно недоступен)")
        
        if enhancement_type == 'landscape':
            return enhance_image_basic(img_array, scale=4, sharpness=1.5)
        elif enhancement_type == 'portrait':
            return enhance_image_basic(img_array, scale=2, sharpness=1.2)
        else:  # auto
            height, width = img_array.shape[:2]
            aspect_ratio = width / height
            
            if aspect_ratio > 1.3:  # Широкое - ландшафт
                return enhance_image_basic(img_array, scale=4, sharpness=1.5)
            else:  # Вертикальное - портрет
                return enhance_image_basic(img_array, scale=2, sharpness=1.2)
                
    except Exception as e:
        st.error(f"Ошибка улучшения: {e}")
        return enhance_image_basic(np.array(image))

def save_image(image, format='PNG'):
    """Сохраняет изображение в bytes"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()

# Сайдбар
with st.sidebar:
    st.title("⚙️ Настройки")
    
    # Проверка моделей
    st.subheader("🧠 Модели нейросетей")
    
    # Кнопка проверки моделей
    if st.button("🔍 Проверить модели", use_container_width=True):
        missing_models = check_and_download_models()
        if missing_models:
            st.error(f"❌ Отсутствует {len(missing_models)} моделей:")
            for model in missing_models:
                st.write(f"• {model['name']} - {model['description']}")
        else:
            st.success("✅ Все модели на месте!")
    
    # Кнопка скачивания Real-ESRGAN
    if st.button("📥 Скачать Real-ESRGAN", use_container_width=True):
        url = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth'
        output_path = MODELS_DIR / 'RealESRGAN_x4plus.pth'
        
        if download_model_with_progress(url, output_path):
            st.success("✅ Модель скачана успешно!")
            st.rerun()
    
    st.divider()
    
    st.subheader("Тип улучшения")
    enhancement_type = st.radio(
        "Выберите режим:",
        ["🤖 Автоопределение", "🎭 Портрет", "🌄 Пейзаж"],
        index=0
    )
    
    enhancement_type = enhancement_type.split(" ")[1].lower()
    
    st.divider()
    
    # Информация о моделях
    st.subheader("📊 Информация")
    
    # Проверяем наличие моделей
    realesrgan_exists = (MODELS_DIR / 'RealESRGAN_x4plus.pth').exists()
    portrait_exists = (MODELS_DIR / 'enhanced_epoch_28_ratio_1.23.pth').exists()
    
    if realesrgan_exists:
        file_size = os.path.getsize(MODELS_DIR / 'RealESRGAN_x4plus.pth') / (1024 * 1024)
        st.success(f"✅ Real-ESRGAN: {file_size:.1f}MB")
    else:
        st.error("❌ Real-ESRGAN: отсутствует")
    
    if portrait_exists:
        st.success("✅ Модель портретов: есть")
    else:
        st.warning("⚠️ Модель портретов: отсутствует")
    
    st.divider()
    
    # Информация о системе
    st.subheader("💻 Система")
    device_name = "GPU 🚀" if torch.cuda.is_available() else "CPU ⚡"
    st.write(f"Устройство: {device_name}")
    
    st.write(f"Макс. размер файла: {MAX_FILE_SIZE_MB}MB")

# Заголовок
st.markdown('<h1 class="main-header">✨ AI Photo Enhancer Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Улучшение качества фотографий с помощью нейросетей Real-ESRGAN и кастомных моделей</p>', unsafe_allow_html=True)

# Проверяем наличие необходимых моделей
missing_models = check_and_download_models()
if missing_models:
    st.warning("""
    ⚠️ **Внимание! Не все модели найдены.**
    
    Для полноценной работы приложения необходимы:
    1. **Real-ESRGAN_x4plus.pth** - для улучшения пейзажей
    2. **enhanced_epoch_28_ratio_1.23.pth** - для улучшения портретов
    
    **Что делать:**
    1. Нажмите кнопку "📥 Скачать Real-ESRGAN" в сайдбаре
    2. Поместите вашу модель портретов в папку `models/`
    3. Перезагрузите страницу
    """)
    
    with st.expander("📋 Инструкция по скачиванию моделей", expanded=True):
        st.markdown("""
        ### Для Windows:
        ```bash
        # Создайте папку models
        mkdir models
        
        # Скачайте Real-ESRGAN
        curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -o models/RealESRGAN_x4plus.pth
        ```
        
        ### Для Mac/Linux:
        ```bash
        mkdir -p models
        wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -O models/RealESRGAN_x4plus.pth
        ```
        
        ### Или в Python:
        ```python
        import requests
        import os
        
        os.makedirs("models", exist_ok=True)
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        
        response = requests.get(url, stream=True)
        with open("models/RealESRGAN_x4plus.pth", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        ```
        """)

# Основной интерфейс
tab1, tab2, tab3 = st.tabs(["🖼️ Улучшение фото", "📊 Сравнение", "ℹ️ О сервисе"])

with tab1:
    # Загрузка фото
    uploaded_file = st.file_uploader(
        "📤 Загрузите фотографию для улучшения",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        help=f"Максимальный размер: {MAX_FILE_SIZE_MB}MB"
    )
    
    if uploaded_file is not None:
        # Проверка размера
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
        if file_size > MAX_FILE_SIZE_MB:
            st.error(f"❌ Файл слишком большой ({file_size:.1f}MB). Максимум: {MAX_FILE_SIZE_MB}MB")
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
                models_manager, models_loaded = init_models()
                st.session_state.models_manager = models_manager
                st.session_state.models_loaded = models_loaded
            else:
                models_manager = st.session_state.models_manager
                models_loaded = st.session_state.models_loaded
            
            if not models_loaded:
                st.warning("⚠️ Модели не загружены. Используется базовое улучшение.")
            
            with st.spinner("🧠 ИИ обрабатывает изображение..."):
                progress_bar = st.progress(0)
                
                # Имитация прогресса
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete + 1)
                
                # Улучшение фото
                enhanced_image = enhance_image_advanced(
                    image, 
                    models_manager, 
                    enhancement_type
                )
                
                progress_bar.empty()
                
                # Показываем результат
                with col2:
                    st.subheader("🚀 Улучшенное")
                    st.image(enhanced_image, use_column_width=True)
                    st.caption(f"Новый размер: {enhanced_image.width}×{enhanced_image.height} пикселей")
                    
                    # Информация об улучшении
                    with st.expander("📊 Детали улучшения"):
                        st.write(f"**Тип улучшения:** {enhancement_type}")
                        st.write(f"**Исходный размер:** {image.width}×{image.height}")
                        st.write(f"**Финальный размер:** {enhanced_image.width}×{enhanced_image.height}")
                        
                        if enhancement_type == 'portrait':
                            st.write(f"**Модель:** Кастомная CNN")
                            st.write(f"**Вход модели:** {PORTRAIT_MODEL_SIZE[0]}×{PORTRAIT_MODEL_SIZE[1]}")
                            st.write(f"**Увеличение выхода:** ×{PORTRAIT_OUTPUT_SCALE}")
                        elif enhancement_type == 'landscape':
                            st.write(f"**Модель:** Real-ESRGAN")
                            st.write(f"**Увеличение:** ×4")
                        else:
                            if models_manager.models:
                                if len(image.width / image.height) > 1.3:
                                    st.write(f"**Определено:** Пейзаж (Real-ESRGAN)")
                                else:
                                    st.write(f"**Определено:** Портрет (Кастомная модель)")
                    
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
                
                st.success("✅ Фото успешно улучшено!")
                st.balloons()

with tab2:
    st.header("🔄 Сравнение ДО и ПОСЛЕ")
    
    if uploaded_file is not None and 'enhanced_image' in locals():
        col_before, col_after = st.columns(2)
        
        with col_before:
            st.subheader("ДО улучшения")
            st.image(image, use_column_width=True)
        
        with col_after:
            st.subheader("ПОСЛЕ улучшения")
            st.image(enhanced_image, use_column_width=True)
        
        # Статистика
        st.divider()
        st.subheader("📈 Статистика улучшения")
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        with col_stat1:
            st.metric(
                "Разрешение",
                f"{image.width}×{image.height}",
                f"{enhanced_image.width}×{enhanced_image.height}"
            )
        
        with col_stat2:
            pixel_increase = ((enhanced_image.width * enhanced_image.height) / 
                            (image.width * image.height))
            st.metric(
                "Количество пикселей",
                f"{(image.width * image.height):,}",
                f"{(enhanced_image.width * enhanced_image.height):,}",
                delta=f"×{pixel_increase:.1f}"
            )
        
        with col_stat3:
            st.metric(
                "Качество",
                "Исходное",
                "Улучшенное",
                delta="Повышено"
            )

with tab3:
    st.header("ℹ️ О сервисе")
    
    st.markdown("""
    ### ✨ Возможности сервиса
    
    **🎯 Основные функции:**
    - Увеличение разрешения фотографий
    - Улучшение детализации и резкости
    - Автоматическое определение типа изображения
    - Поддержка различных форматов
    
    **🧠 Используемые технологии:**
    
    1. **Real-ESRGAN** для пейзажей:
       - Увеличение ×4 (до 4K)
       - Сохранение деталей
       - Улучшение текстур
    
    2. **Кастомная CNN-модель** для портретов:
       - Входной размер: 128×128 пикселей
       - Увеличение выхода: ×2
       - Сохранение качества лиц
    
    **📊 Особенности:**
    - Изображение **никогда не уменьшается**
    - Результат всегда сохраняется или увеличивается
    - Автоматический выбор оптимальной модели
    
    ### 🚀 Как использовать:
    1. Загрузите фотографию
    2. Выберите тип улучшения (или оставьте авто)
    3. Нажмите "Улучшить фото"
    4. Скачайте результат
    
    ### 💡 Советы:
    - Для портретов используйте режим "Портрет"
    - Для пейзажей используйте режим "Пейзаж"
    - "Автоопределение" подбирает модель автоматически
    - Используйте исходные фото хорошего качества
    """)
    
    st.divider()
    
    col_info1, col_info2, col_info3 = st.columns(3)
    
    with col_info1:
        st.markdown("""
        ### 🎭 Портреты
        - Улучшение деталей лица
        - Сохранение естественности
        - Улучшение кожи
        """)
    
    with col_info2:
        st.markdown("""
        ### 🌄 Пейзажи
        - Увеличение до 4K
        - Улучшение текстур
        - Цветокоррекция
        """)
    
    with col_info3:
        st.markdown("""
        ### 🏙️ Архитектура
        - Улучшение линий
        - Детализация
        - Резкость
        """)

# Футер
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>© 2024 AI Photo Enhancer Pro | Powered by Real-ESRGAN & Custom CNN Models</p>
    <p>Streamlit · PyTorch · OpenCV</p>
</div>
""", unsafe_allow_html=True)
