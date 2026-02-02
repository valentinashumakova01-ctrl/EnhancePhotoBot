import subprocess
import sys
import importlib
import streamlit as st
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Заголовок приложения
st.set_page_config(
    page_title="Улучшение качества пейзажных фото",
    page_icon="🌄",
    layout="wide"
)

st.title("🌄 Улучшение качества пейзажных фото")
st.markdown("Использует модель Real-ESRGAN для улучшения качества изображений в 4 раза")

# Функция для проверки и установки зависимостей
def install_required_packages():
    """Устанавливает необходимые пакеты"""
    
    required_packages = [
        'basicsr>=1.4.2',
        'facexlib>=0.3.0',
        'gfpgan>=1.3.8',
        'realesrgan>=0.3.0',
        'opencv-python-headless>=4.8.0',
        'pillow>=10.0.0',
        'numpy>=1.24.0',
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'scipy>=1.10.0',
        'lmdb>=1.4.1',
        'tqdm>=4.65.0',
        'yapf>=0.32.0',
        'tb-nightly>=2.14.0',
        'packaging>=21.3',
        'pyyaml>=6.0',
    ]
    
    # Проверяем основные зависимости
    packages_to_install = []
    
    for package in required_packages:
        package_name = package.split('>=')[0].split('==')[0]
        try:
            importlib.import_module(package_name.replace('-', '_'))
            st.sidebar.success(f"✅ {package_name}")
        except ImportError:
            packages_to_install.append(package)
            st.sidebar.error(f"❌ {package_name}")
    
    # Если есть пакеты для установки
    if packages_to_install:
        with st.spinner("Установка необходимых зависимостей..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, package in enumerate(packages_to_install):
                status_text.text(f"Установка {package}...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    progress_bar.progress((i + 1) / len(packages_to_install))
                    st.sidebar.success(f"✅ Установлен: {package.split('>=')[0]}")
                except subprocess.CalledProcessError as e:
                    st.sidebar.error(f"❌ Ошибка установки {package}: {e}")
            
            status_text.text("Зависимости установлены!")
            st.success("✅ Все зависимости успешно установлены!")
            
            # Перезагружаем страницу
            st.rerun()
    
    return True

# Создаем директории
@st.cache_resource
def setup_directories():
    """Создает необходимые директории"""
    models_dir = Path("models")
    uploads_dir = Path("uploads")
    results_dir = Path("results")
    
    models_dir.mkdir(exist_ok=True)
    uploads_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    return models_dir, uploads_dir, results_dir

# Основной интерфейс
def main():
    # Сайдбар с информацией
    st.sidebar.title("⚙️ Установка зависимостей")
    
    if st.sidebar.button("🔧 Проверить и установить зависимости", type="primary"):
        install_required_packages()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Требуемые пакеты:
    - basicsr
    - facexlib  
    - gfpgan
    - realesrgan
    - torch
    - opencv
    - и другие...
    """)
    
    # Создаем директории
    models_dir, uploads_dir, results_dir = setup_directories()
    
    # Импортируем необходимые библиотеки (после установки)
    try:
        import torch
        import numpy as np
        from PIL import Image
        import urllib.request
        import time
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        import cv2
        
        st.success("✅ Все библиотеки успешно импортированы!")
        
    except ImportError as e:
        st.warning("⚠️ Некоторые библиотеки не установлены. Нажмите кнопку 'Проверить и установить зависимости' в сайдбаре.")
        st.code(f"Ошибка импорта: {e}")
        return
    
    # Функция для загрузки модели
    @st.cache_resource
    def download_and_load_model():
        """Загружает и загружает модель Real-ESRGAN"""
        
        # URL модели
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        model_path = models_dir / "RealESRGAN_x4plus.pth"
        
        # Проверяем, есть ли уже модель
        if not model_path.exists():
            with st.spinner("Загрузка модели Real-ESRGAN (1.07 GB)... Это может занять несколько минут"):
                # Создаем индикатор прогресса
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def download_progress(count, block_size, total_size):
                    percent = min(int(count * block_size * 100 / total_size), 100)
                    progress_bar.progress(percent / 100)
                    status_text.text(f"Загрузка: {percent}%")
                
                try:
                    # Загружаем модель
                    urllib.request.urlretrieve(
                        model_url, 
                        model_path, 
                        reporthook=download_progress
                    )
                    st.success("✅ Модель успешно загружена!")
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                except Exception as e:
                    st.error(f"Ошибка при загрузке модели: {e}")
                    return None
        else:
            st.info("✅ Модель уже загружена")
        
        try:
            # Создаем модель
            model = RRDBNet(
                num_in_ch=3, 
                num_out_ch=3, 
                num_feat=64, 
                num_block=23, 
                num_grow_ch=32, 
                scale=4
            )
            
            # Определяем устройство (CPU/GPU)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            st.info(f"Используется устройство: {device}")
            
            # Создаем улучшатель
            upsampler = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                model=model,
                tile=400,
                tile_pad=10,
                pre_pad=0,
                half=False,
                device=device
            )
            
            return upsampler
        except Exception as e:
            st.error(f"Ошибка при создании модели: {e}")
            return None
    
    # Функция для улучшения изображения
    def enhance_image(input_image, upsampler):
        """Улучшает качество изображения"""
        try:
            # Конвертируем PIL Image в numpy array
            img = np.array(input_image)
            
            # Конвертируем RGB в BGR для OpenCV
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Улучшаем изображение
            output, _ = upsampler.enhance(img, outscale=4)
            
            # Конвертируем обратно в RGB
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            
            # Конвертируем обратно в PIL Image
            output_img = Image.fromarray(output)
            
            return output_img
        except Exception as e:
            st.error(f"Ошибка при улучшении изображения: {e}")
            st.exception(e)
            return None
    
    # Загружаем модель
    with st.spinner("Инициализация модели..."):
        upsampler = download_and_load_model()
    
    if upsampler is None:
        st.error("Не удалось загрузить модель. Попробуйте перезапустить приложение.")
        return
    
    # Интерфейс загрузки изображений
    st.header("📤 Загрузите изображение для улучшения")
    
    # Две колонки для интерфейса
    col1, col2 = st.columns(2)
    
    with col1:
        # Загрузка файла
        uploaded_file = st.file_uploader(
            "Выберите изображение пейзажа",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
            help="Поддерживаются форматы: JPG, PNG, BMP, TIFF, WEBP"
        )
        
        if uploaded_file is not None:
            # Открываем изображение
            try:
                input_image = Image.open(uploaded_file).convert('RGB')
                
                # Ограничиваем размер для предпросмотра
                display_image = input_image.copy()
                if max(input_image.size) > 800:
                    display_image.thumbnail((800, 800), Image.Resampling.LANCZOS)
                
                # Показываем оригинальное изображение
                st.image(display_image, caption="Оригинальное изображение", use_column_width=True)
                
                # Информация об изображении
                st.info(f"""
                **Информация об изображении:**
                - Размер: {input_image.size[0]} x {input_image.size[1]} пикселей
                - Формат: {input_image.format if hasattr(input_image, 'format') else 'Unknown'}
                - Режим: {input_image.mode}
                """)
                
                # Слайдер для выбора масштаба
                scale = st.slider(
                    "Масштаб улучшения",
                    min_value=2,
                    max_value=4,
                    value=4,
                    help="Во сколько раз увеличить изображение"
                )
                
                # Кнопка для улучшения
                if st.button("🚀 Улучшить качество", type="primary", use_container_width=True):
                    with st.spinner("Улучшение качества... Это может занять несколько минут"):
                        # Временно изменяем масштаб в upsampler
                        original_scale = upsampler.scale
                        upsampler.scale = scale
                        
                        # Улучшаем изображение
                        enhanced_image = enhance_image(input_image, upsampler)
                        
                        # Возвращаем оригинальный масштаб
                        upsampler.scale = original_scale
                        
                        if enhanced_image is not None:
                            # Сохраняем результаты
                            input_path = uploads_dir / uploaded_file.name
                            output_path = results_dir / f"enhanced_{uploaded_file.name.split('.')[0]}.png"
                            
                            input_image.save(input_path)
                            enhanced_image.save(output_path, 'PNG', quality=95)
                            
                            # Показываем результат
                            with col2:
                                st.header("✨ Улучшенное изображение")
                                
                                # Ограничиваем размер для отображения
                                display_enhanced = enhanced_image.copy()
                                if max(enhanced_image.size) > 800:
                                    display_enhanced.thumbnail((800, 800), Image.Resampling.LANCZOS)
                                
                                st.image(display_enhanced, 
                                       caption=f"Улучшенное изображение (x{scale})", 
                                       use_column_width=True)
                                
                                st.success(f"""
                                **Результат:**
                                - Новый размер: {enhanced_image.size[0]} x {enhanced_image.size[1]} пикселей
                                - Увеличение: x{scale}
                                - Формат: PNG
                                """)
                                
                                # Скачивание результата
                                with open(output_path, "rb") as file:
                                    st.download_button(
                                        label="📥 Скачать улучшенное изображение",
                                        data=file,
                                        file_name=f"enhanced_{uploaded_file.name.split('.')[0]}.png",
                                        mime="image/png",
                                        use_container_width=True
                                    )
                            
                            # Сравнение
                            st.markdown("---")
                            st.subheader("📊 Сравнение до и после")
                            
                            compare_col1, compare_col2 = st.columns(2)
                            with compare_col1:
                                st.image(input_image, 
                                       caption="До обработки", 
                                       use_column_width=True)
                            with compare_col2:
                                st.image(enhanced_image, 
                                       caption="После обработки", 
                                       use_column_width=True)
                            
                            # Информация об улучшении
                            st.metric(
                                label="Увеличение разрешения",
                                value=f"x{scale}",
                                delta=f"{enhanced_image.size[0] * enhanced_image.size[1] / (input_image.size[0] * input_image.size[1]):.1f}x пикселей"
                            )
            except Exception as e:
                st.error(f"Ошибка при обработке изображения: {e}")
                st.exception(e)
    
    # Если файл не загружен, показываем примеры
    if uploaded_file is None:
        with col2:
            st.header("📋 Инструкция")
            st.markdown("""
            1. **Загрузите** изображение пейзажа в левой колонке
            2. **Выберите** масштаб улучшения (2x, 3x или 4x)
            3. **Нажмите** кнопку "Улучшить качество"
            4. **Дождитесь** обработки (может занять несколько минут)
            5. **Скачайте** результат
            
            **Рекомендуется для:**
            - Пейзажей и природы
            - Городских видов
            - Архитектурных фотографий
            - Фото с хорошим исходным качеством
            
            **Ограничения:**
            - Максимальный размер: 3000x3000 пикселей
            - Форматы: JPG, PNG, BMP, TIFF, WEBP
            - Требует ~2GB свободной памяти
            """)
    
    # Информация о системе
    with st.expander("📊 Информация о системе"):
        st.markdown(f"""
        **Версии библиотек:**
        - PyTorch: {torch.__version__}
        - NumPy: {np.__version__}
        - Pillow: {Image.__version__}
        - OpenCV: {cv2.__version__}
        
        **Система:**
        - Устройство: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}
        - Память модели: 1.07 GB
        - Папка моделей: {models_dir.absolute()}
        - Папка результатов: {results_dir.absolute()}
        """)
        
        if torch.cuda.is_available():
            st.success(f"✅ GPU доступен: {torch.cuda.get_device_name(0)}")
            st.info(f"Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            st.warning("⚠️ GPU не обнаружен. Обработка будет на CPU (медленнее).")

# Запуск приложения
if __name__ == "__main__":
    # Проверяем базовые зависимости
    try:
        import streamlit
        st.success("✅ Streamlit готов к работе")
    except ImportError:
        st.error("Streamlit не установлен. Установите: pip install streamlit")
        st.stop()
    
    # Запускаем основное приложение
    main()
