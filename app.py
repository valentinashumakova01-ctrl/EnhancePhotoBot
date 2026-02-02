import subprocess
import sys
import importlib
import platform
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

# Определяем ОС
SYSTEM = platform.system().lower()

# Функция для установки Windows-специфичных зависимостей
def install_windows_dependencies():
    """Устанавливает зависимости для Windows"""
    
    if SYSTEM != 'windows':
        return True
    
    st.info("Обнаружена система Windows. Настройка окружения...")
    
    # Для Windows используем opencv-python вместо opencv-python-headless
    windows_packages = [
        'opencv-python>=4.8.0',  # Для Windows используем обычный opencv
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.0',
        'pillow>=10.0.0',
        'scipy>=1.10.0',
        'lmdb>=1.4.1',
        'tqdm>=4.65.0',
        'yapf>=0.32.0',
        'packaging>=21.3',
        'pyyaml>=6.0',
        'streamlit>=1.28.0',
    ]
    
    # Для Windows сначала установим базовые пакеты
    with st.spinner("Установка базовых зависимостей для Windows..."):
        for package in windows_packages:
            package_name = package.split('>=')[0]
            st.write(f"Установка {package_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                st.success(f"✅ {package_name}")
            except Exception as e:
                st.warning(f"⚠️ Ошибка установки {package_name}: {e}")
    
    return True

# Функция для установки Real-ESRGAN зависимостей
def install_esrgan_dependencies():
    """Устанавливает зависимости для Real-ESRGAN"""
    
    st.info("Установка Real-ESRGAN зависимостей...")
    
    # Для Windows есть проблемы с некоторыми пакетами, устанавливаем по одному
    esrgan_packages = [
        'basicsr==1.4.2',
        'facexlib==0.3.0',
        'gfpgan==1.3.8',
        'realesrgan==0.3.0',
    ]
    
    success = True
    for package in esrgan_packages:
        package_name = package.split('==')[0]
        try:
            importlib.import_module(package_name.replace('-', '_'))
            st.success(f"✅ {package_name} уже установлен")
        except ImportError:
            with st.spinner(f"Установка {package_name}..."):
                try:
                    # Пробуем разные варианты установки
                    commands = [
                        [sys.executable, "-m", "pip", "install", package],
                        [sys.executable, "-m", "pip", "install", f"{package} --no-deps"],
                        [sys.executable, "-m", "pip", "install", f"git+https://github.com/xinntao/{package_name}.git"]
                    ]
                    
                    installed = False
                    for cmd in commands:
                        try:
                            subprocess.run(cmd, check=True, capture_output=True, text=True)
                            st.success(f"✅ {package_name} установлен")
                            installed = True
                            break
                        except:
                            continue
                    
                    if not installed:
                        st.warning(f"⚠️ {package_name} может потребовать ручной установки")
                        success = False
                        
                except Exception as e:
                    st.error(f"❌ Ошибка установки {package_name}: {e}")
                    success = False
    
    return success

# Основной интерфейс
def main():
    # Сайдбар с управлением
    st.sidebar.title("⚙️ Управление")
    
    if st.sidebar.button("🔄 Установить/Обновить зависимости", type="primary"):
        with st.spinner("Настройка окружения Windows..."):
            # Устанавливаем Windows зависимости
            install_windows_dependencies()
            
            # Устанавливаем ESRGAN зависимости
            install_esrgan_dependencies()
            
            st.success("✅ Зависимости установлены! Перезапустите приложение.")
            st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Для Windows:
    1. Нажмите кнопку установки зависимостей
    2. Если есть ошибки, установите вручную:
    ```
    pip install torch torchvision numpy pillow
    pip install opencv-python streamlit
    ```
    3. Перезапустите приложение
    """)
    
    # Создаем директории
    @st.cache_resource
    def setup_directories():
        models_dir = Path("models")
        uploads_dir = Path("uploads")
        results_dir = Path("results")
        
        models_dir.mkdir(exist_ok=True)
        uploads_dir.mkdir(exist_ok=True)
        results_dir.mkdir(exist_ok=True)
        
        return models_dir, uploads_dir, results_dir
    
    models_dir, uploads_dir, results_dir = setup_directories()
    
    # Пытаемся импортировать библиотеки
    try:
        import torch
        import numpy as np
        from PIL import Image
        import urllib.request
        import time
        import cv2
        
        st.success("✅ Основные библиотеки загружены")
        
        # Пробуем импортировать Real-ESRGAN
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            st.success("✅ Real-ESRGAN библиотеки загружены")
            
        except ImportError as e:
            st.warning(f"""
            ⚠️ Real-ESRGAN библиотеки не найдены:
            {e}
            
            Нажмите кнопку "Установить/Обновить зависимости" в сайдбаре
            """)
            return
            
    except ImportError as e:
        st.error(f"""
        ❌ Ошибка импорта: {e}
        
        Для Windows установите:
        1. Python 3.8-3.10
        2. Microsoft Visual C++ Redistributable
        3. Зависимости через кнопку в сайдбаре
        """)
        return
    
    # Функция для загрузки модели
    @st.cache_resource
    def download_and_load_model():
        """Загружает модель Real-ESRGAN"""
        
        model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        model_path = models_dir / "RealESRGAN_x4plus.pth"
        
        if not model_path.exists():
            with st.spinner("Загрузка модели Real-ESRGAN (1.07 GB)..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def download_progress(count, block_size, total_size):
                    percent = min(int(count * block_size * 100 / total_size), 100)
                    progress_bar.progress(percent / 100)
                    status_text.text(f"Загрузка: {percent}%")
                
                try:
                    urllib.request.urlretrieve(
                        model_url, 
                        model_path, 
                        reporthook=download_progress
                    )
                    st.success("✅ Модель загружена!")
                    progress_bar.empty()
                    status_text.empty()
                except Exception as e:
                    st.error(f"Ошибка загрузки: {e}")
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
            
            # Определяем устройство
            if torch.cuda.is_available():
                device = torch.device('cuda')
                st.success(f"✅ Используется GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device('cpu')
                st.info("ℹ️ Используется CPU (рекомендуется GPU для скорости)")
            
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
            st.error(f"Ошибка создания модели: {e}")
            st.exception(e)
            return None
    
    # Функция улучшения изображения (упрощенная для Windows)
    def enhance_image_simple(input_image, upsampler):
        """Упрощенная функция улучшения для Windows"""
        try:
            # Конвертируем PIL в numpy
            img = np.array(input_image)
            
            # Для Windows используем простую конвертацию
            if len(img.shape) == 3:
                # OpenCV ожидает BGR, а PIL RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Улучшаем изображение
            output, _ = upsampler.enhance(img, outscale=4)
            
            # Конвертируем обратно
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            
            return Image.fromarray(output)
            
        except Exception as e:
            st.error(f"Ошибка обработки: {e}")
            # Альтернативный метод без OpenCV
            try:
                # Пробуем напрямую использовать numpy
                img_np = np.array(input_image)
                output, _ = upsampler.enhance(img_np, outscale=4)
                return Image.fromarray(output)
            except:
                return None
    
    # Загружаем модель
    with st.spinner("Загрузка модели..."):
        upsampler = download_and_load_model()
    
    if upsampler is None:
        st.error("Не удалось загрузить модель")
        return
    
    # Интерфейс загрузки
    st.header("📤 Загрузите изображение")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Выберите изображение",
            type=['jpg', 'jpeg', 'png'],
            help="JPG и PNG форматы"
        )
        
        if uploaded_file:
            try:
                input_image = Image.open(uploaded_file).convert('RGB')
                st.image(input_image, caption="Оригинал", use_column_width=True)
                
                st.info(f"Размер: {input_image.size[0]}x{input_image.size[1]}")
                
                if st.button("🚀 Улучшить качество", type="primary"):
                    with st.spinner("Обработка..."):
                        enhanced = enhance_image_simple(input_image, upsampler)
                        
                        if enhanced:
                            with col2:
                                st.header("✨ Результат")
                                st.image(enhanced, caption="Улучшенное", use_column_width=True)
                                st.success(f"Новый размер: {enhanced.size[0]}x{enhanced.size[1]}")
                                
                                # Сохраняем
                                output_path = results_dir / f"enhanced_{uploaded_file.name}"
                                enhanced.save(output_path, 'PNG')
                                
                                with open(output_path, "rb") as f:
                                    st.download_button(
                                        "📥 Скачать",
                                        f,
                                        file_name=f"enhanced_{uploaded_file.name}",
                                        mime="image/png"
                                    )
                            
                            # Сравнение
                            st.markdown("---")
                            st.subheader("📊 Сравнение")
                            
                            comp1, comp2 = st.columns(2)
                            with comp1:
                                st.image(input_image, caption="До", use_column_width=True)
                            with comp2:
                                st.image(enhanced, caption="После", use_column_width=True)
            except Exception as e:
                st.error(f"Ошибка: {e}")
    
    # Информация для Windows пользователей
    with st.expander("🖥️ Информация для Windows"):
        st.markdown("""
        ### Проблемы и решения для Windows:
        
        1. **Ошибка libGL.so.1** - игнорируйте, для Windows не нужна
        2. **Ошибка Microsoft Visual C++**:
           - Скачайте с [официального сайта](https://aka.ms/vs/17/release/vc_redist.x64.exe)
           - Установите и перезагрузите ПК
        
        3. **Медленная работа**:
           - Установите CUDA для NVIDIA GPU
           - Используйте меньшие изображения
           - Обработка на CPU медленнее
        
        4. **Ошибки установки**:
           ```
           pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
           pip install opencv-python streamlit numpy pillow
           ```
        """)
    
    # Информация о системе
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Система:**")
    st.sidebar.code(f"""
    ОС: {platform.system()} {platform.release()}
    Python: {platform.python_version()}
    PyTorch: {torch.__version__}
    CUDA: {torch.cuda.is_available()}
    """)

# Запуск
if __name__ == "__main__":
    main()
