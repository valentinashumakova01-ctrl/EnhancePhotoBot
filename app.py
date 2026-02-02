import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import urllib.request
import time
from pathlib import Path

# Настройки страницы
st.set_page_config(
    page_title="Улучшение качества пейзажных фото",
    page_icon="🌄",
    layout="wide"
)

# Заголовок приложения
st.title("🌄 Улучшение качества пейзажных фото")
st.markdown("Использует модель Real-ESRGAN для улучшения качества изображений в 4 раза")

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

models_dir, uploads_dir, results_dir = setup_directories()

# Функция для загрузки модели
@st.cache_resource
def download_and_load_model():
    """Загружает и загружает модель Real-ESRGAN"""
    
    # URL модели
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    model_path = models_dir / "RealESRGAN_x4plus.pth"
    
    progress_bar = None
    status_text = None
    
    # Проверяем, есть ли уже модель
    if not model_path.exists():
        with st.spinner("Загрузка модели Real-ESRGAN (1.07 GB)... Это может занять несколько минут"):
            # Создаем индикатор прогресса
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def download_progress(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
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
    
    # Загружаем архитектуру модели
    try:
        # Импортируем здесь, чтобы избежать ошибок при отсутствии зависимостей
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
        
        # Создаем модель
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        # Создаем улучшатель
        upsampler = RealESRGANer(
            scale=4,
            model_path=str(model_path),
            model=model,
            tile=400,  # Размер тайла для обработки больших изображений
            tile_pad=10,
            pre_pad=0,
            half=False  # Используем float32 для лучшей точности
        )
        
        return upsampler
    except ImportError:
        st.error("""
        **Требуемые библиотеки не установлены!**
        
        Установите их командой:
        ```bash
        pip install basicsr facexlib gfpgan realesrgan
        ```
        
        Для Windows может потребоваться установить Visual Studio Build Tools.
        """)
        return None

# Функция для улучшения изображения
def enhance_image(input_image, upsampler):
    """Улучшает качество изображения"""
    try:
        # Конвертируем PIL Image в numpy array
        img = np.array(input_image)
        
        # Улучшаем изображение
        output, _ = upsampler.enhance(img, outscale=4)
        
        # Конвертируем обратно в PIL Image
        output_img = Image.fromarray(output)
        
        return output_img
    except Exception as e:
        st.error(f"Ошибка при улучшении изображения: {e}")
        return None

# Основной интерфейс
def main():
    # Загружаем модель
    with st.spinner("Загрузка модели..."):
        upsampler = download_and_load_model()
    
    if upsampler is None:
        st.warning("Пожалуйста, установите необходимые библиотеки и перезапустите приложение")
        return
    
    # Создаем две колонки для интерфейса
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Загрузите изображение")
        
        # Загрузка файла
        uploaded_file = st.file_uploader(
            "Выберите изображение",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Загрузите изображение пейзажа для улучшения качества"
        )
        
        if uploaded_file is not None:
            # Открываем изображение
            input_image = Image.open(uploaded_file)
            
            # Показываем оригинальное изображение
            st.image(input_image, caption="Оригинальное изображение", use_column_width=True)
            
            # Показываем информацию об изображении
            st.info(f"Размер: {input_image.size[0]}x{input_image.size[1]} пикселей")
            
            # Кнопка для улучшения
            if st.button("🚀 Улучшить качество", type="primary"):
                with st.spinner("Улучшение качества..."):
                    # Улучшаем изображение
                    enhanced_image = enhance_image(input_image, upsampler)
                    
                    if enhanced_image is not None:
                        # Сохраняем результаты
                        input_path = uploads_dir / uploaded_file.name
                        output_path = results_dir / f"enhanced_{uploaded_file.name}"
                        
                        input_image.save(input_path)
                        enhanced_image.save(output_path)
                        
                        # Показываем результат во второй колонке
                        with col2:
                            st.header("✨ Результат")
                            st.image(enhanced_image, caption="Улучшенное изображение", use_column_width=True)
                            st.success(f"Новый размер: {enhanced_image.size[0]}x{enhanced_image.size[1]} пикселей")
                            
                            # Скачивание результата
                            with open(output_path, "rb") as file:
                                btn = st.download_button(
                                    label="📥 Скачать улучшенное изображение",
                                    data=file,
                                    file_name=f"enhanced_{uploaded_file.name}",
                                    mime="image/png"
                                )
                        
                        # Показываем сравнение
                        st.markdown("---")
                        st.subheader("📊 Сравнение")
                        
                        compare_col1, compare_col2 = st.columns(2)
                        with compare_col1:
                            st.image(input_image, caption="До", use_column_width=True)
                        with compare_col2:
                            st.image(enhanced_image, caption="После", use_column_width=True)
    
    # Примеры изображений
    if uploaded_file is None:
        with col2:
            st.header("📋 Примеры")
            st.markdown("""
            **Рекомендуемые типы изображений:**
            - Пейзажи
            - Природа
            - Городские виды
            - Архитектура
            
            **Форматы:** JPG, PNG, BMP, TIFF
            
            **Максимальный размер:** 3000x3000 пикселей
            """)
            
            st.info("""
            ⚠️ **Примечание:**
            - Обработка может занять несколько минут
            - Изображение будет увеличено в 4 раза
            - Для лучших результатов используйте качественные исходные изображения
            """)
    
    # Информация о модели
    with st.expander("ℹ️ О модели Real-ESRGAN"):
        st.markdown("""
        **Real-ESRGAN** - это модель для сверхразрешения изображений, специально обученная для обработки:
        
        - **Пейзажей** - горы, леса, водопады
        - **Городских видов** - здания, улицы
        - **Природных сцен** - закаты, рассветы, облака
        
        **Особенности:**
        - Увеличение разрешения в 4 раза
        - Улучшение детализации
        - Сохранение естественных цветов
        - Обработка артефактов сжатия
        """)

if __name__ == "__main__":
    # Предупреждение о зависимостях
    st.sidebar.title("⚠️ Требования")
    st.sidebar.markdown("""
    Для работы приложения необходимо установить:
    
    ```bash
    pip install streamlit pillow torch torchvision numpy
    pip install basicsr facexlib gfpgan realesrgan
    ```
    
    **Первая загрузка модели может занять 10-15 минут.**
    """)
    
    # Информация о системе
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Информация о системе:**")
    st.sidebar.markdown(f"Модель: RealESRGAN_x4plus.pth")
    st.sidebar.markdown(f"Размер модели: 1.07 GB")
    
    main()
