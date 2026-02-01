# app.py
import streamlit as st
import torch
from PIL import Image
import io
import os

# Должно быть ПЕРВОЙ командой
st.set_page_config(
    page_title="Улучшение изображений",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Улучшение качества изображений")
st.write("Загрузите изображение для обработки")

# Проверяем наличие файлов
st.write("Проверка окружения:")
st.write(f"- PyTorch версия: {torch.__version__}")
st.write(f"- CUDA доступен: {torch.cuda.is_available()}")

# Проверяем файл модели
model_path = "models/enhanced_epoch_28_ratio_1.23.pth"
if os.path.exists(model_path):
    st.success(f"✅ Файл модели найден: {model_path}")
    st.write(f"Размер файла: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    
    # Пробуем загрузить модель (упрощенная версия)
    try:
        # Добавляем safe_globals для совместимости с PyTorch 2.6+
        import torch.serialization
        import numpy as np
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        
        # Загружаем модель
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        st.success("✅ Модель успешно загружена!")
        
        if 'generator' in checkpoint:
            st.write(f"Ключ 'generator' найден в файле модели")
        else:
            st.write(f"Доступные ключи: {list(checkpoint.keys())}")
            
    except Exception as e:
        st.warning(f"⚠️ Ошибка при загрузке модели: {e}")
        st.info("Приложение продолжит работу в демо-режиме")
        
else:
    st.error(f"❌ Файл модели не найден: {model_path}")
    st.write("Содержимое папки models:")
    if os.path.exists("models"):
        st.write(os.listdir("models"))
    else:
        st.write("Папка models не существует")
    st.info("Приложение будет работать в демо-режиме")

# Загрузка изображения
uploaded_file = st.file_uploader("Выберите изображение", type=['png', 'jpg', 'jpeg', 'bmp', 'webp'])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.success(f"✅ Изображение загружено: {image.size[0]}x{image.size[1]}")
        
        # Показываем оригинал
        st.subheader("Предпросмотр оригинала")
        
        # Автоматически подбираем размер для отображения
        max_preview_size = 500
        if image.width > max_preview_size or image.height > max_preview_size:
            ratio = min(max_preview_size / image.width, max_preview_size / image.height)
            preview_width = int(image.width * ratio)
            preview_height = int(image.height * ratio)
            preview_image = image.resize((preview_width, preview_height))
            st.image(preview_image, caption=f"Предпросмотр ({preview_width}x{preview_height})")
        else:
            st.image(image, caption=f"Оригинал ({image.width}x{image.height})")
        
        # Обработка
        if st.button("✨ Улучшить качество", type="primary"):
            with st.spinner("Обрабатываем изображение..."):
                try:
                    # ФИКСИРОВАННЫЙ РАЗМЕР: 128x128 пикселей
                    TARGET_SIZE = 128
                    
                    # Создаем уменьшенные версии для сравнения
                    original_128 = image.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
                    
                    # Здесь должен быть вызов реальной модели
                    # Пока используем демо-обработку (немного улучшаем контраст)
                    from PIL import ImageEnhance
                    enhanced_128 = original_128.copy()
                    enhancer = ImageEnhance.Contrast(enhanced_128)
                    enhanced_128 = enhancer.enhance(1.2)  # Увеличиваем контраст на 20%
                    enhancer = ImageEnhance.Sharpness(enhanced_128)
                    enhanced_128 = enhancer.enhance(1.5)  # Увеличиваем резкость на 50%
                    
                    st.markdown("---")
                    st.subheader("🎯 Результаты обработки (128×128 пикселей)")
                    
                    # Создаем две колонки для сравнения
                    col_before, col_after = st.columns(2)
                    
                    with col_before:
                        st.markdown("### 🟦 До обработки")
                        st.image(original_128, 
                                caption=f"Исходное изображение",
                                use_column_width=True)
                        
                        # Информация о "до"
                        with st.container():
                            st.caption("📏 Размер: 128×128 пикселей")
                            st.caption("🎨 Цветовой режим: RGB")
                            st.caption("📊 Исходный размер: {}×{}".format(image.width, image.height))
                    
                    with col_after:
                        st.markdown("### 🟩 После обработки")
                        st.image(enhanced_128, 
                                caption=f"Улучшенная версия",
                                use_column_width=True)
                        
                        # Информация о "после"
                        with st.container():
                            st.caption("📏 Размер: 128×128 пикселей")
                            st.caption("🎨 Улучшения: Контраст +20%, Резкость +50%")
                            st.caption("✨ Демо-режим: имитация работы нейросети")
                    
                    # Разделитель
                    st.markdown("---")
                    
                    # Статистика обработки
                    st.subheader("📈 Статистика обработки")
                    
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    
                    with stat_col1:
                        st.metric("Исходный размер", 
                                 f"{image.width}×{image.height}", 
                                 f"{(image.width * image.height) / 1000:.0f}K пикс")
                    
                    with stat_col2:
                        st.metric("Целевой размер", 
                                 f"{TARGET_SIZE}×{TARGET_SIZE}", 
                                 f"{(TARGET_SIZE * TARGET_SIZE) / 1000:.0f}K пикс")
                    
                    with stat_col3:
                        scale_factor = TARGET_SIZE / max(image.width, image.height)
                        st.metric("Масштаб", 
                                 f"{scale_factor:.2%}",
                                 f"1:{int(1/scale_factor)}")
                    
                    with stat_col4:
                        st.metric("Режим", 
                                 "Демо", 
                                 "Нейросеть")
                    
                    # Скачивание результатов
                    st.markdown("---")
                    st.subheader("💾 Скачать результаты")
                    
                    # Три кнопки скачивания
                    dl_col1, dl_col2, dl_col3 = st.columns(3)
                    
                    with dl_col1:
                        # Улучшенное изображение 128x128
                        buf_enhanced = io.BytesIO()
                        enhanced_128.save(buf_enhanced, format="PNG", optimize=True)
                        st.download_button(
                            "📥 Улучшенное (128×128)",
                            buf_enhanced.getvalue(),
                            "enhanced_128x128.png",
                            "image/png",
                            use_container_width=True,
                            help="Скачать улучшенную версию 128×128 пикселей"
                        )
                    
                    with dl_col2:
                        # Оригинал 128x128
                        buf_original_128 = io.BytesIO()
                        original_128.save(buf_original_128, format="PNG", optimize=True)
                        st.download_button(
                            "📥 Оригинал (128×128)",
                            buf_original_128.getvalue(),
                            "original_128x128.png",
                            "image/png",
                            use_container_width=True,
                            help="Скачать исходное изображение 128×128 пикселей"
                        )
                    
                    with dl_col3:
                        # Полный оригинал
                        buf_full_original = io.BytesIO()
                        image.save(buf_full_original, format="PNG", optimize=True)
                        st.download_button(
                            "📥 Полный оригинал",
                            buf_full_original.getvalue(),
                            "original_full.png",
                            "image/png",
                            use_container_width=True,
                            help="Скачать исходное изображение в полном размере"
                        )
                    
                    # Разделитель
                    st.markdown("---")
                    
                    # Информация о следующем шаге
                    with st.expander("🔮 Что дальше?"):
                        st.markdown("""
                        ### Для реальной работы с нейросетью:
                        
                        1. **Подготовьте модель** - убедитесь что файл модели корректный
                        2. **Добавьте классы модели** - определите архитектуру нейросети
                        3. **Реализуйте обработку** - замените демо-обработку на вызов модели
                        4. **Настройте преобразования** - добавьте нормализацию и другие преобразования
                        
                        ### Пример кода для реальной модели:
                        ```python
                        # Классы модели
                        class ResidualBlock(torch.nn.Module):
                            # ... ваш код ...
                        
                        class StrongGenerator(torch.nn.Module):
                            # ... ваш код ...
                        
                        # Загрузка модели
                        model = StrongGenerator()
                        model.load_state_dict(checkpoint['generator'])
                        model.eval()
                        
                        # Обработка изображения
                        transform = transforms.Compose([
                            transforms.Resize((128, 128)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                        ])
                        
                        input_tensor = transform(image).unsqueeze(0)
                        with torch.no_grad():
                            output_tensor = model(input_tensor)
                        ```
                        """)
                    
                    # Кнопка для новой обработки
                    if st.button("🔄 Обработать другое изображение", use_container_width=True):
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Ошибка при обработке: {e}")
                    st.info("Попробуйте другое изображение или уменьшите размер")
    
    except Exception as e:
        st.error(f"❌ Ошибка при открытии файла: {e}")
else:
    # Инструкция когда изображение не загружено
    st.info("👆 Загрузите изображение выше для начала работы")
    
    # Пример изображений
    with st.expander("🖼️ Пример ожидаемого результата"):
        st.markdown("""
        ### Как будет выглядеть результат:
        
        После загрузки изображения и нажатия кнопки "Улучшить качество", вы увидите:
        
        1. **Слева**: Исходное изображение, уменьшенное до **128×128** пикселей
        2. **Справа**: Улучшенная версия того же размера (**128×128**)
        3. **Возможность скачать** оба варианта
        
        ### Почему именно 128×128?
        - Это стандартный размер для многих моделей улучшения изображений
        - Баланс между качеством и скоростью обработки
        - Подходит для большинства нейросетевых архитектур
        """)

# Информация
with st.expander("ℹ️ Техническая информация"):
    st.write(f"""
    ## Информация о системе:
    
    - **Streamlit версия**: {st.__version__}
    - **PyTorch версия**: {torch.__version__}
    - **Pillow версия**: {Image.__version__}
    - **Устройство**: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}
    - **Размер итогового изображения**: 128×128 пикселей
    
    ## Режим работы:
    {'**Демо-режим** (имитация обработки)' if not os.path.exists(model_path) else '**Готов к работе с моделью**'}
    
    ## Структура проекта:
    ```
    {os.getcwd()}/
    ├── app.py                  # Этот файл
    ├── models/                 # Папка с моделью
    │   └── enhanced_epoch_28_ratio_1.23.pth
    └── requirements.txt        # Зависимости
    ```
    """)

# Футер
st.markdown("---")
st.caption("🎯 Улучшение качества изображений | Фиксированный размер 128×128 | Streamlit Cloud")

# Кнопка перезагрузки
if st.button("🔄 Обновить страницу", type="secondary"):
    st.rerun()
