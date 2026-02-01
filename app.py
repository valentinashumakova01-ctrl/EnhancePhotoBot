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

# Проверяем версии
st.write(f"PyTorch версия: {torch.__version__}")
st.write(f"CUDA доступен: {torch.cuda.is_available()}")

# Проверяем файл модели
model_path = "models/enhanced_epoch_28_ratio_1.23.pth"
if os.path.exists(model_path):
    st.success(f"✅ Файл модели найден: {model_path}")
    st.write(f"Размер файла: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    
    # Пробуем загрузить модель
    try:
        # Определяем устройство
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.write(f"Используется устройство: {device}")
        
        # Пробуем загрузить модель (для PyTorch 2.10.0)
        try:
            # Сначала пробуем стандартный способ
            checkpoint = torch.load(model_path, map_location=device)
            st.success("✅ Модель загружена успешно!")
        except:
            # Если не получается, пробуем с weights_only=False
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            st.success("✅ Модель загружена с weights_only=False!")
        
    except Exception as e:
        st.warning(f"⚠️ Ошибка при загрузке модели: {e}")
        
else:
    st.error(f"❌ Файл модели не найден: {model_path}")

# Загрузка изображения
uploaded_file = st.file_uploader("Выберите изображение", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.success(f"✅ Изображение загружено: {image.size[0]}x{image.size[1]}")
        
        # Показываем изображение
        st.subheader("Предпросмотр")
        st.image(image, use_column_width=True)
        
        # Обработка
        if st.button("Обработать изображение", type="primary"):
            with st.spinner("Обработка..."):
                # ФИКСИРОВАННЫЙ РАЗМЕР: 128x128 пикселей
                TARGET_SIZE = 128
                
                # Создаем уменьшенные версии ОДИНАКОВОГО размера
                original_128 = image.resize((TARGET_SIZE, TARGET_SIZE))
                enhanced_128 = image.resize((TARGET_SIZE, TARGET_SIZE))
                
                # Показываем результат - ОДИНАКОВОГО РАЗМЕРА
                st.subheader(f"Результат ({TARGET_SIZE}×{TARGET_SIZE})")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**До обработки**")
                    st.image(original_128, use_column_width=True)
                with col2:
                    st.markdown("**После обработки**")
                    st.image(enhanced_128, use_column_width=True)
                
                # Кнопка скачивания
                buf = io.BytesIO()
                enhanced_128.save(buf, format="PNG")
                
                st.download_button(
                    "📥 Скачать результат",
                    buf.getvalue(),
                    "enhanced_image.png",
                    "image/png"
                )
                
    except Exception as e:
        st.error(f"Ошибка: {e}")

# Информация
with st.expander("ℹ️ О приложении"):
    st.write("""
    ## Информация о системе:
    
    **Установленные версии:**
    - Streamlit: 1.53.1
    - PyTorch: 2.10.0
    - Pillow: 12.1.0
    
    **Файл модели:** enhanced_epoch_28_ratio_1.23.pth
    
    **Режим работы:** Демонстрационный (пока без реальной нейросети)
    
    Для добавления нейросети необходимо:
    1. Добавить классы модели PyTorch
    2. Реализовать загрузку весов
    3. Добавить функцию обработки
    """)

st.markdown("---")
st.caption("Streamlit Cloud | Улучшение изображений | PyTorch 2.10.0")
