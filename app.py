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
else:
    st.error(f"❌ Файл модели не найден: {model_path}")
    st.write("Содержимое папки models:")
    if os.path.exists("models"):
        st.write(os.listdir("models"))
    else:
        st.write("Папка models не существует")

# Загрузка изображения
uploaded_file = st.file_uploader("Выберите изображение", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.success(f"✅ Изображение загружено: {image.size[0]}x{image.size[1]}")
        
        # Показываем изображение
        st.subheader("Предпросмотр")
        st.image(image, use_column_width=True)
        
        # Простая обработка
        if st.button("Обработать изображение"):
            with st.spinner("Обработка..."):
                # Просто изменяем размер для демонстрации
                enhanced = image.resize((256, 256))
                
                # Показываем результат
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Оригинал")
                with col2:
                    st.image(enhanced, caption="Обработанная версия")
                
                # Кнопка скачивания
                buf = io.BytesIO()
                enhanced.save(buf, format="PNG")
                
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
    Это демо-версия приложения для улучшения качества изображений.
    
    Для полноценной работы необходимо:
    1. Убедиться что файл модели в папке models/
    2. Добавить код загрузки модели PyTorch
    3. Реализовать функцию улучшения изображений
    """)

st.markdown("---")
st.caption("Streamlit Cloud | Улучшение изображений")
