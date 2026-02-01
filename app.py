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
        
        # Определяем максимальный размер для отображения
        max_display_size = 400
        
        # Вычисляем новый размер с сохранением пропорций
        if image.width > max_display_size or image.height > max_display_size:
            ratio = min(max_display_size / image.width, max_display_size / image.height)
            new_width = int(image.width * ratio)
            new_height = int(image.height * ratio)
            display_image = image.resize((new_width, new_height))
        else:
            display_image = image.copy()
            new_width = image.width
            new_height = image.height
        
        # Показываем изображение
        st.subheader("Предпросмотр")
        st.image(display_image, use_column_width=False, width=new_width)
        
        # Простая обработка
        if st.button("Обработать изображение", type="primary"):
            with st.spinner("Обработка..."):
                # Создаем "улучшенную" версию (просто изменяем размер для демонстрации)
                enhanced_size = 256
                enhanced = image.resize((enhanced_size, enhanced_size))
                
                # Для сравнения создаем версию оригинала того же размера
                original_for_comparison = image.resize((enhanced_size, enhanced_size))
                
                st.markdown("---")
                st.subheader("📊 Сравнение результатов")
                
                # Создаем две колонки одинаковой ширины
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### До")
                    st.image(original_for_comparison, 
                            caption=f"Оригинал ({enhanced_size}x{enhanced_size})",
                            use_column_width=True)
                
                with col2:
                    st.markdown("### После")
                    st.image(enhanced, 
                            caption=f"Улучшенная версия ({enhanced_size}x{enhanced_size})",
                            use_column_width=True)
                
                # Разделитель
                st.markdown("---")
                
                # Таблица сравнения
                st.subheader("📈 Статистика")
                comp_col1, comp_col2, comp_col3 = st.columns(3)
                
                with comp_col1:
                    st.metric("Размер оригинала", f"{image.width}×{image.height}")
                
                with comp_col2:
                    st.metric("Размер после обработки", f"{enhanced_size}×{enhanced_size}")
                
                with comp_col3:
                    st.metric("Коэффициент", f"{enhanced_size/max(image.width, image.height):.2f}x")
                
                # Скачивание
                st.markdown("---")
                st.subheader("💾 Скачать результат")
                
                # Две кнопки скачивания в одном ряду
                dl_col1, dl_col2 = st.columns(2)
                
                with dl_col1:
                    # Кнопка для скачивания улучшенного изображения
                    buf_enhanced = io.BytesIO()
                    enhanced.save(buf_enhanced, format="PNG")
                    st.download_button(
                        "📥 Скачать улучшенную версию",
                        buf_enhanced.getvalue(),
                        "enhanced_image.png",
                        "image/png",
                        use_container_width=True
                    )
                
                with dl_col2:
                    # Кнопка для скачивания оригинала
                    buf_original = io.BytesIO()
                    image.save(buf_original, format="PNG")
                    st.download_button(
                        "📥 Скачать оригинал",
                        buf_original.getvalue(),
                        "original_image.png",
                        "image/png",
                        use_container_width=True
                    )
                
                # Кнопка для обработки другого изображения
                st.markdown("---")
                if st.button("🔄 Обработать другое изображение", use_container_width=True):
                    st.rerun()
                
    except Exception as e:
        st.error(f"Ошибка: {e}")
else:
    # Инструкция когда изображение не загружено
    st.info("👆 Загрузите изображение выше для начала работы")

# Информация
with st.expander("ℹ️ О приложении"):
    st.write("""
    ## Как это работает?
    
    1. **Загрузите** изображение в формате PNG, JPG или JPEG
    2. **Нажмите** кнопку "Обработать изображение"
    3. **Сравните** результаты "До" и "После" обработки
    4. **Скачайте** улучшенную версию
    
    ## Технические детали:
    
    - **Текущая версия**: Демо-режим (имитация обработки)
    - **Исходный размер**: Сохраняется оригинальный размер
    - **Размер обработки**: Все изображения масштабируются до 256×256 пикселей
    - **Сравнение**: Оригинал и результат показываются одинакового размера для наглядности
    
    ## Для полноценной работы необходимо:
    1. Убедиться что файл модели в папке models/
    2. Добавить код загрузки модели PyTorch
    3. Реализовать функцию улучшения изображений с использованием нейросети
    """)

# Футер
st.markdown("---")
st.caption("🎯 Streamlit Cloud | Улучшение изображений | Версия 1.0")

# Кнопка перезагрузки
if st.button("🔄 Перезагрузить приложение", type="secondary"):
    st.rerun()
