import streamlit as st

# ВАЖНО: Эта строка должна быть самой первой
st.set_page_config(
    page_title="Улучшение изображений",
    page_icon="🖼️",
    layout="wide"
)

# Импортируем все остальное ПОСЛЕ set_page_config
import torch
from PIL import Image
import io
import os
import numpy as np
from torchvision import transforms

st.title("🖼️ Улучшение качества изображений")

# Определяем устройство
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Информация о системе
col1, col2, col3 = st.columns(3)
with col1:
    st.info(f"PyTorch: {torch.__version__}")
with col2:
    st.info(f"NumPy: {np.__version__}")
with col3:
    st.info(f"Устройство: {'GPU 🚀' if device == 'cuda' else 'CPU ⚙️'}")

# 1. Классы моделей
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

# 2. Основная функция загрузки модели
@st.cache_resource
def load_model():
    weights_path = "models/enhanced_epoch_28_ratio_1.23.pth"
    
    if not os.path.exists(weights_path):
        st.error(f"❌ Файл не найден: {weights_path}")
        st.info("Поместите файл с весами в папку models/")
        return None, None
    
    try:
        # Способ 1: Используем safe_globals
        import torch.serialization
        
        # Добавляем необходимые глобальные объекты
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        
        # Загружаем с weights_only=False
        checkpoint = torch.load(
            weights_path, 
            map_location=device,
            weights_only=False
        )
        
        model = StrongGenerator().to(device)
        
        # Ищем правильный ключ
        if 'generator' in checkpoint:
            model.load_state_dict(checkpoint['generator'])
            st.success("✅ Модель загружена (ключ 'generator')")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success("✅ Модель загружена (ключ 'model_state_dict')")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            st.success("✅ Модель загружена (ключ 'state_dict')")
        else:
            # Пробуем загрузить напрямую
            try:
                model.load_state_dict(checkpoint)
                st.success("✅ Модель загружена (прямая загрузка)")
            except Exception as e:
                st.error(f"❌ Не удалось загрузить веса: {e}")
                if hasattr(checkpoint, 'keys'):
                    st.info(f"Доступные ключи: {list(checkpoint.keys())}")
                return None, None
        
        model.eval()
        return model, device
        
    except Exception as e:
        st.error(f"❌ Способ 1 не сработал: {str(e)}")
        
        # Пробуем альтернативный способ
        return load_model_alternative()

# 3. Альтернативная функция загрузки модели
@st.cache_resource
def load_model_alternative():
    weights_path = "models/enhanced_epoch_28_ratio_1.23.pth"
    
    if not os.path.exists(weights_path):
        return None, None
    
    try:
        # Способ 2: Используем контекстный менеджер safe_globals
        import torch.serialization
        
        with torch.serialization.safe_globals([np.core.multiarray.scalar]):
            checkpoint = torch.load(
                weights_path, 
                map_location=device,
                weights_only=False
            )
        
        model = StrongGenerator().to(device)
        
        if 'generator' in checkpoint:
            model.load_state_dict(checkpoint['generator'])
            st.success("✅ Модель загружена (способ 2: generator)")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success("✅ Модель загружена (способ 2: model_state_dict)")
        else:
            model.load_state_dict(checkpoint)
            st.success("✅ Модель загружена (способ 2: прямая загрузка)")
        
        model.eval()
        return model, device
        
    except Exception as e:
        st.error(f"❌ Способ 2 не сработал: {str(e)}")
        
        # Способ 3: Используем pickle напрямую
        try:
            import pickle
            
            with open(weights_path, 'rb') as f:
                checkpoint = pickle.load(f)
            
            model = StrongGenerator().to(device)
            
            if 'generator' in checkpoint:
                model.load_state_dict(checkpoint['generator'])
                st.success("✅ Модель загружена (способ 3: generator)")
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                st.success("✅ Модель загружена (способ 3: model_state_dict)")
            else:
                model.load_state_dict(checkpoint)
                st.success("✅ Модель загружена (способ 3: прямая загрузка)")
            
            model.eval()
            return model, device
            
        except Exception as e3:
            st.error(f"❌ Способ 3 не сработал: {str(e3)}")
            
            # Способ 4: Пробуем загрузить по-старому
            try:
                # Для очень старых версий PyTorch
                checkpoint = torch.load(
                    weights_path, 
                    map_location=device
                )
                
                model = StrongGenerator().to(device)
                
                if 'generator' in checkpoint:
                    model.load_state_dict(checkpoint['generator'])
                    st.success("✅ Модель загружена (способ 4: generator)")
                else:
                    model.load_state_dict(checkpoint)
                    st.success("✅ Модель загружена (способ 4: прямая загрузка)")
                
                model.eval()
                return model, device
                
            except Exception as e4:
                st.error(f"❌ Все способы не сработали: {str(e4)}")
                return None, None

# 4. Обработка изображения
def enhance_image(image, model, device):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    output_tensor = output_tensor.squeeze(0).cpu()
    output_img = output_tensor * 0.5 + 0.5
    output_img = torch.clamp(output_img, 0, 1)
    output_img = transforms.ToPILImage()(output_img)
    
    return output_img

# 5. Загружаем модель
st.markdown("---")
with st.spinner("🔄 Загружаем модель..."):
    model, device = load_model()

if model is None:
    st.error("❌ Не удалось загрузить модель. Проверьте файл с весами.")
    
    # Показать информацию для отладки
    with st.expander("🔧 Отладочная информация"):
        st.code(f"""
Текущая директория: {os.getcwd()}
Содержимое models/: {os.listdir('models') if os.path.exists('models') else 'Папка не существует'}
PyTorch: {torch.__version__}
NumPy: {np.__version__}
Устройство: {device}
        """)
    
    st.stop()

# 6. Интерфейс
st.subheader("📤 Загрузите изображение")
uploaded_file = st.file_uploader(
    "Выберите файл изображения", 
    type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
    label_visibility="collapsed"
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert('RGB')
        st.success(f"✅ Изображение загружено: {image.size[0]}×{image.size[1]} пикселей")
        
        # Предпросмотр
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Оригинал")
            # Уменьшаем для предпросмотра если слишком большое
            if image.size[0] > 500 or image.size[1] > 500:
                preview = image.copy()
                preview.thumbnail((500, 500))
                st.image(preview, use_column_width=True)
                st.caption(f"Предпросмотр (оригинал: {image.size[0]}×{image.size[1]})")
            else:
                st.image(image, use_column_width=True)
        
        # Кнопка обработки
        st.markdown("---")
        if st.button("✨ УЛУЧШИТЬ КАЧЕСТВО", type="primary", use_container_width=True):
            with st.spinner("🔄 Обрабатываем изображение..."):
                try:
                    # Показать прогресс
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Подготавливаем изображение...")
                    progress_bar.progress(30)
                    
                    enhanced = enhance_image(image, model, device)
                    
                    status_text.text("Завершаем обработку...")
                    progress_bar.progress(90)
                    
                    with col2:
                        st.subheader("Улучшенное")
                        st.image(enhanced, use_column_width=True)
                        st.caption(f"Размер: 128×128 пикселей")
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Обработка завершена!")
                    
                    # Сравнение
                    st.markdown("---")
                    st.subheader("📊 Сравнение")
                    
                    compare_col1, compare_col2 = st.columns(2)
                    with compare_col1:
                        st.image(image.resize((256, 256)), caption="Оригинал (256×256)")
                    with compare_col2:
                        st.image(enhanced, caption="Улучшенная версия (128×128)")
                    
                    # Скачивание
                    st.markdown("---")
                    st.subheader("💾 Скачать результат")
                    
                    # Формат выбора
                    format_option = st.radio(
                        "Выберите формат для скачивания:",
                        ["PNG (рекомендуется)", "JPEG", "WEBP"],
                        horizontal=True
                    )
                    
                    # Подготовка файла
                    if "PNG" in format_option:
                        buf = io.BytesIO()
                        enhanced.save(buf, format="PNG", optimize=True)
                        mime_type = "image/png"
                        file_name = "enhanced_image.png"
                    elif "JPEG" in format_option:
                        buf = io.BytesIO()
                        enhanced.save(buf, format="JPEG", quality=95, optimize=True)
                        mime_type = "image/jpeg"
                        file_name = "enhanced_image.jpg"
                    else:  # WEBP
                        buf = io.BytesIO()
                        enhanced.save(buf, format="WEBP", quality=90)
                        mime_type = "image/webp"
                        file_name = "enhanced_image.webp"
                    
                    # Кнопки скачивания
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    
                    with col_dl1:
                        st.download_button(
                            label=f"📥 Скачать улучшенное",
                            data=buf.getvalue(),
                            file_name=file_name,
                            mime=mime_type,
                            use_container_width=True
                        )
                    
                    with col_dl2:
                        buf_orig = io.BytesIO()
                        image.save(buf_orig, format="PNG", optimize=True)
                        st.download_button(
                            label="📥 Скачать оригинал",
                            data=buf_orig.getvalue(),
                            file_name="original_image.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                    with col_dl3:
                        if st.button("🔄 Новое изображение", use_container_width=True):
                            st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Ошибка обработки: {str(e)}")
    
    except Exception as e:
        st.error(f"❌ Ошибка при открытии файла: {str(e)}")
else:
    st.info("👆 Загрузите изображение выше")

# Информация о модели
with st.expander("ℹ️ Информация о модели"):
    st.markdown("""
    ### Технические детали
    
    **Архитектура модели:**
    - StrongGenerator с 6 остаточными блоками (Residual Blocks)
    - Входной размер: 128×128 пикселей
    - Выходной размер: 128×128 пикселей
    - Коэффициент улучшения: ×1.23 (по названию файла)
    
    **Загрузка модели:**
    Приложение пробует 4 разных способа загрузки модели для совместимости
    с разными версиями PyTorch.
    
    **Обработка изображений:**
    1. Изображение масштабируется до 128×128 пикселей
    2. Нормализуется значения пикселей
    3. Обрабатывается нейросетью
    4. Результат преобразуется обратно в изображение
    """)

# Отладочная информация
with st.expander("🔧 Техническая информация"):
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        st.write(f"**Параметры модели:** {total_params:,}")
    
    st.write(f"**Путь к весам:** models/enhanced_epoch_28_ratio_1.23.pth")
    st.write(f"**Размер файла:** {os.path.getsize('models/enhanced_epoch_28_ratio_1.23.pth') / 1024 / 1024:.2f} MB")

# Футер
st.markdown("---")
st.caption("🎯 Улучшение качества изображений | PyTorch + Streamlit | Версия 1.0")

# Кнопка перезагрузки
if st.button("🔄 Перезагрузить приложение", type="secondary"):
    st.rerun()
