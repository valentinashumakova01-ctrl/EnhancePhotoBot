import streamlit as st
import torch
from PIL import Image
import io
import os
import sys
import subprocess
from torchvision import transforms

# Проверяем и устанавливаем правильную версию numpy
try:
    import numpy as np
    st.success(f"NumPy версия: {np.__version__}")
except ImportError:
    st.warning("NumPy не установлен. Устанавливаем...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy<2.0.0"])
    import numpy as np

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

# 2. Функция для загрузки модели с обработкой ошибок numpy
@st.cache_resource
def load_model():
    # Сначала убедимся, что у нас правильная версия numpy
    try:
        import numpy as np
        np_version = np.__version__
        st.info(f"NumPy версия: {np_version}")
        
        # Если numpy 2.x, нужно установить 1.x
        if np_version.startswith('2.'):
            st.warning("Обнаружен NumPy 2.x, который несовместим с некоторыми моделями PyTorch.")
            st.warning("Устанавливаем NumPy 1.x...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.24.4"])
            import importlib
            importlib.reload(np)
    
    except Exception as e:
        st.error(f"Ошибка с NumPy: {e}")

    # Путь к файлу с весами
    weights_paths = [
        "models/enhanced_epoch_28_ratio_1.23.pth",
        "./models/enhanced_epoch_28_ratio_1.23.pth",
        "enhanced_epoch_28_ratio_1.23.pth"
    ]
    
    found_path = None
    for path in weights_paths:
        if os.path.exists(path):
            found_path = path
            st.success(f"Файл найден: {path}")
            break
    
    if not found_path:
        st.error("Файл с весами не найден. Искали в:")
        for path in weights_paths:
            st.error(f"  - {path}")
        st.info("Пожалуйста, поместите файл enhanced_epoch_28_ratio_1.23.pth в папку models/")
        return None, None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.info(f"Используется устройство: {'GPU' if device == 'cuda' else 'CPU'}")
    
    try:
        # Пробуем загрузить с разными параметрами
        try:
            checkpoint = torch.load(found_path, 
                                   map_location=device, 
                                   weights_only=False)
        except Exception as e1:
            st.warning(f"Первая попытка загрузки не удалась: {e1}")
            st.info("Пробуем альтернативный способ загрузки...")
            try:
                # Для старых версий PyTorch
                checkpoint = torch.load(found_path, 
                                       map_location=device)
            except Exception as e2:
                st.error(f"Вторая попытка загрузки не удалась: {e2}")
                raise
        
        model = StrongGenerator().to(device)
        
        # Пробуем разные ключи для загрузки весов
        if 'generator' in checkpoint:
            model.load_state_dict(checkpoint['generator'])
            st.success("Загружен ключ 'generator'")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success("Загружен ключ 'model_state_dict'")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            st.success("Загружен ключ 'state_dict'")
        else:
            # Пробуем загрузить напрямую
            try:
                model.load_state_dict(checkpoint)
                st.success("Загружены веса напрямую")
            except:
                st.error("Не удалось найти подходящие веса в файле")
                st.info(f"Доступные ключи: {list(checkpoint.keys())}")
                return None, None
        
        model.eval()
        st.success("✅ Модель загружена успешно!")
        return model, device
        
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        
        # Подробная информация об ошибке
        with st.expander("🔍 Подробности ошибки"):
            st.code(f"""
Ошибка: {str(e)}
PyTorch версия: {torch.__version__}
NumPy версия: {np.__version__ if 'np' in locals() else 'Не загружен'}
Устройство: {device}
Путь к файлу: {found_path}
Размер файла: {os.path.getsize(found_path) / 1024 / 1024:.2f} MB
            """)
        
        # Предложения по решению
        st.info("💡 Возможные решения:")
        st.info("1. Установите NumPy версии 1.x: `pip install numpy==1.24.4`")
        st.info("2. Проверьте совместимость версий PyTorch")
        st.info("3. Убедитесь, что файл весов не поврежден")
        
        # Кнопка для установки правильной версии numpy
        if st.button("🔄 Установить NumPy 1.24.4"):
            with st.spinner("Устанавливаем NumPy 1.24.4..."):
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.24.4"])
                    st.success("NumPy установлен! Перезагрузите страницу.")
                    st.rerun()
                except Exception as install_error:
                    st.error(f"Ошибка установки: {install_error}")
        
        return None, None

# 3. Функция для обработки изображения
def enhance_image(image, model, device):
    try:
        # Преобразования
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Подготовка входного тензора
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Обработка
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Конвертация обратно в изображение
        output_tensor = output_tensor.squeeze(0).cpu()
        output_img = output_tensor * 0.5 + 0.5
        output_img = torch.clamp(output_img, 0, 1)
        output_img = transforms.ToPILImage()(output_img)
        
        return output_img
        
    except Exception as e:
        st.error(f"Ошибка обработки изображения: {str(e)}")
        raise

# 4. Интерфейс Streamlit
st.set_page_config(
    page_title="Улучшение изображений",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Улучшение качества изображений")
st.markdown("---")

# Информация о системе
col_sys1, col_sys2, col_sys3 = st.columns(3)
with col_sys1:
    st.info(f"PyTorch: {torch.__version__}")
with col_sys2:
    try:
        import numpy as np
        st.info(f"NumPy: {np.__version__}")
    except:
        st.warning("NumPy: Не загружен")
with col_sys3:
    st.info(f"Устройство: {'GPU 🚀' if torch.cuda.is_available() else 'CPU ⚙️'}")

# Загрузка модели
with st.spinner("Загружаем модель..."):
    model, device = load_model()

if model is None:
    st.stop()

# Основной интерфейс
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 Загрузите изображение")
    
    uploaded_file = st.file_uploader(
        "Выберите изображение", 
        type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
        help="Поддерживаются форматы: PNG, JPG, JPEG, BMP, WEBP",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.success(f"✅ Изображение загружено: {image.size[0]}x{image.size[1]} пикселей")
        except Exception as e:
            st.error(f"❌ Ошибка открытия файла: {e}")
            image = None
    else:
        image = None
        st.info("👆 Загрузите изображение выше")

with col2:
    if image:
        st.subheader("👁️ Предпросмотр")
        # Показываем миниатюру
        max_size = 300
        if image.width > max_size or image.height > max_size:
            ratio = min(max_size / image.width, max_size / image.height)
            new_size = (int(image.width * ratio), int(image.height * ratio))
            preview = image.resize(new_size, Image.Resampling.LANCZOS)
            st.image(preview, caption=f"Предпросмотр ({new_size[0]}x{new_size[1]})")
        else:
            st.image(image, caption=f"Оригинал ({image.width}x{image.height})")

# Обработка
if image and model:
    st.markdown("---")
    
    if st.button("✨ УЛУЧШИТЬ КАЧЕСТВО", type="primary", use_container_width=True):
        with st.spinner("Обрабатываем изображение..."):
            try:
                # Прогресс бар
                progress_bar = st.progress(0)
                
                # Шаг 1: Подготовка
                progress_bar.progress(20)
                st.info("🔧 Подготавливаем изображение...")
                
                # Шаг 2: Обработка
                progress_bar.progress(60)
                st.info("🔄 Обрабатываем нейросетью...")
                enhanced = enhance_image(image, model, device)
                
                # Шаг 3: Завершение
                progress_bar.progress(100)
                st.success("✅ Обработка завершена!")
                
                # Результаты
                st.subheader("📊 Результаты")
                
                # Сравнение
                col_before, col_after = st.columns(2)
                
                with col_before:
                    st.image(image.resize((256, 256)), 
                            caption="Оригинал", 
                            use_column_width=True)
                
                with col_after:
                    st.image(enhanced, 
                            caption="Улучшенная версия", 
                            use_column_width=True)
                
                # Скачивание
                st.subheader("💾 Скачать результат")
                
                format_col1, format_col2 = st.columns([2, 1])
                
                with format_col1:
                    format_option = st.selectbox(
                        "Формат",
                        ["PNG", "JPEG", "WEBP"]
                    )
                
                with format_col2:
                    quality = 95
                    if format_option != "PNG":
                        quality = st.slider("Качество", 1, 100, 95)
                
                # Подготовка файла
                buf = io.BytesIO()
                if format_option == "PNG":
                    enhanced.save(buf, format="PNG", optimize=True)
                    mime_type = "image/png"
                    file_ext = "png"
                elif format_option == "JPEG":
                    enhanced.save(buf, format="JPEG", quality=quality, optimize=True)
                    mime_type = "image/jpeg"
                    file_ext = "jpg"
                else:  # WEBP
                    enhanced.save(buf, format="WEBP", quality=quality)
                    mime_type = "image/webp"
                    file_ext = "webp"
                
                byte_im = buf.getvalue()
                
                # Кнопки скачивания
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                
                with col_dl1:
                    st.download_button(
                        label=f"📥 {format_option}",
                        data=byte_im,
                        file_name=f"enhanced.{file_ext}",
                        mime=mime_type,
                        use_container_width=True
                    )
                
                with col_dl2:
                    # Оригинал
                    buf_orig = io.BytesIO()
                    image.save(buf_orig, format="PNG")
                    st.download_button(
                        label="📥 Оригинал",
                        data=buf_orig.getvalue(),
                        file_name="original.png",
                        mime="image/png",
                        use_container_width=True
                    )
                
                with col_dl3:
                    # Оба изображения
                    if st.button("🔄 Обработать еще", use_container_width=True):
                        st.rerun()
                
            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")

# Информация
with st.expander("ℹ️ Информация о приложении"):
    st.markdown("""
    ### 📝 Описание
    Это приложение использует нейросеть для улучшения качества изображений.
    
    ### 🛠 Технические детали
    - **Модель**: StrongGenerator с остаточными блоками
    - **Архитектура**: 6 Residual Blocks, 128 каналов
    - **Входной размер**: 128×128 пикселей
    - **Форматы**: PNG, JPG, JPEG, BMP, WEBP
    
    ### ⚠️ Ограничения
    - Изображения автоматически изменяются до 128×128
    - Обработка на CPU может быть медленной
    - Рекомендуется использовать изображения до 5MB
    """)

# Футер
st.markdown("---")
st.caption("🎯 Улучшение качества изображений | PyTorch + Streamlit | [Папка models/]")

# Кнопка для решения проблем
if st.button("🔄 Перезагрузить приложение"):
    st.rerun()
