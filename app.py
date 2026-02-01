import streamlit as st
import torch
from PIL import Image
import io
import os
from torchvision import transforms

# 1. Классы моделей (копируем из вашего кода)
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

# 2. Функция для загрузки модели
@st.cache_resource
def load_model():
    # Путь к файлу с весами (относительный путь)
    weights_path = "models/enhanced_epoch_28_ratio_1.23.pth"
    
    # Проверяем существование файла
    if not os.path.exists(weights_path):
        st.error(f"Файл с весами не найден по пути: {weights_path}")
        st.info("Пожалуйста, убедитесь что файл находится в папке models/")
        return None, None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        checkpoint = torch.load(weights_path, 
                               map_location=device, 
                               weights_only=False)
        model = StrongGenerator().to(device)
        model.load_state_dict(checkpoint['generator'])
        model.eval()
        st.success(f"Модель загружена успешно! Устройство: {'GPU' if device == 'cuda' else 'CPU'}")
        return model, device
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {str(e)}")
        st.info("Возможные причины:")
        st.info("1. Неправильный формат файла")
        st.info("2. Несоответствие структуры модели")
        st.info("3. Файл поврежден")
        return None, None

# 3. Функция для обработки изображения
def enhance_image(image, model, device):
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

# 4. Интерфейс Streamlit
st.set_page_config(
    page_title="Улучшение изображений",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Улучшение качества изображений")
st.write("Загрузите изображение для улучшения качества с помощью нейросети")

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Информация о модели
    st.subheader("Информация о модели")
    st.info(f"Устройство: {'GPU доступен' if torch.cuda.is_available() else 'Только CPU'}")
    
    # Размер изображения
    st.subheader("Параметры обработки")
    show_original_size = st.checkbox("Показать оригинальный размер", value=True)
    
    # Кэширование
    st.subheader("Производительность")
    use_cache = st.checkbox("Использовать кэширование", value=True)
    
    st.markdown("---")
    st.caption("Приложение использует архитектуру с остаточными блоками")

# Основная область
col1, col2 = st.columns([2, 1])

with col1:
    # Загрузка модели
    model, device = load_model()
    
    if model is None:
        st.stop()
    
    # Загрузка изображения
    st.subheader("📤 Загрузка изображения")
    
    # Два способа загрузки
    tab1, tab2 = st.tabs(["Файл", "URL"])
    
    image = None
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Выберите файл", 
            type=['png', 'jpg', 'jpeg', 'bmp', 'webp'],
            help="Поддерживаются форматы: PNG, JPG, JPEG, BMP, WEBP",
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert('RGB')
                st.success(f"Изображение загружено: {image.size[0]}x{image.size[1]} пикселей")
            except Exception as e:
                st.error(f"Ошибка открытия файла: {e}")
    
    with tab2:
        url = st.text_input("Введите URL изображения", placeholder="https://example.com/image.jpg")
        if url:
            try:
                import requests
                from io import BytesIO
                
                response = requests.get(url)
                image = Image.open(BytesIO(response.content)).convert('RGB')
                st.success(f"Изображение загружено: {image.size[0]}x{image.size[1]} пикселей")
            except Exception as e:
                st.error(f"Ошибка загрузки по URL: {e}")

with col2:
    if image is not None:
        st.subheader("👁️ Предпросмотр")
        if show_original_size:
            st.image(image, caption="Оригинальное изображение", use_column_width=True)
        else:
            # Показываем уменьшенную версию
            preview_size = min(300, image.size[0], image.size[1])
            st.image(image.resize((preview_size, preview_size)), 
                    caption=f"Предпросмотр ({preview_size}x{preview_size})")

# Обработка изображения
if image is not None:
    st.markdown("---")
    
    col_left, col_center, col_right = st.columns([1, 2, 1])
    
    with col_center:
        if st.button("✨ Улучшить качество", type="primary", use_container_width=True):
            with st.spinner("Обрабатываем изображение... Это может занять несколько секунд"):
                try:
                    # Улучшаем изображение
                    enhanced = enhance_image(image, model, device)
                    
                    # Показываем результаты
                    st.subheader("📊 Результаты")
                    
                    # Две колонки для сравнения
                    col_before, col_after = st.columns(2)
                    
                    with col_before:
                        st.image(image.resize((256, 256)), 
                                caption="Оригинал (128x128)", 
                                use_column_width=True)
                    
                    with col_after:
                        st.image(enhanced, 
                                caption="Улучшенная версия", 
                                use_column_width=True)
                    
                    # Статистика
                    st.subheader("📈 Статистика")
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    
                    with col_stat1:
                        st.metric("Размер оригинала", f"{image.size[0]}x{image.size[1]}")
                    
                    with col_stat2:
                        st.metric("Размер после обработки", "128x128")
                    
                    with col_stat3:
                        st.metric("Устройство", "GPU" if device == 'cuda' else "CPU")
                    
                    # Кнопка для скачивания результата
                    st.subheader("💾 Скачать результат")
                    
                    # Формат для скачивания
                    format_option = st.selectbox(
                        "Формат файла",
                        ["PNG", "JPEG", "BMP"]
                    )
                    
                    # Качество для JPEG
                    quality = 95
                    if format_option == "JPEG":
                        quality = st.slider("Качество JPEG", 1, 100, 95)
                    
                    # Конвертация в выбранный формат
                    buf = io.BytesIO()
                    if format_option == "PNG":
                        enhanced.save(buf, format="PNG", optimize=True)
                        mime_type = "image/png"
                        file_ext = "png"
                    elif format_option == "JPEG":
                        enhanced.save(buf, format="JPEG", quality=quality, optimize=True)
                        mime_type = "image/jpeg"
                        file_ext = "jpg"
                    else:  # BMP
                        enhanced.save(buf, format="BMP")
                        mime_type = "image/bmp"
                        file_ext = "bmp"
                    
                    byte_im = buf.getvalue()
                    
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        st.download_button(
                            label=f"Скачать как {format_option}",
                            data=byte_im,
                            file_name=f"enhanced_image.{file_ext}",
                            mime=mime_type,
                            use_container_width=True
                        )
                    
                    with col_dl2:
                        # Также можно скачать оригинал
                        buf_orig = io.BytesIO()
                        image.save(buf_orig, format="PNG")
                        st.download_button(
                            label="Скачать оригинал",
                            data=buf_orig.getvalue(),
                            file_name="original_image.png",
                            mime="image/png",
                            use_container_width=True
                        )
                    
                except Exception as e:
                    st.error(f"Ошибка при обработке: {str(e)}")
                    st.info("Попробуйте:")
                    st.info("1. Изображение меньшего размера")
                    st.info("2. Другой формат файла")
                    st.info("3. Проверить подключение к GPU")

# Информация для разработчика
with st.expander("🔧 Информация для разработчика"):
    st.code("""
Структура проекта:
your_project/
├── app.py              # Этот файл
├── models/
│   └── enhanced_epoch_28_ratio_1.23.pth
├── requirements.txt
└── README.md
""", language="bash")
    
    if model is not None:
        st.write("Параметры модели:")
        total_params = sum(p.numel() for p in model.parameters())
        st.write(f"Всего параметров: {total_params:,}")

# Футер
st.markdown("---")
st.caption("🎯 Улучшение качества изображений | PyTorch + Streamlit | Веса модели в папке models/")
