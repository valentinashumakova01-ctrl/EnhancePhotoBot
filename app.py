# app.py
import streamlit as st
import torch
from PIL import Image
import io
import os
import numpy as np
from torchvision import transforms

# Должно быть ПЕРВОЙ командой
st.set_page_config(
    page_title="Улучшение изображений",
    page_icon="🖼️",
    layout="wide"
)

st.title("🖼️ Улучшение качества изображений")
st.write("Загрузите изображение для обработки нейросетью")

# 1. Определяем классы модели (ТОЧНО ТАК ЖЕ КАК В КОЛАБЕ)
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

# 2. Функция загрузки модели
@st.cache_resource
def load_model():
    model_path = "models/enhanced_epoch_28_ratio_1.23.pth"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Файл модели не найден: {model_path}")
        return None
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        st.info(f"Используется устройство: {device}")
        
        # Загружаем модель (способ для PyTorch 2.6+)
        import torch.serialization
        
        # Разрешаем загрузку numpy объектов
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        
        # Загружаем с weights_only=False
        checkpoint = torch.load(
            model_path, 
            map_location=device,
            weights_only=False
        )
        
        # Создаем модель
        model = StrongGenerator().to(device)
        
        # Загружаем веса
        if 'generator' in checkpoint:
            model.load_state_dict(checkpoint['generator'])
            st.success("✅ Модель загружена (ключ 'generator')")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            st.success("✅ Модель загружена (ключ 'model_state_dict')")
        else:
            # Пробуем загрузить напрямую
            model.load_state_dict(checkpoint)
            st.success("✅ Модель загружена (прямая загрузка)")
        
        model.eval()
        return model, device
        
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {str(e)}")
        return None, None

# 3. Функция обработки изображения
def enhance_image(image, model, device):
    # ТОЧНО ТАКИЕ ЖЕ ПРЕОБРАЗОВАНИЯ КАК В КОЛАБЕ
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Подготовка тензора
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Обработка моделью
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Конвертация обратно в изображение
    output_tensor = output_tensor.squeeze(0).cpu()
    output_img = output_tensor * 0.5 + 0.5  # Денормализация
    output_img = torch.clamp(output_img, 0, 1)
    output_img = transforms.ToPILImage()(output_img)
    
    return output_img

# 4. Загружаем модель при запуске
st.markdown("---")
with st.spinner("Загружаем модель нейросети..."):
    model, device = load_model()

if model is None:
    st.error("Не удалось загрузить модель. Проверьте файл models/enhanced_epoch_28_ratio_1.23.pth")
    st.stop()

# 5. Интерфейс загрузки изображения
uploaded_file = st.file_uploader(
    "Выберите изображение для улучшения качества", 
    type=['png', 'jpg', 'jpeg', 'bmp']
)

if uploaded_file:
    try:
        # Открываем изображение
        image = Image.open(uploaded_file).convert('RGB')
        
        # Показываем оригинал
        st.subheader("📷 Оригинальное изображение")
        col1, col2 = st.columns(2)
        
        with col1:
            # Показываем в оригинальном размере
            st.image(image, caption=f"Размер: {image.size[0]}×{image.size[1]}", use_column_width=True)
        
        with col2:
            # Показываем уменьшенную до 128x128 версию (как будет подаваться в модель)
            preview_128 = image.resize((128, 128))
            st.image(preview_128, caption="Как будет подаваться в модель (128×128)", use_column_width=True)
        
        # Кнопка обработки
        st.markdown("---")
        if st.button("✨ УЛУЧШИТЬ КАЧЕСТВО С ПОМОЩЬЮ НЕЙРОСЕТИ", type="primary", use_container_width=True):
            with st.spinner("Нейросеть обрабатывает изображение..."):
                try:
                    # Обрабатываем моделью
                    enhanced = enhance_image(image, model, device)
                    
                    # Показываем результаты
                    st.subheader("🎯 Результат улучшения")
                    
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        st.markdown("### До улучшения (128×128)")
                        original_128 = image.resize((128, 128))
                        st.image(original_128, use_column_width=True)
                    
                    with result_col2:
                        st.markdown("### После улучшения (128×128)")
                        st.image(enhanced, use_column_width=True)
                    
                    # Сравнение деталей
                    st.markdown("---")
                    st.subheader("🔍 Сравнение деталей")
                    
                    # Берем небольшой фрагмент для сравнения
                    detail_size = 64
                    original_detail = original_128.crop((32, 32, 32+detail_size, 32+detail_size))
                    enhanced_detail = enhanced.crop((32, 32, 32+detail_size, 32+detail_size))
                    
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.image(original_detail.resize((256, 256)), 
                                caption="Фрагмент оригинала (увеличен 4×)", 
                                use_column_width=True)
                    
                    with detail_col2:
                        st.image(enhanced_detail.resize((256, 256)), 
                                caption="Фрагмент после улучшения (увеличен 4×)", 
                                use_column_width=True)
                    
                    # Скачивание
                    st.markdown("---")
                    st.subheader("💾 Скачать улучшенное изображение")
                    
                    buf = io.BytesIO()
                    enhanced.save(buf, format="PNG", optimize=True)
                    
                    st.download_button(
                        "📥 Скачать улучшенное изображение (128×128 PNG)",
                        buf.getvalue(),
                        "enhanced_image_128x128.png",
                        "image/png",
                        use_container_width=True
                    )
                    
                    # Также можно скачать в большем размере
                    st.info("💡 Совет: Для лучшего качества вы можете увеличить улучшенное изображение в графическом редакторе")
                    
                except Exception as e:
                    st.error(f"❌ Ошибка при обработке: {str(e)}")
                    st.info("Попробуйте другое изображение или проверьте модель")
    
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке изображения: {str(e)}")
else:
    st.info("👆 Загрузите изображение выше для улучшения качества")

# Техническая информация
with st.expander("🔧 Техническая информация"):
    st.write(f"""
    ### Информация о системе:
    - **PyTorch версия**: {torch.__version__}
    - **Устройство**: {'GPU 🚀' if torch.cuda.is_available() else 'CPU ⚙️'}
    - **Модель**: StrongGenerator с 6 остаточными блоками
    - **Размер обработки**: 128×128 пикселей
    
    ### Архитектура модели:
    ```python
    class StrongGenerator:
        - 1 начальный слой Conv2d(3, 128)
        - 6 остаточных блоков (ResidualBlock)
        - 2 финальных слоя Conv2d
        - Skip connection с коэффициентом 0.3
    ```
    
    ### Преобразования:
    1. Изменение размера до 128×128
    2. Конвертация в тензор
    3. Нормализация: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    4. Обработка нейросетью
    5. Денормализация и конвертация обратно в PIL Image
    """)

st.markdown("---")
st.caption("Нейросеть для улучшения качества изображений | PyTorch + Streamlit")
