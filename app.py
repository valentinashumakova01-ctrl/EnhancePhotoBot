import streamlit as st
import torch
from PIL import Image
import io
import os
import numpy as np
from torchvision import transforms

st.set_page_config(
    page_title="Улучшение изображений",
    page_icon="🖼️",
    layout="wide"
)

st.title("EnhancePhotoBot")
st.title("🖼️ Улучшение качества изображений с нейросетью")
st.write("Загрузите изображение для обработки нейросетью")

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

import os
import torch
import streamlit as st

@st.cache_resource
def load_model():
    model_path = "weights/enhanced_epoch_30_ratio_1.23.pth"
    
    if not os.path.exists(model_path):
        st.error(f"Файл модели не найден: {model_path}")
        return None, None
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Загружаем чекпоинт
        try:
            checkpoint = torch.load(model_path, map_location=device)
            st.success("✅ Модель загружена")
        except Exception as e:
            st.error(f"❌ Ошибка загрузки модели: {e}")
            return None, None
        
        # Создаем модель
        model = StrongGenerator().to(device)
        
        try:
            if 'generator' in checkpoint:
                model.load_state_dict(checkpoint['generator'])
                st.success("✅ Веса загружены")
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                st.success("✅ Веса загружены")
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
                st.success("✅ Веса загружены")
            else:
                
                try:
                    model.load_state_dict(checkpoint)
                    st.success("✅ Веса загружены")
                except:
                    st.error("❌ Не удалось определить формат чекпоинта")
                    return None, None
        except Exception as e:
            st.error(f"❌ Ошибка загрузки весов: {e}")
            return None, None
        
        model.eval()
        return model, device
        
    except Exception as e:
        st.error(f"❌ Ошибка при создании модели: {e}")
        return None, None

# Функция обработки изображения нейросетью
def enhance_image_with_model(image, model, device):
    """Обработка изображения нейросетью (как в Colab)"""
    try:
        # Преобразование
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
        
        # Денормализация
        output_img = output_tensor * 0.5 + 0.5
        output_img = torch.clamp(output_img, 0, 1)
        
        # Конвертация в PIL Image
        output_img = transforms.ToPILImage()(output_img)
        
        return output_img
        
    except Exception as e:
        st.error(f"❌ Ошибка при обработке изображения: {e}")
        return None

# Загрузка модели
with st.spinner("🔄 Загружаем нейросеть..."):
    model, device = load_model()

if model is None:
    st.error("Не удалось загрузить модель. Проверьте файл модели.")
    st.stop()

# Интерфейс
uploaded_file = st.file_uploader(
    "Выберите изображение для улучшения", 
    type=['png', 'jpg', 'jpeg', 'bmp', 'webp']
)

if uploaded_file:
    try:
        # Открываем изображение
        image = Image.open(uploaded_file).convert('RGB')
        st.success(f"✅ Изображение загружено: {image.size[0]}×{image.size[1]} пикселей")
        
        # Показываем оригинал
        st.subheader("📷 Оригинальное изображение")
        
        col1, col2 = st.columns(2)
        with col1:
            # Полный размер
            st.image(image, caption="Полный размер", use_column_width=True)
        
        with col2:
            # Уменьшенный до 128x128 (как будет обрабатываться)
            preview_128 = image.resize((128, 128))
            st.image(preview_128, caption="Для обработки (128×128)", use_column_width=True)
        
        # Кнопка обработки
        if st.button("✨ ЗАПУСТИТЬ НЕЙРОСЕТЬ", type="primary", use_container_width=True):
            with st.spinner("Нейросеть улучшает качество изображения..."):
                # Обработка нейросетью
                enhanced = enhance_image_with_model(image, model, device)
                
                if enhanced is not None:
                    # Показываем результаты
                    st.success("✅ Обработка завершена!")
                    
                    # Сравнение
                    st.subheader("🎯 Сравнение до и после")
                    
                    # Создаем оригинал того же размера для сравнения
                    original_128 = image.resize((128, 128))
                    
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        st.markdown("### **ДО** обработки")
                        st.image(original_128, use_column_width=True)
                        st.caption("Исходное изображение 128×128")
                    
                    with comp_col2:
                        st.markdown("### **ПОСЛЕ** обработки")
                        st.image(enhanced, use_column_width=True)
                        st.caption("Улучшенное нейросетью 128×128")
                    
                    # Увеличенные фрагменты для сравнения деталей
                    st.markdown("---")
                    st.subheader("🔍 Сравнение деталей (увеличенные фрагменты)")
                    
                    # Берем центральный фрагмент
                    crop_size = 64
                    original_crop = original_128.crop(
                        (32, 32, 32 + crop_size, 32 + crop_size)
                    )
                    enhanced_crop = enhanced.crop(
                        (32, 32, 32 + crop_size, 32 + crop_size)
                    )
                    
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.image(
                            original_crop.resize((256, 256)), 
                            caption="Фрагмент оригинала (×4)",
                            use_column_width=True
                        )
                    
                    with detail_col2:
                        st.image(
                            enhanced_crop.resize((256, 256)), 
                            caption="Фрагмент после улучшения (×4)",
                            use_column_width=True
                        )
                    
                    # Скачивание
                    st.markdown("---")
                    st.subheader("💾 Скачать результат")
                    
                    buf = io.BytesIO()
                    enhanced.save(buf, format="PNG", optimize=True)
                    
                    col_dl1, col_dl2 = st.columns(2)
                    
                    with col_dl1:
                        st.download_button(
                            "📥 Скачать улучшенное (128×128)",
                            buf.getvalue(),
                            "enhanced_128x128.png",
                            "image/png",
                            use_container_width=True
                        )
                    
                    with col_dl2:
                        buf_original = io.BytesIO()
                        original_128.save(buf_original, format="PNG", optimize=True)
                        st.download_button(
                            "📥 Скачать оригинал (128×128)",
                            buf_original.getvalue(),
                            "original_128x128.png",
                            "image/png",
                            use_container_width=True
                        )
                    
                    # Информация о модели
                    st.markdown("---")
                    with st.expander("📊 Техническая информация о обработке"):
                        st.write(f"""
                        ### Параметры обработки:
                        
                        - **Модель**: StrongGenerator с 6 остаточными блоками
                        - **Архитектура**: Skip connection с коэффициентом 0.3
                        - **Размер входа/выхода**: 128×128 пикселей
                        - **Устройство**: {device.upper()}
                        - **PyTorch версия**: {torch.__version__}
                        
                        ### Преобразования:
                        1. Resize до 128×128
                        2. ToTensor()
                        3. Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                        4. Обработка нейросетью
                        5. Denormalize: output * 0.5 + 0.5
                        6. Clamp(0, 1)
                        7. Конвертация в PIL Image
                        """)
                
                else:
                    st.error("Не удалось обработать изображение")
    
    except Exception as e:
        st.error(f"❌ Ошибка: {e}")

else:
    st.info("👆 Загрузите изображение выше для улучшения качества нейросетью")

# Информация о приложении
with st.expander("ℹ️ О нейросети"):
    st.markdown("""
    ## 🧠 Как работает нейросеть?
    
    Эта нейросеть использует архитектуру **StrongGenerator** с **остаточными блоками (Residual Blocks)** для улучшения качества изображений:
    
    ### Архитектура модели:
    
    1. **Входной слой**: 
       - Conv2d(3, 128, kernel_size=3, padding=1) + ReLU активация
    
    2. **Остаточные блоки (6 блоков)**:
       - Каждый блок состоит из двух сверточных слоев с ReLU активацией между ними
       - Используется Residual Connection (остаточное соединение)
       - Сохраняет информацию исходных признаков
    
    3. **Выходной слой**:
       - Conv2d(128, 64, kernel_size=3, padding=1) + ReLU
       - Conv2d(64, 3, kernel_size=3, padding=1)
    
    4. **Skip Connection (глобальное)**:
       - Исходное изображение добавляется к выходу с коэффициентом 0.3
       - Формула: `output = input + 0.3 * network_output`
    
    ### Принцип работы:
    - Модель обучается находить разницу между изображениями низкого и высокого качества
    - Добавляет недостающие детали и улучшает резкость
    - Сохраняет общую структуру и цветовую гамму оригинала
    - Работает как "улучшающий фильтр", а не как полная перерисовка
    
    ### Технические детали модели:
    - **Тип модели**: Генераторная нейросеть (Generative Adversarial Network - GAN)
    - **Размер входных данных**: 128×128 пикселей
    - **Каналы**: RGB (3 канала)
    - **Обучение**: 30 эпох
    
    ### Особенности обработки:
    - Изображения нормализуются в диапазон [-1, 1]
    - Автоматическое определение устройства (GPU/CPU)
    - Сохранение баланса между улучшением качества и сохранением оригинального вида
    """)

st.markdown("---")
st.caption("Нейросеть для улучшения качества изображений | PyTorch 2.10.0 | Streamlit Cloud")
