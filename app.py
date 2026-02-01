import streamlit as st
import torch
from PIL import Image
import io
import requests
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
    # URL весов модели (замените на ваш реальный URL)
    weights_url = "https://example.com/enhanced_epoch_28_ratio_1.23.pth"
    weights_path = "model_weights.pth"
    
    # Скачиваем файл весов (если еще не скачан)
    if not os.path.exists(weights_path):
        try:
            response = requests.get(weights_url)
            with open(weights_path, 'wb') as f:
                f.write(response.content)
            st.success("Модель загружена!")
        except:
            st.error("Не удалось загрузить модель. Проверьте URL весов.")
            return None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        model = StrongGenerator().to(device)
        model.load_state_dict(checkpoint['generator'])
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
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
st.title("🚀 Улучшение качества изображений")
st.write("Загрузите изображение для улучшения качества с помощью нейросети")

# Загрузка модели
model, device = load_model()

if model is None:
    st.stop()

# Загрузка изображения
uploaded_file = st.file_uploader(
    "Выберите изображение", 
    type=['png', 'jpg', 'jpeg', 'bmp'],
    help="Загрузите изображение в формате PNG, JPG или JPEG"
)

if uploaded_file is not None:
    # Открываем изображение
    image = Image.open(uploaded_file).convert('RGB')
    
    # Показываем оригинал
    st.subheader("Оригинальное изображение")
    st.image(image, caption=f"Размер: {image.size}", use_column_width=True)
    
    # Кнопка для обработки
    if st.button("Улучшить качество", type="primary"):
        with st.spinner("Обрабатываем изображение..."):
            try:
                # Улучшаем изображение
                enhanced = enhance_image(image, model, device)
                
                # Показываем результат
                st.subheader("Улучшенное изображение")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image.resize((256, 256)), caption="Оригинал (уменьшено)", use_column_width=True)
                
                with col2:
                    st.image(enhanced, caption="Улучшенная версия", use_column_width=True)
                
                # Кнопка для скачивания результата
                buf = io.BytesIO()
                enhanced.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Скачать улучшенное изображение",
                    data=byte_im,
                    file_name="enhanced_image.png",
                    mime="image/png"
                )
                
            except Exception as e:
                st.error(f"Ошибка при обработке: {e}")
else:
    st.info("👆 Пожалуйста, загрузите изображение для начала работы")

# Информация о модели
with st.expander("ℹ️ О модели"):
    st.write("""
    Эта модель использует архитектуру с остаточными блоками (Residual Blocks) 
    для улучшения качества изображений. Модель была обучена на датасете изображений.
    
    **Технические детали:**
    - Размер входного изображения: 128x128 пикселей
    - Архитектура: 6 остаточных блоков
    - Устройство обработки: {'GPU' if device == 'cuda' else 'CPU'}
    """)

# Футер
st.markdown("---")
st.caption("Приложение для улучшения качества изображений | Использует PyTorch и Streamlit")
