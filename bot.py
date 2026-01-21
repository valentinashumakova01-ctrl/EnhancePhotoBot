import telebot
from PIL import Image
import io
import cv2
import numpy as np
import tempfile
import os

# Получите токен у @BotFather
TOKEN = "YOUR_BOT_TOKEN"
bot = telebot.TeleBot(TOKEN)

def enhance_photo(image_bytes):
    """Улучшение загруженного фото"""
    # Конвертируем bytes в PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Преобразуем в numpy array для OpenCV
    img_array = np.array(image)
    
    # Улучшение качества (базовый пример)
    # 1. Увеличение разрешения
    scale_factor = 2
    new_size = (img_array.shape[1] * scale_factor, img_array.shape[0] * scale_factor)
    enhanced = cv2.resize(img_array, new_size, interpolation=cv2.INTER_CUBIC)
    
    # 2. Улучшение резкости
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # 3. Улучшение цвета
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=0)
    
    # Конвертируем обратно в PIL Image
    result_image = Image.fromarray(enhanced)
    
    # Сохраняем в bytes
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format='PNG')
    
    return img_byte_arr.getvalue()

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, 
                 "👋 Привет! Отправь мне фотографию, и я улучшу её качество!\n"
                 "Просто отправь любое изображение как файл.")

@bot.message_handler(content_types=['photo', 'document'])
def handle_photo(message):
    try:
        bot.send_message(message.chat.id, "⏳ Обрабатываю изображение...")
        
        if message.photo:
            # Получаем фото максимального качества
            file_id = message.photo[-1].file_id
        elif message.document:
            if not message.document.mime_type.startswith('image/'):
                bot.reply_to(message, "Пожалуйста, отправьте изображение!")
                return
            file_id = message.document.file_id
        
        # Скачиваем файл
        file_info = bot.get_file(file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        # Улучшаем качество
        enhanced_photo = enhance_photo(downloaded_file)
        
        # Отправляем результат
        bot.send_photo(message.chat.id, enhanced_photo, 
                      caption="✅ Качество улучшено!")
        
    except Exception as e:
        bot.reply_to(message, f"❌ Ошибка: {str(e)}")

@bot.message_handler(func=lambda message: True)
def handle_text(message):
    bot.reply_to(message, "Отправьте мне фотографию для улучшения качества!")

if __name__ == "__main__":
    print("Бот запущен...")
    bot.polling(none_stop=True)
