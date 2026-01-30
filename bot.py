import os
import logging
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
import asyncio
from datetime import datetime

# Загрузка переменных окружения
load_dotenv()

import telebot
from PIL import Image
import io
import cv2
import numpy as np
import tempfile

# Конфигурация
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
ADMIN_IDS = [int(id.strip()) for id in os.getenv('ADMIN_IDS', '').split(',') if id.strip()]
MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', 20 * 1024 * 1024))  # 20MB по умолчанию

# Проверка токена
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не найден в .env файле")

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Инициализация бота
bot = telebot.TeleBot(TELEGRAM_TOKEN, parse_mode='HTML')

# Импорт утилит из проекта
try:
    sys.path.append(str(Path(__file__).parent))
    from utils.image_processing import load_image, save_image
    from utils.model_loader import enhance_image, classify_image
    from config.settings import IMAGE_SETTINGS
    logger.info("Утилиты импортированы")
except ImportError as e:
    logger.warning(f"Не удалось импортировать утилиты: {e}")
    # Запасные функции
    def load_image(image_bytes):
        return Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    def save_image(image, format='PNG'):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format)
        return img_byte_arr.getvalue()
    
    def enhance_image(image, enhancement_type='auto'):
        # Простое улучшение как fallback
        img_array = np.array(image)
        
        # Увеличение разрешения
        scale = 2
        h, w = img_array.shape[:2]
        enhanced = cv2.resize(img_array, (w*scale, h*scale), 
                             interpolation=cv2.INTER_CUBIC)
        
        # Улучшение резкости
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return Image.fromarray(enhanced)

class PhotoEnhancerBot:
    def __init__(self):
        self.user_sessions = {}  # Хранение сессий пользователей
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        
    def check_file_size(self, file_size: int) -> bool:
        """Проверка размера файла"""
        return file_size <= MAX_FILE_SIZE
    
    def get_file_extension(self, filename: str) -> str:
        """Получение расширения файла"""
        return Path(filename).suffix.lower()
    
    def is_supported_format(self, filename: str) -> bool:
        """Проверка формата файла"""
        ext = self.get_file_extension(filename)
        return ext in self.supported_formats

enhancer_bot = PhotoEnhancerBot()

def log_user_action(user_id: int, username: str, action: str):
    """Логирование действий пользователя"""
    logger.info(f"User {user_id} ({username}): {action}")

def create_keyboard():
    """Создание клавиатуры с кнопками"""
    from telebot import types
    
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    
    btn1 = types.KeyboardButton('🖼️ Улучшить качество')
    btn2 = types.KeyboardButton('📋 Инструкция')
    
    markup.add(btn1, btn2)
    return markup

def create_enhancement_keyboard():
    """Клавиатура для выбора типа улучшения"""
    from telebot import types
    
    markup = types.InlineKeyboardMarkup(row_width=2)
    
    buttons = [
        types.InlineKeyboardButton('🎭 Портрет', callback_data='enhance_portrait'),
        types.InlineKeyboardButton('🌄 Пейзаж', callback_data='enhance_landscape')
    ]
    
    markup.add(*buttons)
    return markup

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    """Обработчик команд /start и /help"""
    user_id = message.from_user.id
    username = message.from_user.username or message.from_user.first_name
    
    log_user_action(user_id, username, "started the bot")
    
    welcome_text = f"""
👋 Привет, {message.from_user.first_name}!

Я - бот для улучшения качества фотографий с помощью нейросетей.

✨ <b>Что я умею:</b>
• Улучшать качество фотографий
• Увеличивать разрешение в 2-4 раза
• Улучшать детализацию

📤 <b>Как использовать:</b>
1. Отправьте мне фотографию
2. Выберите тип улучшения
3. Получите результат!

🎯 <b>Поддерживаемые форматы:</b> JPG, PNG, BMP, WebP
📏 <b>Макс. размер:</b> 80MB

<b>Команды:</b>
/start - начать работу
/help - помощь

Просто отправьте мне фотографию!
    """
    
    bot.reply_to(message, welcome_text, reply_markup=create_keyboard())

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
    """Обработка фотографий, отправленных напрямую"""
    user_id = message.from_user.id
    username = message.from_user.username or message.from_user.first_name
    
    log_user_action(user_id, username, "sent a photo")
    
    try:
        # Отправляем статус обработки
        status_msg = bot.reply_to(message, "⏳ Загружаю фотографию...")
        
        # Получаем фото наилучшего качества
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)
        
        # Проверяем размер файла
        if not enhancer_bot.check_file_size(file_info.file_size or 0):
            bot.edit_message_text(
                "❌ Файл слишком большой. Максимальный размер: 80MB",
                chat_id=message.chat.id,
                message_id=status_msg.message_id
            )
            return
        
        # Скачиваем файл
        downloaded_file = bot.download_file(file_info.file_path)
        
        bot.edit_message_text(
            "🔍 Анализирую изображение...",
            chat_id=message.chat.id,
            message_id=status_msg.message_id
        )
        
        # Загружаем изображение
        image = load_image(downloaded_file)
        
        # Определяем тип улучшения
        bot.edit_message_text(
            "🎯 Выберите тип улучшения:",
            chat_id=message.chat.id,
            message_id=status_msg.message_id,
            reply_markup=create_enhancement_keyboard()
        )
        
        # Сохраняем изображение во временную сессию
        enhancer_bot.user_sessions[user_id] = {
            'image': image,
            'image_bytes': downloaded_file,
            'message_id': status_msg.message_id
        }
        
    except Exception as e:
        logger.error(f"Ошибка при обработке фото: {e}")
        error_msg = f"❌ Ошибка при обработке: {str(e)}"
        
        if 'status_msg' in locals():
            bot.edit_message_text(
                error_msg,
                chat_id=message.chat.id,
                message_id=status_msg.message_id
            )
        else:
            bot.reply_to(message, error_msg)

@bot.message_handler(content_types=['document'])
def handle_document(message):
    """Обработка документов (изображений)"""
    user_id = message.from_user.id
    username = message.from_user.username or message.from_user.first_name
    
    try:
        # Проверяем, что это изображение
        if not message.document.mime_type.startswith('image/'):
            bot.reply_to(message, "❌ Пожалуйста, отправьте изображение (JPG, PNG, etc.)")
            return
        
        log_user_action(user_id, username, f"sent a document: {message.document.file_name}")
        
        # Проверяем формат файла
        if not enhancer_bot.is_supported_format(message.document.file_name):
            bot.reply_to(message, f"❌ Неподдерживаемый формат файла. Используйте: {', '.join(enhancer_bot.supported_formats)}")
            return
        
        # Проверяем размер файла
        if not enhancer_bot.check_file_size(message.document.file_size):
            bot.reply_to(message, "❌ Файл слишком большой. Максимальный размер: 20MB")
            return
        
        status_msg = bot.reply_to(message, "⏳ Загружаю файл...")
        
        # Скачиваем файл
        file_info = bot.get_file(message.document.file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        
        bot.edit_message_text(
            "🔍 Анализирую изображение...",
            chat_id=message.chat.id,
            message_id=status_msg.message_id
        )
        
        # Загружаем изображение
        image = load_image(downloaded_file)
        
        bot.edit_message_text(
            "🎯 Выберите тип улучшения:",
            chat_id=message.chat.id,
            message_id=status_msg.message_id,
            reply_markup=create_enhancement_keyboard()
        )
        
        # Сохраняем в сессию
        enhancer_bot.user_sessions[user_id] = {
            'image': image,
            'image_bytes': downloaded_file,
            'message_id': status_msg.message_id,
            'filename': message.document.file_name
        }
        
    except Exception as e:
        logger.error(f"Ошибка при обработке документа: {e}")
        error_msg = f"❌ Ошибка при обработке: {str(e)}"
        
        if 'status_msg' in locals():
            bot.edit_message_text(
                error_msg,
                chat_id=message.chat.id,
                message_id=status_msg.message_id
            )
        else:
            bot.reply_to(message, error_msg)

@bot.callback_query_handler(func=lambda call: True)
def handle_callback(call):
    """Обработка callback-запросов от кнопок"""
    user_id = call.from_user.id
    
    try:
        if call.data.startswith('enhance_'):
            # Извлекаем тип улучшения
            enhance_type = call.data.replace('enhance_', '')
            
            if enhance_type == 'portrait':
                enhancement_type = 'портрет'
            else: 
                enhance_type == 'landscape'
            
            # Получаем изображение из сессии
            if user_id not in enhancer_bot.user_sessions:
                bot.answer_callback_query(call.id, "❌ Сессия истекла. Отправьте фото заново.")
                return
            
            session = enhancer_bot.user_sessions[user_id]
            image = session['image']
            
            # Обновляем статус
            bot.edit_message_text(
                "⚙️ Улучшаю качество... Это может занять некоторое время.",
                chat_id=call.message.chat.id,
                message_id=call.message.message_id
            )
            
            # Улучшаем изображение
            enhanced_image = enhance_image(image, enhancement_type)
            
            # Подготавливаем результат
            bot.edit_message_text(
                "📤 Отправляю результат...",
                chat_id=call.message.chat.id,
                message_id=call.message.message_id
            )
            
            # Сохраняем изображение
            enhanced_bytes = save_image(enhanced_image, 'PNG')
            
            # Отправляем результат
            bot.send_photo(
                call.message.chat.id,
                enhanced_bytes,
                caption=f"✅ Улучшенное изображение\nТип улучшения: {enhancement_type}",
                reply_to_message_id=call.message.message_id
            )
            
            # Удаляем сессию
            del enhancer_bot.user_sessions[user_id]
            
            # Обновляем оригинальное сообщение
            bot.edit_message_text(
                "✅ Готово! Результат отправлен выше.",
                chat_id=call.message.chat.id,
                message_id=call.message.message_id
            )
            
        elif call.data.startswith('setting_'):
            # Обработка настроек
            setting = call.data.replace('setting_', '')
            bot.answer_callback_query(call.id, f"Настройка '{setting}' будет реализована позже")
            
    except Exception as e:
        logger.error(f"Ошибка в callback: {e}")
        bot.answer_callback_query(call.id, f"❌ Ошибка: {str(e)}")

@bot.message_handler(func=lambda message: True)
def handle_text(message):
    """Обработка текстовых сообщений"""
    text = message.text.strip().lower()
    
    if text == '🖼️ улучшить качество':
        bot.reply_to(message, "📸 Отправьте мне фотографию, и я улучшу её качество!")
    
    elif text == '📋 инструкция':
        instruction_text = """
📋 <b>Инструкция по использованию:</b>

1. <b>Отправьте фото:</b> Вы можете:
   • Напрямую отправить фото
   • Отправить как документ (для лучшего качества)
   
2. <b>Выберите тип улучшения:</b>
   • 🎭 <b>Портрет</b> - для лиц и людей
   • 🌄 <b>Пейзаж</b> - для природы и видов
   
3. <b>Получите результат:</b> Улучшенное фото будет отправлено вам

💡 <b>Советы:</b>
• Для лучшего качества отправляйте фото как документ
• Размер файла не должен превышать 80MB
• Поддерживаются форматы: JPG, PNG, BMP, WebP
        """
        bot.reply_to(message, instruction_text)
    
    elif text == '⚙️ настройки':
        send_settings(message)
    
    elif text == '📊 статистика':
        send_stats(message)
    
    else:
        bot.reply_to(
            message,
            "🤔 Не понял ваше сообщение. Используйте кнопки или отправьте фотографию."
        )

def main():
    """Основная функция запуска бота"""
    logger.info("Запуск бота для улучшения качества фотографий")
    logger.info(f"Токен: {'установлен' if TELEGRAM_TOKEN else 'НЕ УСТАНОВЛЕН!'}")
    logger.info(f"Админы: {ADMIN_IDS}")
    logger.info(f"Макс. размер файла: {MAX_FILE_SIZE / 1024 / 1024:.1f} MB")
    
    if not TELEGRAM_TOKEN:
        logger.error("Токен бота не найден! Добавьте TELEGRAM_BOT_TOKEN в .env файл")
        return
    
    try:
        # Получаем информацию о боте
        bot_info = bot.get_me()
        logger.info(f"Бот @{bot_info.username} успешно запущен!")
        logger.info(f"Имя бота: {bot_info.first_name}")
        
        # Запускаем бота
        logger.info("Ожидание сообщений...")
        bot.infinity_polling(timeout=60, long_polling_timeout=60)
        
    except telebot.apihelper.ApiException as e:
        logger.error(f"Ошибка Telegram API: {e}")
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Неожиданная ошибка: {e}")

if __name__ == '__main__':
    main()
