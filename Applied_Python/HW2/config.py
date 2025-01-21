import os
from dotenv import load_dotenv

# Загрузка переменных из .env файла
load_dotenv()

# Чтение токенов из переменной окружения
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEATHER_API_TOKEN = os.getenv("WEATHER_API_TOKEN")
TRANSLATOR_API_TOKEN = os.getenv("TRANSLATOR_API_TOKEN")
NUTRITION_API_TOKEN = os.getenv("NUTRITION_API_TOKEN")


if not BOT_TOKEN:
    raise ValueError("Переменная окружения BOT_TOKEN не установлена!")

if not WEATHER_API_TOKEN:
    raise ValueError("Переменная окружения WEATHER_API_TOKEN не установлена!")

if not TRANSLATOR_API_TOKEN:
    raise ValueError("Переменная окружения TRANSLATOR_API_TOKEN не установлена!")

if not NUTRITION_API_TOKEN:
    raise ValueError("Переменная окружения NUTRITION_API_TOKEN не установлена!")
