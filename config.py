from pathlib import Path
from dotenv import load_dotenv
import os

# Загрузка переменных из .env
load_dotenv()

# Путь до папки проектов
PROJECTS_FOLDER = Path("projects")

# Название модели Whisper
WHISPER_MODEL = "small"  # или 'base'

# Настройки OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo")
BLOCK_DURATION_SECONDS = int(os.getenv("BLOCK_DURATION_SECONDS", 120))

# Настройки логирования
LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
