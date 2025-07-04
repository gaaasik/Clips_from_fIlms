import shutil
from pathlib import Path
from logger import setup_logger
from config import PROJECTS_FOLDER

logger = setup_logger(__name__)

def create_project_structure(video_path: str) -> Path:
    """Создает структуру папок для одного видео"""
    video_name = Path(video_path).stem
    project_path = PROJECTS_FOLDER / video_name

    if project_path.exists():
        logger.warning(f"Проект '{video_name}' уже существует! Пропускаем создание структуры.")
        return project_path

    logger.info(f"Создание проекта для видео: {video_name}")

    (project_path / "input").mkdir(parents=True, exist_ok=True)
    (project_path / "transcription").mkdir(exist_ok=True)
    (project_path / "analysis").mkdir(exist_ok=True)
    (project_path / "timecodes").mkdir(exist_ok=True)
    (project_path / "clips").mkdir(exist_ok=True)

    logger.info(f"Структура папок создана: {project_path}")

    # Копируем оригинальный файл в input
    input_video_path = project_path / "input" / Path(video_path).name
    shutil.copy(video_path, input_video_path)
    logger.info(f"Видео скопировано в {input_video_path}")

    return project_path
