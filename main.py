from config import PROJECTS_FOLDER
from core.clip_enhancer import enhance_clip
from core.utils import find_first_video_file
from core.project_manager import create_project_structure
from core.transcriber import transcribe_video
from core.analyzer import analyze_video
from core.timecode_manager import collect_timecodes
from core.clipper import cut_clips
from logger import setup_logger
from pathlib import Path

logger = setup_logger("main")

def main():
    logger.info("Запуск системы автоматической обработки видео")

    try:
        # 1. Найти первое видео в папке projects/
        video_path = find_first_video_file(str(PROJECTS_FOLDER))
        video_filename = Path(video_path).name
    except FileNotFoundError:
        logger.error("Не найдено ни одного .mp4 видео в папке projects/")
        return

    # 2. Создать структуру проекта
    project_path = create_project_structure(video_path)

    # 3. Транскрипция (если нет ready.txt)
    transcribe_video(project_path, video_filename)

    # 4. Анализ текста через GPT (если нет ready.txt)
    analyze_video(project_path)

    # 5. Сбор финальных таймкодов (если нет ready.txt)
    collect_timecodes(project_path)

    # 6. Нарезка клипов (если нет ready.txt)
    cut_clips(project_path, video_filename)

    enhance_clip(
        project_path=Path("projects/Кухня _ Сезон 1 _ Серия 1"),
        clip_path=Path("projects/Кухня _ Сезон 1 _ Серия 1/clips/clip_01.mp4"),
        overlay_path=Path("templates/pingivpn_overlay.mp4"),
        title_hint="kitchen_s1e1"
    )
    logger.info("✅ Обработка видео завершена полностью!")

if __name__ == "__main__":
    main()
