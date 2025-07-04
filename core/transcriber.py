import whisper
import json
from pathlib import Path
from logger import setup_logger
from config import WHISPER_MODEL

logger = setup_logger(__name__)

def transcribe_video(project_path: Path, video_filename: str):
    """Транскрибирует видеофайл, если ещё не была выполнена"""

    transcription_folder = project_path / "transcription"
    ready_flag = transcription_folder / "ready.txt"

    # Если флаг уже существует — пропускаем
    if ready_flag.exists():
        logger.info(f"Транскрипция уже выполнена ранее, пропускаем. ({ready_flag})")
        return

    input_video_path = project_path / "input" / video_filename

    logger.info(f"Загрузка модели Whisper: {WHISPER_MODEL}")
    model = whisper.load_model(WHISPER_MODEL)

    logger.info(f"Транскрипция видео: {input_video_path}")
    result = model.transcribe(str(input_video_path))

    # Сохраняем текст в .txt
    transcription_text_path = transcription_folder / "full_transcription.txt"
    with open(transcription_text_path, "w", encoding="utf-8") as f_txt:
        for segment in result["segments"]:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            f_txt.write(f"[{start_time:.2f} - {end_time:.2f}] {text}\n")

    logger.info(f"Текстовая транскрипция сохранена в {transcription_text_path}")

    # Сохраняем данные сегментов в .json
    transcription_json_path = transcription_folder / "full_transcription.json"
    with open(transcription_json_path, "w", encoding="utf-8") as f_json:
        json.dump(result["segments"], f_json, indent=4, ensure_ascii=False)

    logger.info(f"JSON транскрипция сохранена в {transcription_json_path}")

    # Сохраняем флаг завершения
    ready_flag.write_text("ready")
    logger.info(f"Файл-флаг транскрипции создан: {ready_flag}")
