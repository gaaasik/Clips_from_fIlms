import json
from pathlib import Path
from moviepy import VideoFileClip
from logger import setup_logger
logger = setup_logger(__name__)

def load_timecodes(project_path: Path) -> list:
    timecodes_file = project_path / "timecodes" / "final_timecodes.json"
    if not timecodes_file.exists():
        raise FileNotFoundError(f"Файл таймкодов не найден: {timecodes_file}")

    with open(timecodes_file, "r", encoding="utf-8") as f:
        timecodes = json.load(f)
    logger.info(f"Загружено {len(timecodes)} таймкодов для нарезки")
    return timecodes


def cut_clips(project_path: Path, video_filename: str):
    input_video_path = project_path / "input" / video_filename
    output_folder = project_path / "clips"
    ready_flag = output_folder / "ready.txt"

    if ready_flag.exists():
        logger.info("Клипы уже были нарезаны ранее, пропускаем.")
        return

    output_folder.mkdir(exist_ok=True)
    logger.info(f"Открытие видео для нарезки: {input_video_path}")
    video = VideoFileClip(str(input_video_path))

    timecodes = load_timecodes(project_path)

    for idx, tc in enumerate(timecodes):
        start = tc["start"]
        end = tc["end"]
        comment = tc.get("comment", "clip")

        if start >= end:
            logger.warning(f"Неверный таймкод: start {start} >= end {end}, пропускаем.")
            continue

        if start > video.duration:
            logger.warning(f"Таймкод {start} за пределами видео ({video.duration}), пропускаем.")
            continue

        end = min(end, video.duration)
        clip = video.subclipped(start, end)


        clip_filename = f"clip_{idx+1:02d}.mp4"
        output_path = output_folder / clip_filename

        logger.info(f"Создание клипа: {output_path} (с {start} до {end})")
        clip.write_videofile(str(output_path), codec="libx264", audio_codec="aac")

    ready_flag.write_text("ready")
    logger.info("Клипы успешно нарезаны. Файл-флаг создан.")
