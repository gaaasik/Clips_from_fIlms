import json
from pathlib import Path
from logger import setup_logger

logger = setup_logger(__name__)

def collect_timecodes(project_path: Path):
    """Собирает интересные моменты из всех файлов анализа и сохраняет финальные таймкоды"""
    analysis_folder = project_path / "analysis"
    timecodes_folder = project_path / "timecodes"
    timecodes_folder.mkdir(exist_ok=True)

    ready_flag = timecodes_folder / "ready.txt"
    if ready_flag.exists():
        logger.info("Финальные таймкоды уже собраны ранее, пропускаем.")
        return

    final_timecodes = []
    block_duration = 120

    for idx, analysis_file in enumerate(sorted(analysis_folder.glob("block_*_analysis.json"))):
        with open(analysis_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("interesting"):
            start_offset = data.get("start_offset_sec", 0)
            end_offset = data.get("end_offset_sec", block_duration)

            absolute_start = idx * block_duration + start_offset
            absolute_end = idx * block_duration + end_offset

            final_timecodes.append({
                "start": absolute_start,
                "end": absolute_end,
                "comment": data.get("comment", "интересный момент")
            })

    if not final_timecodes:
        logger.warning("Не найдено интересных моментов для нарезки!")
        return

    output_file = timecodes_folder / "final_timecodes.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_timecodes, f, indent=4, ensure_ascii=False)

    ready_flag.write_text("ready")
    logger.info(f"Финальные таймкоды сохранены в {output_file}, флаг готовности установлен.")
