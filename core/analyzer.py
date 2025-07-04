import json
import time
from pathlib import Path
from openai import OpenAI
from logger import setup_logger
from config import OPENAI_API_KEY, OPENAI_MODEL, BLOCK_DURATION_SECONDS

logger = setup_logger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)


def load_transcription(project_path: Path) -> list:
    transcription_file = project_path / "transcription" / "full_transcription.json"
    with open(transcription_file, "r", encoding="utf-8") as f:
        return json.load(f)


def split_segments_into_blocks(segments: list) -> list:
    blocks = []
    current_block = []
    current_start_time = None
    current_duration = 0

    for segment in segments:
        if current_start_time is None:
            current_start_time = segment["start"]

        current_block.append(segment)
        current_duration = segment["end"] - current_start_time

        if current_duration >= BLOCK_DURATION_SECONDS:
            blocks.append(current_block)
            current_block = []
            current_start_time = None
            current_duration = 0

    if current_block:
        blocks.append(current_block)

    logger.info(f"Разбито на {len(blocks)} блока(ов)")
    return blocks


def generate_block_text(block: list) -> str:
    return " ".join([segment['text'].strip() for segment in block]).strip()


def safe_openai_request(prompt: str, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "Ты опытный видео-редактор, помогающий находить лучшие моменты для нарезки коротких клипов."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Ошибка запроса в OpenAI: {e}. Попытка {attempt + 1} из {retries}")
            time.sleep(delay)
    raise RuntimeError("Ошибка при запросе OpenAI после всех попыток")


def analyze_block_with_gpt(block_text: str) -> dict:
    logger.info(f"Отправка блока в OpenAI ({len(block_text)} символов)")

    prompt = (
        f"Вот текст из фрагмента видео (примерно 2 минуты):\n\n"
        f"{block_text}\n\n"
        "Определи, есть ли в этом фрагменте вирусный, эмоциональный или интересный момент.\n"
        "Если есть, верни JSON в формате:\n"
        "{'interesting': true, 'start_offset_sec': 30, 'end_offset_sec': 90, 'comment': 'описание момента'}\n"
        "Если нет интересного момента, верни:\n"
        "{'interesting': false}\n"
    )

    reply_content = safe_openai_request(prompt)

    logger.debug(f"Ответ GPT: {reply_content}")
    try:
        return json.loads(reply_content.replace("'", "\""))
    except Exception as e:
        logger.error(f"Ошибка парсинга ответа от GPT: {e}")
        return {"interesting": False}


def save_block_analysis(project_path: Path, block_index: int, analysis_result: dict):
    analysis_folder = project_path / "analysis"
    analysis_file = analysis_folder / f"block_{block_index:02d}_analysis.json"
    with open(analysis_file, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=4, ensure_ascii=False)
    logger.info(f"Анализ блока сохранен: {analysis_file}")


def analyze_video(project_path: Path):
    analysis_folder = project_path / "analysis"
    ready_flag = analysis_folder / "ready.txt"
    if ready_flag.exists():
        logger.info("Анализ уже выполнен ранее, пропускаем.")
        return

    segments = load_transcription(project_path)
    blocks = split_segments_into_blocks(segments)

    for idx, block in enumerate(blocks):
        block_text = generate_block_text(block)
        if len(block_text) > 2500:
            block_text = block_text[:2500]

        analysis_result = analyze_block_with_gpt(block_text)
        save_block_analysis(project_path, idx, analysis_result)

    ready_flag.write_text("ready")
    logger.info(f"Анализ завершен. Файл-флаг создан: {ready_flag}")
