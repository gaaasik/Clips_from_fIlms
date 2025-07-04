from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.VideoClip import VideoClip
from pathlib import Path
import numpy as np
import json
import subprocess
from datetime import datetime
from openai import OpenAI
from config import OPENAI_API_KEY, OPENAI_MODEL
import tempfile
from logger import setup_logger

import logging
from logger import setup_logger

logger = setup_logger("clip_enhancer")
logger.setLevel(logging.DEBUG)        # ← добавьте эту строку

def generate_ai_metadata(project_path: Path):
    logger.info("📡 Генерация метаданных по транскрипции")
    client = OpenAI(api_key=OPENAI_API_KEY)

    transcript_path = project_path / "transcription" / "full_transcription.json"
    with open(transcript_path, "r", encoding="utf-8") as f:
        segments = json.load(f)

    selected_text = ""
    max_time = 180
    for seg in segments:
        if seg["end"] > max_time:
            break
        selected_text += seg["text"].strip() + " "

    prompt = (
        "Ты специалист по продвижению коротких видео. На основе этой расшифровки придумай метаданные, "
        "которые помогут клипу попасть в рекомендации TikTok, YouTube Shorts и Instagram Reels.\n"
        "Ответ строго в формате JSON:\n"
        "{\n"
        "  \"title\": \"...\",\n"
        "  \"description\": \"...\",\n"
        "  \"hashtags\": [\"#tag1\", \"#tag2\", ...],\n"
        "  \"genre\": \"...\",\n"
        "  \"platforms\": [\"tiktok\", \"reels\", \"shorts\"],\n"
        "  \"audience\": \"...\",\n"
        "  \"content_type\": \"...\",\n"
        "  \"description_of_scene\": \"...\"\n"
        "}\n"
        f"Текст:\n{selected_text.strip()}\n"
    )

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=700
    )

    raw_reply = response.choices[0].message.content
    try:
        metadata = json.loads(raw_reply.replace("'", '"'))
        metadata["clip_duration"] = round(segments[-1]["end"], 2)
        metadata["creation_time"] = datetime.now().isoformat()
    except Exception as e:
        metadata = {
            "error": "json_parse_failed",
            "raw_reply": raw_reply,
            "exception": str(e),
            "prompt_used": prompt
        }

    output_file = project_path / "ready_to_post" / "meta_ai.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
    logger.info(f"✅ AI metadata saved to: {output_file}")

def remove_green_screen(clip, color=(0, 255, 0), threshold=60):
    logger.info("🎨 Удаление зелёного фона")
    def make_mask_frame(t):
        frame = clip.get_frame(t)
        diff = np.linalg.norm(frame - color, axis=2)
        mask = (diff > threshold).astype(np.float32)
        return mask
    mask = VideoClip(make_mask_frame, duration=clip.duration).with_duration(clip.duration)
    mask.fps = clip.fps
    return clip.with_mask(mask)

def load_transcription(project_path: Path):
    path = project_path / "transcription" / "full_transcription.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
def create_subtitles(
        transcript,
        speed_multiplier: float = 1.15,
        font_size: int = 36,
        color: str = "white",
        font_path: str = "C:/Windows/Fonts/arial.ttf"   # укажите свой путь, если шрифт не Windows
):
    """
    Генерирует список TextClip-ов с учётом новой сигнатуры MoviePy-2.x.

    :param transcript: список словарей {"text": str, "start": float, "end": float}
    :param speed_multiplier: во-сколько раз вы ускоряете ролик (из enhance_clip)
    :param font_size: размер шрифта
    :param color: цвет текста
    :param font_path: путь к .ttf / .otf; если None — будет системный шрифт Pillow
    """
    subtitle_clips = []

    for seg in transcript:
        text  = seg["text"].strip()
        start = seg["start"] / speed_multiplier
        end   = seg["end"]   / speed_multiplier

        try:
            txt_clip = (
                TextClip(
                    font=font_path,          # ⬅ 1-й позиционный аргумент теперь font!
                    text=text,               # текст передаём именованным
                    font_size=font_size,     # fontsize → font_size
                    color=color,
                    method="caption",
                    size=(1000, None)        # ширина 1000px, высота автоподбор
                )
                .with_start(start)
                .with_end(end)
                .with_position(("center", "bottom"))
            )
            subtitle_clips.append(txt_clip)

        except Exception as e:
            logger.error(f"❌ Subtitle failed: {text[:30]} — {e}")

    return subtitle_clips



def embed_metadata(infile: Path, outfile: Path, metadata: dict):
    cmd = ["ffmpeg", "-y", "-i", str(infile)]
    for key, value in metadata.items():
        cmd += ["-metadata", f"{key}={value}"]
    cmd += ["-c", "copy", str(outfile)]
    subprocess.run(cmd, check=True)

def speed_up(clip, factor):
    new_clip = clip.time_transform(lambda t: t / factor).with_duration(clip.duration / factor)
    if clip.audio:
        new_audio = clip.audio.time_transform(lambda t: t / factor).with_duration(clip.audio.duration / factor)
        new_clip = new_clip.with_audio(new_audio)
    return new_clip

def enhance_all_clips(project_path: Path, overlay_path: Path):
    clips_folder = project_path / "clips"
    for clip_path in sorted(clips_folder.glob("*.mp4")):
        enhance_clip(project_path, clip_path, overlay_path, title_hint=clip_path.stem)

def enhance_clip(project_path: Path, clip_path: Path,
                 overlay_path: Path, title_hint: str = "clip"):

    output_folder = project_path / "ready_to_post"
    output_folder.mkdir(exist_ok=True)

    existing = list(output_folder.glob("final_*.mp4"))
    next_id  = len(existing) + 1
    base_filename = f"final_{next_id:02d}_{title_hint}"
    temp_path  = output_folder / f"{base_filename}_temp.mp4"
    final_path = output_folder / f"{base_filename}.mp4"

    # ── 1. загрузка и ускорение ─────────────────────────────────────────────────
    logger.debug("📂 loading source %s", clip_path)
    base   = VideoFileClip(str(clip_path))
    speed  = 1.15
    logger.debug("⚡ speeding ×%.2f", speed)
    base   = speed_up(base, speed)

    # ── 2. подготовка канвы ─────────────────────────────────────────────────────
    W, H = 1080, 1920
    base_resized = (
        base.resized(width=W)
            .with_position(("center", "center"))
            .with_background_color(color=(0, 0, 0), size=(W, H))
    )
    main_clip = base_resized.with_start(3)

    # ── 3. оверлей ──────────────────────────────────────────────────────────────
    overlay = (
        VideoFileClip(str(overlay_path))
            .resized(width=400)
            .with_position(("center", H - 400))
    )
    overlay = remove_green_screen(overlay).with_start(3)

    # ── 4. субтитры ─────────────────────────────────────────────────────────────
    transcript = load_transcription(project_path)
    subtitles  = create_subtitles(transcript, speed_multiplier=speed)

    # ── 5. титульный экран ─────────────────────────────────────────────────────
    title_clip = (
        TextClip(
            font="C:/Windows/Fonts/arialbd.ttf",
            text=title_hint.replace("_", " "),
            font_size=70,
            color="white",
            method="caption",
            size=(W, None)
        )
        .with_duration(3)
        .with_position("center")
        .with_start(0)           # ← раньше было .set_start(0)
    )

    # ── 6. сборка слоёв ─────────────────────────────────────────────────────────
    logger.debug("🖇 assembling layers")
    layers = [title_clip, main_clip, overlay] + subtitles
    total_duration = base_resized.duration + 3
    full_video = CompositeVideoClip(layers).with_duration(total_duration)
    logger.debug("✅ composite ready, duration %.2f", total_duration)

    # ── 7. рендер ───────────────────────────────────────────────────────────────
    logger.debug("💾 writing temp video %s", temp_path)
    full_video.write_videofile(
        str(temp_path),
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(temp_path.with_suffix(".m4a")),
        remove_temp=True,
        logger=None
    )

    # ── 8. метаданные ───────────────────────────────────────────────────────────
    metadata = {
        "title": f"{title_hint.title()} Clip",
        "author": "pingivpn",
        "comment": "Auto-generated viral clip.",
        "software": "pingivpn-autocutter v1.0",
        "genre": "short",
        "description": f"Clip from {title_hint.replace('_', ' ')} for viral social posting",
        "creation_time": datetime.now().isoformat()
    }
    logger.debug("📝 embedding metadata")
    embed_metadata(temp_path, final_path, metadata)

    # ── 9. JSON-описание ────────────────────────────────────────────────────────
    meta_json_path = output_folder / f"{base_filename}.json"
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump({**metadata,
                   "filename": final_path.name,
                   "duration": round(total_duration, 2)},
                  f, indent=4, ensure_ascii=False)

    logger.info("🎉 готово: %s", final_path)
