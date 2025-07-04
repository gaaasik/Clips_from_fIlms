import os

def find_first_video_file(folder_path: str) -> str:
    """Ищет первый mp4 файл в папке"""
    for file in os.listdir(folder_path):
        if file.endswith(".mp4"):
            return os.path.join(folder_path, file)
    raise FileNotFoundError(f"Не найдено видеофайлов в папке {folder_path}")
