import os
import requests
import zipfile
import io
from huggingface_hub import snapshot_download

# Создаем директорию для моделей, если ее нет
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# --- 1. Загрузка модели Vintix ---
vintix_checkpoint_path = os.path.join(models_dir, "vintix_checkpoint")
if not os.path.exists(vintix_checkpoint_path):
    print(f"Загрузка модели Vintix в {vintix_checkpoint_path}...")
    snapshot_download(
        repo_id="dunnolab/Vintix",
        local_dir=vintix_checkpoint_path,
        local_dir_use_symlinks=False # Рекомендуется для Windows
    )
    print("Модель Vintix успешно загружена!")
else:
    print("Директория с моделью Vintix уже существует, загрузка пропущена.")


# --- 2. Загрузка модели Vosk для распознавания речи ---
vosk_model_dir = os.path.join(models_dir, "vosk")
vosk_model_path = os.path.join(vosk_model_dir, "vosk-model-small-ru-0.22")
vosk_model_url = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"

if not os.path.exists(vosk_model_path):
    print(f"\nЗагрузка модели Vosk из {vosk_model_url}...")
    
    try:
        # Создаем директорию для всех моделей Vosk
        if not os.path.exists(vosk_model_dir):
            os.makedirs(vosk_model_dir)
            
        # Скачиваем zip-архив в память
        response = requests.get(vosk_model_url, stream=True)
        response.raise_for_status()
        
        # Распаковываем архив
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(vosk_model_dir)
            
        print(f"Модель Vosk успешно загружена и распакована в {vosk_model_dir}")
        
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при скачивании модели Vosk: {e}")
    except zipfile.BadZipFile:
        print("Ошибка: скачанный файл не является корректным zip-архивом.")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
else:
    print("\nДиректория с моделью Vosk уже существует, загрузка пропущена.")

print("\nВсе модели готовы к работе.")
