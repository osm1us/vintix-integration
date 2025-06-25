import os
from huggingface_hub import snapshot_download

# Создаем директорию для моделей, если ее нет
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# Путь для сохранения чекпоинта
checkpoint_path = os.path.join(models_dir, "vintix_checkpoint")

print(f"Загрузка модели в {checkpoint_path}...")

# Скачиваем модель из Hugging Face
snapshot_download(
    repo_id="dunnolab/Vintix",
    local_dir=checkpoint_path,
    local_dir_use_symlinks=False # Рекомендуется для Windows
)

print("Модель успешно загружена!")
