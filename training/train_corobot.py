import argparse
import os
import torch
import yaml
import h5py
import numpy as np
import glob
from datetime import datetime
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# Предполагаемые импорты из библиотеки Vintix
# Точные пути могут потребовать корректировки при реализации
from vintix.vintix.data.torch_dataloaders import MultiTaskMapDataset
from vintix.vintix.training.utils.train_utils import initialize_model, configure_optimizers
from vintix.vintix.nn.model import Vintix

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CorobotTrainer")


def compute_and_save_statistics(log_dir: str):
    """
    Вычисляет статистики (mean, std) по всем логам и сохраняет их.
    Это необходимо для нормализации данных при обучении.
    """
    logger.info(f"Вычисление статистик для логов в директории: {log_dir}")
    h5_files = glob.glob(os.path.join(log_dir, "*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"В директории {log_dir} не найдено HDF5 лог-файлов.")

    all_observations = []
    all_actions = []

    for file_path in tqdm(h5_files, desc="Обработка лог-файлов"):
        with h5py.File(file_path, 'r') as hf:
            if "episode_data" in hf:
                all_observations.append(hf["episode_data"]["proprio_observation"][:])
                all_actions.append(hf["episode_data"]["action"][:])

    if not all_observations:
        raise ValueError("Не удалось загрузить данные из HDF5 файлов.")
        
    full_obs_np = np.concatenate(all_observations, axis=0)
    full_act_np = np.concatenate(all_actions, axis=0)
    
    metadata = {
        "observation_dim": full_obs_np.shape[1],
        "action_dim": full_act_np.shape[1],
        "observation_mean": full_obs_np.mean(axis=0).tolist(),
        "observation_std": full_obs_np.std(axis=0).tolist(),
        "action_mean": full_act_np.mean(axis=0).tolist(),
        "action_std": full_act_np.std(axis=0).tolist(),
    }
    
    # Сохраняем метаданные в json, который ожидает FoundationMapDataset
    metadata_path = os.path.join(log_dir, 'training_logs.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Статистики вычислены и сохранены в {metadata_path}")
    return metadata


def train(args):
    """Основная функция для дообучения модели Vintix."""
    logger.info("--- Запуск дообучения модели Vintix для Corobot ---")

    # 1. Настройка устройства (GPU, если доступен, иначе CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Используемое устройство: {device}")

    # 2. Загрузка конфигурации датасета
    with open(args.dataset_config, 'r') as f:
        dataset_config = yaml.safe_load(f)
    logger.info(f"Конфигурация датасета '{args.dataset_config}' загружена.")
    
    task_info = list(dataset_config.values())[0]
    log_dir = task_info['path']

    # 3. Вычисление/загрузка статистик и создание датасета
    try:
        metadata = compute_and_save_statistics(log_dir)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Ошибка при обработке логов: {e}")
        logger.error("Обучение не может быть продолжено без данных. Запустите run_vintix.py для сбора логов.")
        return

    dataset = MultiTaskMapDataset(
        data_dir='.', # Корень проекта
        datasets_info={log_dir: task_info['group']},
        trajectory_len=args.context_len,
        trajectory_sparsity=1, 
        ep_sparsity=[task_info['episode_sparsity']],
    )
    
    if len(dataset) == 0:
        logger.error("Созданный датасет пуст. Возможно, в логах слишком мало шагов для формирования хотя бы одной траектории.")
        logger.error(f"Длина контекста: {args.context_len}. Убедитесь, что у вас есть эпизоды длиннее этого значения.")
        return
        
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 # Для Windows лучше ставить 0
    )
    logger.info(f"Датасет создан. Обнаружено {len(dataset)} возможных траекторий.")

    # 4. Инициализация модели для дообучения
    logger.info(f"Загрузка базовой модели из: {args.load_checkpoint}")
    
    # Создаем подобие конфиг-объекта для совместимости с функциями Vintix
    class SimpleConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    model_args = {k: v for k, v in args.__dict__.items() if k not in ['dataset_config', 'load_checkpoint', 'save_dir_base', 'save_dir']}
    model_args['load_ckpt'] = args.load_checkpoint
    
    # Параметры, которых нет в наших args, но которые нужны Vintix (берем из train.py)
    model_args.setdefault('action_emb_dim', 511)
    model_args.setdefault('observation_emb_dim', 511)
    model_args.setdefault('reward_emb_dim', 2)
    model_args.setdefault('hidden_dim', 1024)
    model_args.setdefault('transformer_depth', 20)
    model_args.setdefault('transformer_heads', 16)
    model_args.setdefault('attn_dropout', 0.0)
    model_args.setdefault('residual_dropout', 0.0)
    model_args.setdefault('normalize_qk', True)
    model_args.setdefault('bias', True)
    model_args.setdefault('parallel_residual', False)
    model_args.setdefault('shared_attention_norm', False)
    model_args.setdefault('norm_class', "LayerNorm")
    model_args.setdefault('mlp_class', "GptNeoxMLP")
    model_args.setdefault('intermediate_size', 4096)
    model_args.setdefault('inner_ep_pos_enc', False)
    model_args.setdefault('norm_acs', False)
    model_args.setdefault('norm_obs', True)

    vintix_config = SimpleConfig(**model_args)
    
    model, _ = initialize_model(vintix_config, metadata, device)
    model.to(device)

    # 5. Настройка оптимизатора
    logger.info("Настройка оптимизатора (AdamW)...")
    opt_config = SimpleConfig(optimizer="AdamW", weight_decay=1e-2, betas=(0.9, 0.99), lr=args.lr, clip_grad=None)
    optimizer = configure_optimizers(opt_config, model)

    # 6. Основной цикл обучения
    logger.info("--- Начало цикла обучения ---")
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{args.epochs}")
        for batch in pbar:
            # Перемещаем данные на нужное устройство
            for key in batch:
                batch[key] = batch[key].to(device)
            
            optimizer.zero_grad()
            _, loss = model(batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Эпоха {epoch + 1} завершена. Средняя потеря (loss): {avg_loss:.4f}")

    # 7. Сохранение дообученной модели
    logger.info("--- Обучение завершено ---")
    model.save_model(args.save_dir, metadata)
    logger.info(f"Дообученная модель успешно сохранена в: {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Дообучение модели Vintix для Corobot.")
    
    parser.add_argument(
        '--dataset_config', 
        type=str, 
        default='training/corobot_dataset_config.yaml',
        help='Путь к YAML файлу с конфигурацией датасета.'
    )
    parser.add_argument(
        '--load_checkpoint', 
        type=str, 
        default='models/vintix_checkpoint',
        help='Путь к базовой модели для начала дообучения.'
    )
    parser.add_argument(
        '--save_dir_base', 
        type=str, 
        default='models',
        help='Базовая директория для сохранения новой дообученной модели.'
    )
    parser.add_argument('--epochs', type=int, default=20, help='Количество эпох обучения.')
    parser.add_argument('--batch_size', type=int, default=2, help='Размер батча. Уменьшите, если не хватает памяти GPU.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Скорость обучения (learning rate).')
    parser.add_argument('--context_len', type=int, default=32, help='Длина траектории (контекста) для обучения.')

    args = parser.parse_args()
    
    # Создаем уникальную директорию для сохранения новой модели
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.save_dir = os.path.join(args.save_dir_base, f"vintix_finetuned_{timestamp}")
    os.makedirs(args.save_dir, exist_ok=True)
    
    train(args) 