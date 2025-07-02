import torch
import numpy as np
from environment import ManipulatorEnv
from sac_agent import SAC
import os
import time
import csv
import re # Добавим для поиска эпизодов

def find_latest_episode(model_dir):
    """Находит путь к последнему сохраненному чекпоинту или модели в папке."""
    max_episode = 0
    latest_path = None
    is_checkpoint = False

    # Сначала ищем новый формат (чекпоинты)
    for f in os.listdir(model_dir):
        if f.startswith("checkpoint_episode_") and f.endswith(".pth"):
            match = re.search(r'checkpoint_episode_(\d+).pth', f)
            if match:
                is_checkpoint = True
                episode_num = int(match.group(1))
                if episode_num > max_episode:
                    max_episode = episode_num
    
    if is_checkpoint:
        latest_path = os.path.join(model_dir, f"checkpoint_episode_{max_episode}.pth")
        return max_episode, latest_path, True

    # Если чекпоинты не найдены, ищем старый формат
    for f in os.listdir(model_dir):
        if f.endswith("_actor.pth"):
            match = re.search(r'episode_(\d+)_actor.pth', f)
            if match:
                episode_num = int(match.group(1))
                if episode_num > max_episode:
                    max_episode = episode_num

    if max_episode > 0:
        latest_path = os.path.join(model_dir, f"episode_{max_episode}")
    
    return max_episode, latest_path, False


def train(model_save_dir, load_from_path=None, start_episode=0, is_checkpoint=True):
    """
    Главная функция для запуска процесса обучения.

    :param model_save_dir: Путь к папке для сохранения моделей и логов.
    :param load_from_path: (Опционально) Путь к файлам существующей модели для дообучения.
    :param start_episode: (Опционально) Номер эпизода, с которого начинать обучение.
    :param is_checkpoint: (Опционально) True, если модель в формате чекпоинта.
    """
    # --- 1. Гиперпараметры обучения ---
    max_episodes = 5000          # Общее количество эпизодов для обучения
    max_steps_per_episode = 500  # Максимальная длина одного эпизода
    start_timesteps = 1500       # Количество шагов со случайными действиями для заполнения буфера
    save_model_freq = 200        # Как часто (в эпизодах) сохранять модель
    
    os.makedirs(model_save_dir, exist_ok=True)
    
    # --- Настройка лог-файла ---
    log_file_path = os.path.join(model_save_dir, "training_log.csv")
    # Если продолжаем обучение, открываем файл на дозапись. Иначе - создаем новый.
    file_mode = 'a' if load_from_path else 'w'
    log_file = open(log_file_path, file_mode, newline='', encoding='utf-8')
    log_writer = csv.writer(log_file)
    # Если это новый файл, пишем заголовок
    if file_mode == 'w':
        log_writer.writerow(['episode', 'steps', 'reward', 'distance'])
    print(f"INFO: Логи {'дозаписываются' if file_mode == 'a' else 'сохраняются'} в: {log_file_path}")

    # --- 2. Инициализация среды и агента ---
    env = ManipulatorEnv(render=False) 
    
    state_dim = 14 
    action_dim = 6
    max_action = np.pi 

    agent = SAC(state_dim, action_dim, max_action)

    # Если указан путь для загрузки, загружаем веса модели
    if load_from_path:
        agent.load(load_from_path, is_checkpoint=is_checkpoint)
        print("-" * 50)
        print(f"ЗАГРУЗКА: Модель успешно загружена из '{load_from_path}'")
        print("-" * 50)
    
    print("-" * 50)
    print(f"Начинаем обучение. Всего эпизодов: {max_episodes}, Шагов в эпизоде: {max_steps_per_episode}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Max action: {max_action:.2f}")
    print(f"Модели будут сохраняться в: {model_save_dir}")
    print("-" * 50)

    # --- 3. Главный цикл обучения ---
    total_timesteps = 0
    for episode in range(start_episode, max_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(max_steps_per_episode):
            # В начале собираем опыт, выполняя случайные действия
            if total_timesteps < start_timesteps:
                action = np.random.uniform(low=-max_action, high=max_action, size=action_dim)
            else:
                action = agent.select_action(state)

            # Выполняем действие в среде
            next_state, reward, done, info = env.step(action)
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            total_timesteps += 1

            if total_timesteps > start_timesteps:
                agent.update()

            if done:
                break
        
        print(f"Эпизод: {episode+1}, Шагов: {step+1}, Награда: {episode_reward:.2f}, Дистанция: {info.get('distance', -1):.3f}")

        # Запись в лог-файл
        distance = info.get('distance', -1)
        log_writer.writerow([episode + 1, step + 1, episode_reward, distance])
        # Принудительно сбрасываем буфер на диск, чтобы не терять логи при сбое
        log_file.flush()

        # Сохраняем модель
        if (episode + 1) % save_model_freq == 0:
            save_path = os.path.join(model_save_dir, f"checkpoint_episode_{episode+1}.pth")
            agent.save(save_path)
            print(f"--- Чекпоинт сохранен в {save_path} ---")

    log_file.close()
    env.close()
    print("Обучение завершено.")


def main():
    """Интерактивное меню для выбора режима обучения."""
    models_base_dir = "models"
    if not os.path.exists(models_base_dir):
        os.makedirs(models_base_dir)

    print("\n" + "="*50)
    print(" РЕЖИМ ОБУЧЕНИЯ МОДЕЛИ МАНИПУЛЯТОРА ".center(50))
    print("="*50)
    print("1. Начать новое обучение")
    print("2. Продолжить обучение существующей модели")
    print("0. Выход")
    
    choice = input("Выберите опцию: ")

    if choice == '1':
        # Новое обучение
        run_name = f"sac_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        model_save_dir = os.path.join(models_base_dir, run_name)
        train(model_save_dir)

    elif choice == '2':
        # Продолжить обучение
        saved_models = [d for d in os.listdir(models_base_dir) if os.path.isdir(os.path.join(models_base_dir, d))]
        if not saved_models:
            print("\nОшибка: Не найдено сохраненных моделей для дообучения.")
            return

        print("\nВыберите модель для дообучения:")
        for i, model_name in enumerate(saved_models):
            print(f"{i+1}. {model_name}")

        try:
            model_choice_idx = int(input("Введите номер модели: ")) - 1
            if not 0 <= model_choice_idx < len(saved_models):
                raise ValueError
        except (ValueError, IndexError):
            print("Ошибка: Неверный выбор.")
            return

        selected_model_dir = os.path.join(models_base_dir, saved_models[model_choice_idx])
        
        start_episode, load_path, is_checkpoint = find_latest_episode(selected_model_dir)

        if load_path is None:
            print(f"\nОшибка: В папке '{selected_model_dir}' не найдено файлов моделей или чекпоинтов.")
            print("Начинаем обучение с нуля в этой же папке.")
            start_episode = 0
            is_checkpoint = True # Новые запуски всегда используют чекпоинты
            load_path = None # Убедимся, что ничего не загружаем
        else:
            print(f"\nПродолжаем обучение модели '{saved_models[model_choice_idx]}'.")
            print(f"Начинаем с эпизода {start_episode + 1}.")
            
        train(selected_model_dir, load_from_path=load_path, start_episode=start_episode, is_checkpoint=is_checkpoint)
    
    elif choice == '0':
        print("Выход.")
    
    else:
        print("Неверный выбор. Пожалуйста, введите 1, 2 или 0.")


if __name__ == '__main__':
    main() 