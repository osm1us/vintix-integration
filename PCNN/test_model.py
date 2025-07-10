import argparse
import time
import numpy as np
import torch
from sac_agent import SAC
from environment import ManipulatorEnv
import os
import re


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


def run_test(env, agent, num_episodes, model_path):
    """
    Загружает и тестирует агента в среде с визуализацией.
    """
    print("-" * 50)
    print(f"Тестирование модели: {model_path}")
    print(f"Количество эпизодов: {num_episodes}")
    print("-" * 50)

    # --- 2. Цикл тестирования ---
    success_count = 0
    total_steps = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(700): # Макс. шагов в эпизоде, как при обучении
            # Выбираем действие детерминированно (без шума)
            action = agent.select_action(state, deterministic=True)
            
            state, reward, done, info = env.step(action)
            episode_reward += reward
            
            time.sleep(1./240.) # Небольшая задержка для плавности картинки
            
            if done:
                break
        
        final_distance = info.get('distance', -1)
        
        # В done уже заложена проверка на высоту + захват.
        if done and final_distance < 0.15: # Дополнительно убедимся, что мы близко
            success_count += 1
            print(f"Эпизод {episode+1:02d}: УСПЕХ! Шагов: {step+1}, Финальная награда: {episode_reward:.2f}")
        else:
            print(f"Эпизод {episode+1:02d}: Провал. Шагов: {step+1}, Дистанция: {final_distance:.3f}, Награда: {episode_reward:.2f}")
        
        total_steps += (step + 1)
        
    # --- 3. Вывод статистики ---
    success_rate = (success_count / num_episodes) * 100
    avg_steps = total_steps / num_episodes
    
    print("\n" + "="*50)
    print(" РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ ".center(50))
    print("="*50)
    print(f"Успешность (Success Rate): {success_count} / {num_episodes} ({success_rate:.1f}%)")
    print(f"Среднее количество шагов за эпизод: {avg_steps:.1f}")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="Тестирование обученного SAC агента.")
    parser.add_argument(
        'model_path', 
        type=str,
        nargs='?', # Делаем путь к модели необязательным
        default=None,
        help="Путь к папке с моделью. Если не указан, будет предложен выбор."
    )
    parser.add_argument(
        '--episodes', 
        type=int, 
        default=10, 
        help="Количество эпизодов для тестирования."
    )

    args = parser.parse_args()

    model_dir = args.model_path
    
    # --- Интерактивный выбор модели, если путь не указан ---
    if not model_dir:
        models_base_dir = "models"
        if not os.path.exists(models_base_dir) or not os.listdir(models_base_dir):
            print(f"Ошибка: Папка '{models_base_dir}' пуста или не существует. Нечего тестировать.")
            return

        saved_models = sorted([d for d in os.listdir(models_base_dir) if os.path.isdir(os.path.join(models_base_dir, d))])
        if not saved_models:
            print(f"Ошибка: В '{models_base_dir}' не найдено папок с моделями.")
            return
        
        print("\n" + "="*50)
        print(" Выберите модель для тестирования ".center(50))
        print("="*50)
        for i, model_name in enumerate(saved_models):
            print(f"{i+1}. {model_name}")
        print("0. Выход")
        
        try:
            choice = int(input("Введите номер модели: "))
            if choice == 0:
                print("Выход.")
                return
            if not 1 <= choice <= len(saved_models):
                raise ValueError
            model_dir = os.path.join(models_base_dir, saved_models[choice - 1])
        except (ValueError, IndexError):
            print("Ошибка: Неверный выбор.")
            return

    # --- Поиск и загрузка модели ---
    _, model_file_path, is_checkpoint = find_latest_episode(model_dir)

    if model_file_path is None:
        print(f"Ошибка: В папке '{model_dir}' не найдено файлов модели (.pth).")
        return

    # Инициализация среды
    env = ManipulatorEnv(render=True)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SAC(state_dim, action_dim, max_action)
    
    try:
        agent.load(model_file_path, is_checkpoint)
        print(f"\nМодель успешно загружена из '{model_file_path}'")
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить модель.")
        print(f"Ошибка: {e}")
        env.close()
        return
        
    run_test(env, agent, args.episodes, model_path=model_dir)

    env.close()
    print("Тестирование завершено.")


if __name__ == '__main__':
    main() 