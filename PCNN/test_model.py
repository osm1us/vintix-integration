import torch
import numpy as np
from environment import ManipulatorEnv
from sac_agent import SAC
import os
import time
import re
import matplotlib.pyplot as plt
import pybullet as p

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

def test_agent(model_path, is_checkpoint, num_episodes=10):
    """
    Загружает и тестирует агента в среде с визуализацией.

    :param model_path: Путь к файлу чекпоинта или базовый путь к модели.
    :param is_checkpoint: True, если модель в формате чекпоинта.
    :param num_episodes: Количество тестовых эпизодов.
    """
    print("-" * 50)
    print(f"Тестирование модели: {model_path}")
    print(f"Количество эпизодов: {num_episodes}")
    print("-" * 50)

    # --- 1. Инициализация среды и агента ---
    # Включаем рендер для визуализации
    env = ManipulatorEnv(render=True)
    
    state_dim = 14 
    action_dim = 6
    max_action = np.pi 
    agent = SAC(state_dim, action_dim, max_action)
    
    try:
        agent.load(model_path, is_checkpoint=is_checkpoint)
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить модель из {model_path}.")
        print(f"Ошибка: {e}")
        env.close()
        return

    # --- Настройка для живой визуализации камеры ---
    plt.ion() # Включаем интерактивный режим
    fig, ax = plt.subplots(figsize=(6, 6))
    img_plot = ax.imshow(np.zeros((320, 320, 4))) 
    ax.set_title("Simulated Camera View")
    plt.show(block=False)

    # --- 2. Цикл тестирования ---
    success_count = 0
    total_steps = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(500): # Макс. шагов в эпизоде, как при обучении
            # Выбираем действие детерминированно (без шума)
            action = agent.select_action(state, deterministic=True)
            
            state, reward, done, info = env.step(action)
            episode_reward += reward
            
            # --- Обновление вида с камеры ---
            _, _, rgba_img_flat, _, _ = p.getCameraImage(
                width=320,
                height=320,
                viewMatrix=env.camera_view_matrix,
                projectionMatrix=env.camera_proj_matrix,
                renderer=p.ER_TINY_RENDERER
            )
            rgba_img = np.reshape(rgba_img_flat, (320, 320, 4))
            img_plot.set_data(rgba_img)
            # Обновляем заголовок с 2D координатами цели, которые видит агент
            ax.set_title(f"Simulated Camera View\nTarget (norm): ({state[0]:.2f}, {state[1]:.2f})")
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            time.sleep(1./240.) # Небольшая задержка для плавности картинки
            
            if done:
                break
        
        final_distance = info.get('distance', -1)
        
        if done:
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

    plt.close(fig) # Закрываем окно с графиком
    env.close()

def main():
    """Интерактивное меню для выбора модели для тестирования."""
    models_base_dir = "models"
    if not os.path.exists(models_base_dir) or not os.listdir(models_base_dir):
        print("Ошибка: Папка 'models' пуста или не существует. Нечего тестировать.")
        return

    print("\n" + "="*50)
    print(" РЕЖИМ ТЕСТИРОВАНИЯ МОДЕЛИ ".center(50))
    print("="*50)
    
    saved_models = [d for d in os.listdir(models_base_dir) if os.path.isdir(os.path.join(models_base_dir, d))]
    if not saved_models:
        print("Ошибка: Не найдено сохраненных моделей для тестирования.")
        return

    print("Выберите модель для тестирования:")
    for i, model_name in enumerate(saved_models):
        print(f"{i+1}. {model_name}")
    print("0. Выход")
    
    try:
        choice = int(input("Выберите опцию: "))
        if choice == 0:
            print("Выход.")
            return
        if not 1 <= choice <= len(saved_models):
            raise ValueError
    except ValueError:
        print("Ошибка: Неверный выбор.")
        return

    selected_model_dir = os.path.join(models_base_dir, saved_models[choice - 1])
    
    episode_num, model_path, is_checkpoint = find_latest_episode(selected_model_dir)

    if model_path:
        print(f"\nНайден последний чекпоинт: эпизод {episode_num}")
        
        # Запрашиваем количество эпизодов для теста
        try:
            num_episodes_str = input("Введите количество эпизодов для теста [по умолчанию: 10]: ")
            if num_episodes_str.strip() == "":
                num_episodes = 10
            else:
                num_episodes = int(num_episodes_str)
            if num_episodes <= 0: raise ValueError
        except ValueError:
            print("Некорректный ввод. Будет использовано значение по умолчанию (10).")
            num_episodes = 10
            
        test_agent(model_path, is_checkpoint, num_episodes=num_episodes)
    else:
        print(f"\nОшибка: В папке '{selected_model_dir}' не найдено файлов моделей или чекпоинтов.")

if __name__ == '__main__':
    main() 