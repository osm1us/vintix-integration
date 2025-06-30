import torch
import numpy as np
from environment import ManipulatorEnv
from sac_agent import SAC
import os
import time

def train():
    """
    Главная функция для запуска процесса обучения.
    """
    # --- 1. Гиперпараметры обучения ---
    max_episodes = 5000          # Общее количество эпизодов для обучения
    max_steps_per_episode = 500  # Максимальная длина одного эпизода
    start_timesteps = 1000       # Количество шагов со случайными действиями для заполнения буфера
    save_model_freq = 200        # Как часто (в эпизодах) сохранять модель
    
    run_name = f"sac_{int(time.time())}" # Уникальное имя для этого запуска
    model_save_dir = f"models/{run_name}"
    os.makedirs(model_save_dir, exist_ok=True)
    print(f"INFO: Модели будут сохраняться в: {model_save_dir}")

    # --- 2. Инициализация среды и агента ---
    # Запускаем среду без рендера для максимальной скорости обучения
    # render=True можно включить для отладки
    env = ManipulatorEnv(render=True) 
    
    # Определяем размерности пространства состояний и действий
    # obs = env.reset()
    # state_dim = obs.shape[0]
    # action_dim = env.num_controllable_joints
    
    # Эти значения мы знаем из отчета
    state_dim = 14 
    action_dim = 6
    
    # Максимальное значение действия (углы в радианах). 
    # Это важно для Actor, чтобы он масштабировал свой выход.
    # Для большинства роботов-манипуляторов pi - разумный предел.
    max_action = np.pi 

    agent = SAC(state_dim, action_dim, max_action)
    
    print("-" * 50)
    print(f"Начинаем обучение. Эпизодов: {max_episodes}, Шагов в эпизоде: {max_steps_per_episode}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}, Max action: {max_action:.2f}")
    print("-" * 50)

    # --- 3. Главный цикл обучения ---
    total_timesteps = 0
    for episode in range(max_episodes):
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
            
            # Сохраняем переход в буфер
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # Обновляем состояние и награду
            state = next_state
            episode_reward += reward
            total_timesteps += 1

            # Обновляем агента, если в буфере достаточно данных
            if total_timesteps > start_timesteps:
                agent.update()

            if done:
                break
        
        print(f"Эпизод: {episode+1}, Шагов: {step+1}, Награда: {episode_reward:.2f}, Дистанция: {info.get('distance', -1):.3f}")

        # Сохраняем модель
        if (episode + 1) % save_model_freq == 0:
            save_path = os.path.join(model_save_dir, f"episode_{episode+1}")
            agent.save(save_path)
            print(f"--- Модель сохранена в {save_path}_actor.pth и {save_path}_critic.pth ---")

    env.close()
    print("Обучение завершено.")


if __name__ == '__main__':
    # Из-за особенностей PyBullet и multiprocessing на Windows,
    # лучше обернуть вызов в main guard.
    train() 