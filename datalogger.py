import os
import h5py
import numpy as np
import datetime
from typing import List, Dict, Any

class HDF5Logger:
    """
    Класс для записи данных о работе робота (эпизодов) в файлы формата HDF5.

    Каждый эпизод (например, одна попытка взять кубик) сохраняется в отдельный
    HDF5 файл. Внутри файла создается группа, соответствующая эпизоду,
    которая содержит четыре набора данных:
    - proprio_observation: Проприоцептивное состояние робота (углы сочленений).
    - action: Действие, предпринятое агентом (например, дельта углов).
    - reward: Награда за каждый шаг.
    - step_num: Номер шага внутри эпизода.
    """

    def __init__(self, log_dir: str):
        """
        Инициализирует логгер.

        Args:
            log_dir (str): Директория для сохранения HDF5 файлов.
        """
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        self.episode_buffer: Dict[str, List[Any]] = {}
        self.reset_episode_buffer()

    def reset_episode_buffer(self):
        """Сбрасывает внутренний буфер для начала нового эпизода."""
        self.episode_buffer = {
            "proprio_observation": [],
            "action": [],
            "reward": [],
            "step_num": [],
        }

    def log_step(self, observation: np.ndarray, action: np.ndarray, reward: float, step_num: int):
        """
        Записывает данные одного шага в буфер.

        Args:
            observation (np.ndarray): Вектор проприоцептивного состояния (обычно 6 углов сочленений).
            action (np.ndarray): Вектор действий, предпринятых агентом (обычно дельта для 3 углов).
            reward (float): Награда за текущий шаг.
            step_num (int): Номер шага в эпизоде.
        """
        self.episode_buffer["proprio_observation"].append(observation)
        self.episode_buffer["action"].append(action)
        self.episode_buffer["reward"].append(reward)
        self.episode_buffer["step_num"].append(step_num)

    def finish_episode(self, final_reward: float):
        """
        Завершает эпизод, применяет финальную награду и сохраняет данные в HDF5.

        Args:
            final_reward (float): Финальная награда за эпизод (+1.0 за успех, -1.0 за провал).
        """
        if not self.episode_buffer["step_num"]:
            print("Предупреждение: Попытка сохранить пустой эпизод. Запись отменена.")
            return

        # Устанавливаем финальную награду для последнего шага
        if self.episode_buffer["reward"]:
            self.episode_buffer["reward"][-1] = final_reward

        # Формируем имя файла
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_path = os.path.join(self.log_dir, f"episode_{timestamp}.h5")

        try:
            with h5py.File(file_path, 'w') as hf:
                # Создаем группу для этого эпизода
                episode_group = hf.create_group("episode_data")

                # Конвертируем буферы в numpy массивы и сохраняем
                for key, data in self.episode_buffer.items():
                    numpy_data = np.array(data, dtype=np.float32 if key != 'step_num' else np.int32)
                    
                    # Для reward и step_num делаем правильную форму
                    if key == 'reward':
                        numpy_data = numpy_data.reshape(-1, 1)
                    
                    episode_group.create_dataset(key, data=numpy_data, compression="gzip")
            
            print(f"Эпизод успешно сохранен в: {file_path}")

        except Exception as e:
            print(f"Ошибка при сохранении эпизода в HDF5: {e}")
        
        finally:
            # Сбрасываем буфер для следующего эпизода
            self.reset_episode_buffer()

if __name__ == '__main__':
    # Пример использования логгера
    print("Демонстрация работы HDF5Logger...")
    
    # 1. Инициализация
    logger = HDF5Logger(log_dir="data/training_logs_demo")

    # 2. Начало эпизода (неявное, через первый log_step)
    logger.reset_episode_buffer()
    print("Начат новый эпизод.")

    # 3. Симуляция нескольких шагов
    num_steps = 10
    # Данные соответствуют реальному использованию в run_vintix.py
    observation_dim = 6 # 6 углов сочленений
    action_dim = 3      # дельта для первых 3-х суставов

    for i in range(num_steps):
        # Генерируем случайные данные для примера
        mock_observation = np.random.rand(observation_dim).astype(np.float32)
        mock_action = np.random.rand(action_dim).astype(np.float32)
        mock_reward = 0.0
        
        logger.log_step(mock_observation, mock_action, mock_reward, step_num=i)
        print(f"Шаг {i} записан.")

    # 4. Завершение эпизода (симуляция успеха)
    final_reward_success = 1.0
    logger.finish_episode(final_reward_success)
    print("Эпизод (успех) завершен и сохранен.")

    # 5. Симуляция второго, короткого эпизода (провал)
    logger.reset_episode_buffer()
    print("\nНачат второй эпизод.")
    for i in range(3):
        mock_observation = np.random.rand(observation_dim).astype(np.float32)
        mock_action = np.random.rand(action_dim).astype(np.float32)
        logger.log_step(mock_observation, mock_action, 0.0, i)
        print(f"Шаг {i} записан.")
    
    final_reward_fail = -1.0
    logger.finish_episode(final_reward_fail)
    print("Эпизод (провал) завершен и сохранен.")

    # Проверка созданных файлов
    print(f"\nПроверьте содержимое папки 'data/training_logs_demo'") 