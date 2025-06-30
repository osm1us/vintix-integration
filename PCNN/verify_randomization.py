import time
import matplotlib.pyplot as plt
import numpy as np
from environment import ManipulatorEnv
import pybullet as p

def verify():
    """
    Скрипт для визуальной проверки работы Domain Randomization.
    - Показывает основное окно симуляции (твои глаза).
    - В отдельном окне показывает то, что "видит" виртуальная камера робота (глаза агента).
    
    Каждые 3 секунды среда будет сбрасываться, и вы сможете увидеть,
    как меняется фон (текстура/белый цвет) и ракурс камеры в обоих окнах.
    """
    env = None
    # Убедимся, что matplotlib не падает в безголовом режиме, если он есть
    fig = None 
    try:
        # Запускаем с рендером, чтобы видеть основное окно
        env = ManipulatorEnv(render=True)
        
        # --- Настройка для живой визуализации камеры робота ---
        plt.ion() 
        fig, ax = plt.subplots()
        # imshow() вернет объект, который мы можем обновлять
        img_plot = ax.imshow(np.zeros((320, 320, 4))) 
        ax.set_title("Глаза робота (виртуальная камера)")
        fig.canvas.draw()
        
        print("--- Запуск верификации Domain Randomization ---")
        print("Смотрите на оба окна. Сброс среды каждые 3 секунды.")
        print("Нажмите Ctrl+C в терминале, чтобы остановить.")

        while p.isConnected(env.physics_client):
            # Сбрасываем среду, чтобы применилась рандомизация
            obs = env.reset()
            
            # Получаем изображение с виртуальной камеры (то, что видит агент)
            # Мы обращаемся к данным среды, чтобы гарантировать, что используем ту же матрицу, что и агент
            _, _, rgba_img_flat, _, _ = p.getCameraImage(
                width=320,
                height=320,
                viewMatrix=env.camera_view_matrix,
                projectionMatrix=env.camera_proj_matrix,
                renderer=p.ER_TINY_RENDERER
            )
            rgba_img = np.reshape(rgba_img_flat, (320, 320, 4))

            # Обновляем окно matplotlib
            img_plot.set_data(rgba_img)
            # Обновляем заголовок с текущим FOV для наглядности
            ax.set_title(f"Глаза робота (FOV: {env.camera_proj_matrix[0]:.1f})")
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            # Просто ждем 3 секунды, чтобы ты мог все рассмотреть
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\nВерификация остановлена.")
    finally:
        if env:
            env.close()
        if fig:
            plt.ioff()
            plt.close(fig)
        print("Окна закрыты.")

if __name__ == '__main__':
    verify() 