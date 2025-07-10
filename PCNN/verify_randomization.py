import matplotlib.pyplot as plt
import numpy as np
from environment import ManipulatorEnv
import pybullet as p
import os

def verify_spawn_area():
    """
    Скрипт для визуальной проверки области появления (спавна) кубика.

    Он запускает симуляцию в фоновом режиме (без GUI), 200 раз сбрасывает
    среду и собирает начальные координаты кубика. Затем строит 2D-график,
    показывающий точное распределение точек появления. Это позволяет
    визуально оценить и проверить текущий диапазон, заданный в environment.py.
    """
    env = None
    spawn_points = []
    num_episodes = 200

    # Подавляем лишний вывод PyBullet в консоль
    # Сохраняем оригинальные дескрипторы
    original_stdout = os.dup(1)
    original_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)

    try:
        print(f"--- Запуск верификации области спавна ---")
        print(f"Симуляция будет сброшена {num_episodes} раз в фоновом режиме...")
        
        # Перенаправляем вывод в /dev/null
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        
        # Запускаем без рендера, нам не нужно окно симуляции
        env = ManipulatorEnv(render=False)

        for i in range(num_episodes):
            env.reset()
            # Получаем начальную позицию кубика, которая была установлена в reset
            pos, _ = p.getBasePositionAndOrientation(env.cube_id)
            spawn_points.append(pos)
        
    finally:
        # Восстанавливаем стандартный вывод
        os.dup2(original_stdout, 1)
        os.dup2(original_stderr, 2)
        os.close(devnull)
        os.close(original_stdout)
        os.close(original_stderr)

        if env:
            env.close()
        print("Сбор данных завершен. Закрытие симулятора.")

    # --- Построение графика ---
    if spawn_points:
        x_coords = [p[0] for p in spawn_points]
        y_coords = [p[1] for p in spawn_points]
        
        plt.figure(figsize=(8, 8))
        plt.scatter(x_coords, y_coords, s=10, alpha=0.7)
        plt.title(f'Область появления кубика ({num_episodes} точек)')
        plt.xlabel("Ось X (m)")
        plt.ylabel("Ось Y (m)")
        plt.grid(True)
        plt.axis('equal') # Делаем масштаб осей одинаковым
        
        # Находим и отображаем границы
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        plt.axvline(x=min_x, color='r', linestyle='--', label=f'X_min: {min_x:.3f}')
        plt.axvline(x=max_x, color='r', linestyle='--', label=f'X_max: {max_x:.3f}')
        plt.axhline(y=min_y, color='g', linestyle='--', label=f'Y_min: {min_y:.3f}')
        plt.axhline(y=max_y, color='g', linestyle='--', label=f'Y_max: {max_y:.3f}')
        
        plt.legend()
        print("\nГраницы области появления (реальные):")
        print(f"  X: ({min_x:.3f}, {max_x:.3f})")
        print(f"  Y: ({min_y:.3f}, {max_y:.3f})")
        print("\nОжидаемые границы (из кода):")
        print("  X: (0.378, 0.522)")
        print("  Y: (-0.128, 0.128)")

        plt.show()
    else:
        print("Не удалось собрать точки для построения графика.")


if __name__ == '__main__':
    verify_spawn_area() 