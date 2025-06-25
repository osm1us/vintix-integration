"""
Сопоставление координат: пиксели -> реальный мир.

Этот модуль представляет собой второй и ключевой этап настройки "зрения" робота.
После того как камера откалибрована (искажения устранены), этот модуль
позволяет создать точное сопоставление между 2D-координатами на изображении
и 2D-координатами на реальной рабочей плоскости (например, на столе).

Принцип работы:
1.  **Устранение дисторсии**: Сначала пиксельные координаты исправляются с
    помощью параметров, полученных на этапе калибровки камеры.
2.  **Гомография**: Используется матрица гомографии (3x3) для перевода
    "чистых" пиксельных координат в реальные мировые координаты (например, в см).
    Эта матрица вычисляется один раз на основе 4-х пар точек:
    ты указываешь 4 пиксельные координаты на изображении и соответствующие
    им 4 реальные координаты на столе.

Как использовать:
1.  Убедись, что у тебя есть файл `camera_params.json`, полученный
    после запуска `camera_calibration.py`.
2.  Запусти интерактивный скрипт или используй `main.py` для отображения
    видеопотока с камеры.
3.  Определи 4 опорные точки на рабочей плоскости (например, углы белого листа А4).
    - Запиши их пиксельные координаты `(u, v)` с видеопотока.
    - Измерь их реальные координаты `(X, Y)` линейкой от точки отсчета робота.
4.  Используй метод `calculate_and_save_homography`, передав ему эти 4 пары точек.
    Он создаст файл `homography_matrix.json`.
5.  После этого `CoordinateMapper` готов к работе: метод `pixel_to_world` будет
    преобразовывать любые пиксели в реальные координаты.
"""

import cv2
import numpy as np
import json
import os

class CoordinateMapper:
    """
    Класс для преобразования пиксельных координат в мировые координаты.
    """
    def __init__(self, camera_params_path="camera_params.json", homography_path="homography_matrix.json"):
        """
        Инициализирует маппер.

        Args:
            camera_params_path (str): Путь к файлу с параметрами камеры.
            homography_path (str): Путь к файлу с матрицей гомографии.
        """
        self.camera_matrix = None
        self.dist_coeffs = None
        self.homography_matrix = None
        self.is_ready = False

        # 1. Загружаем параметры калибровки камеры
        if not os.path.exists(camera_params_path):
            print(f"Предупреждение: Файл параметров камеры '{camera_params_path}' не найден.")
            print("Необходимо сначала запустить camera_calibration.py")
        else:
            try:
                with open(camera_params_path, 'r') as f:
                    params = json.load(f)
                    self.camera_matrix = np.array(params["camera_matrix"])
                    self.dist_coeffs = np.array(params["distortion_coefficients"])
                print(f"Параметры камеры успешно загружены из '{camera_params_path}'.")
            except Exception as e:
                print(f"Ошибка при загрузке параметров камеры: {e}")
                return

        # 2. Загружаем матрицу гомографии
        if not os.path.exists(homography_path):
            print(f"Предупреждение: Файл матрицы гомографии '{homography_path}' не найден.")
            print("Используйте метод 'calculate_and_save_homography', чтобы создать его.")
        else:
            self.load_homography_matrix(homography_path)

        self._check_if_ready()

    def _check_if_ready(self):
        """Проверяет, готовы ли все компоненты для работы."""
        if self.camera_matrix is not None and self.dist_coeffs is not None and self.homography_matrix is not None:
            self.is_ready = True
            print("CoordinateMapper готов к работе.")
        else:
            self.is_ready = False
            print("CoordinateMapper не готов. Проверьте файлы конфигурации.")

    def load_homography_matrix(self, path="homography_matrix.json"):
        """
        Загружает матрицу гомографии из файла.

        Args:
            path (str): Путь к файлу.
        """
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self.homography_matrix = np.array(data['homography_matrix'])
            print(f"Матрица гомографии успешно загружена из '{path}'.")
            self._check_if_ready()
        except Exception as e:
            print(f"Ошибка при загрузке матрицы гомографии: {e}")
            self.homography_matrix = None

    def calculate_and_save_homography(self, pixel_points, world_points, path="homography_matrix.json"):
        """
        Вычисляет и сохраняет матрицу гомографии.

        Args:
            pixel_points (list of tuples): Список 4-х пиксельных координат (u, v).
            world_points (list of tuples): Список 4-х соответствующих мировых координат (X, Y).
            path (str): Путь для сохранения файла.
        """
        if self.camera_matrix is None:
            print("Ошибка: Параметры камеры не загружены. Невозможно вычислить гомографию.")
            return False

        if len(pixel_points) != 4 or len(world_points) != 4:
            raise ValueError("Необходимо ровно 4 пары точек для вычисления гомографии.")

        # Сначала "выпрямляем" пиксельные координаты
        pixel_points_np = np.array(pixel_points, dtype=np.float32).reshape(-1, 1, 2)
        undistorted_pixel_points = cv2.undistortPoints(pixel_points_np, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)

        # Вычисляем матрицу гомографии
        self.homography_matrix, status = cv2.findHomography(undistorted_pixel_points, np.array(world_points))

        if self.homography_matrix is None:
            print("Не удалось вычислить матрицу гомографии. Проверьте точки.")
            return False

        # Сохраняем матрицу
        try:
            data = {"homography_matrix": self.homography_matrix.tolist()}
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Матрица гомографии успешно вычислена и сохранена в '{path}'.")
            self._check_if_ready()
            return True
        except Exception as e:
            print(f"Ошибка при сохранении матрицы гомографии: {e}")
            return False

    def pixel_to_world(self, u, v):
        """
        Преобразует одну пиксельную координату (u, v) в мировую (X, Y).

        Args:
            u (int or float): Пиксельная координата X.
            v (int or float): Пиксельная координата Y.

        Returns:
            tuple: Кортеж (X, Y) в мировых координатах или None, если система не готова.
        """
        if not self.is_ready:
            print("Ошибка: CoordinateMapper не готов к преобразованию.")
            return None

        # 1. "Выпрямляем" точку
        pixel_coords = np.array([[[u, v]]], dtype=np.float32)
        undistorted_coords = cv2.undistortPoints(pixel_coords, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
        
        # 2. Применяем гомографию
        # Нам нужно передать точку в гомогенных координатах (u, v, 1)
        uv_hom = np.array([undistorted_coords[0][0][0], undistorted_coords[0][0][1], 1])
        xyw_hom = self.homography_matrix @ uv_hom
        
        # 3. Нормализуем, разделив на w
        if xyw_hom[2] != 0:
            x = xyw_hom[0] / xyw_hom[2]
            y = xyw_hom[1] / xyw_hom[2]
            return (x, y)
        else:
            return None

# Пример использования (раскомментировать, когда будут реальные данные)
if __name__ == '__main__':
    print("Демонстрация работы CoordinateMapper.")

    # Создаем "заглушки" для файлов, чтобы продемонстрировать вычисление
    if not os.path.exists("camera_params.json"):
        print("\nСоздаем тестовый файл 'camera_params.json'...")
        dummy_cam_params = {
            "camera_matrix": [[1000, 0, 640], [0, 1000, 360], [0, 0, 1]],
            "distortion_coefficients": [[0, 0, 0, 0, 0]]
        }
        with open("camera_params.json", 'w') as f:
            json.dump(dummy_cam_params, f, indent=4)
    
    # 1. Инициализация. Он увидит, что матрицы гомографии нет.
    mapper = CoordinateMapper()

    # 2. Вычисление матрицы гомографии (это нужно будет сделать один раз с реальными данными)
    print("\nШаг 2: Вычисление матрицы гомографии (с тестовыми данными)...")
    
    # ПРИМЕР: Представим, что мы нашли 4 угла листа А4 на картинке
    # и измерили их реальные координаты на столе.
    # ЗАМЕНИТЬ ЭТИ ЗНАЧЕНИЯ НА РЕАЛЬНЫЕ!
    pixel_points_example = [
        (100, 100),  # Левый верхний угол на картинке
        (1180, 100), # Правый верхний
        (1180, 820), # Правый нижний
        (100, 820)   # Левый нижний
    ]
    # Реальные координаты этих углов на столе (в см)
    world_points_example = [
        (0, 0),      # Левый верхний - наша точка (0,0)
        (29.7, 0),   # Правый верхний (длина А4)
        (29.7, 21.0),# Правый нижний (длина и ширина А4)
        (0, 21.0)    # Левый нижний (ширина А4)
    ]

    mapper.calculate_and_save_homography(pixel_points_example, world_points_example)

    # 3. Использование
    if mapper.is_ready:
        print("\nШаг 3: Тестирование преобразования...")
        test_pixel = (640, 460) # Примерно центр листа
        world_coords = mapper.pixel_to_world(test_pixel[0], test_pixel[1])
        
        if world_coords:
            print(f"Пиксель {test_pixel} -> Мировые координаты (см): ({world_coords[0]:.2f}, {world_coords[1]:.2f})")
            # Ожидаемый результат для тестовых данных: примерно (14.85, 10.5)

    # Очистка созданных "заглушек"
    if os.path.exists("camera_params.json") and "dummy" in open("camera_params.json").read():
        os.remove("camera_params.json")
    if os.path.exists("homography_matrix.json"):
        os.remove("homography_matrix.json")
        print("\nТестовые файлы удалены.") 