"""
Сопоставление координат: пиксели -> реальный мир.

Этот модуль позволяет создать точное сопоставление между 2D-координатами на 
изображении и реальными мировыми координатами на рабочей плоскости (например, на столе).

Принцип работы:
1.  **Загрузка калибровки**: Модуль загружает параметры камеры, включая
    матрицу камеры, коэффициенты дисторсии, а также векторы вращения (rvecs) и 
    трансляции (tvecs) из файла `camera_params.yaml`. Эти векторы описывают
    положение калибровочной доски относительно камеры.
2.  **Вычисление гомографии**: На основе этих векторов и матрицы камеры
    вычисляется матрица гомографии. Эта матрица позволяет "спроецировать"
    пиксельные координаты на рабочую плоскость, положение которой было
    зафиксировано во время калибровки.
3.  **Преобразование**: Метод `pixel_to_world` использует вычисленную
    матрицу для преобразования любой точки с изображения в реальные координаты.

Как использовать:
1.  Запустите `camera_calibration.py` для получения файла `camera_params.yaml`.
2.  Убедитесь, что путь к этому файлу правильно указан в `config.py`.
3.  Создайте экземпляр `CoordinateMapper`, и он будет готов к работе.
"""

import cv2
import numpy as np
import json
import os
import yaml
import logging

from config import settings

logger = logging.getLogger(__name__)

class CoordinateMapper:
    """
    Преобразует 2D координаты с изображения камеры в 3D мировые координаты робота.
    """

    def __init__(self, work_plane_z: float = 0.0):
        """
        Инициализирует маппер с использованием файла калибровки из глобальных настроек.

        Args:
            work_plane_z (float): Высота рабочей плоскости по оси Z в мировых координатах.
        """
        self.work_plane_z = work_plane_z
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.inv_homography_matrix = None
        
        calibration_file = settings.system.CAMERA_PARAMS_PATH

        if not self._load_calibration(calibration_file):
            raise ValueError(f"Не удалось загрузить или обработать файл калибровки: {calibration_file}")

    def _load_calibration(self, file_path: str) -> bool:
        """Загружает данные калибровки из YAML файла."""
        try:
            with open(file_path, 'r') as f:
                calib_data = yaml.safe_load(f)
            
            self.mtx = np.array(calib_data["camera_matrix"])
            self.dist = np.array(calib_data["dist_coeff"])
            self.rvecs = np.array(calib_data["rvecs"])
            self.tvecs = np.array(calib_data["tvecs"])

            logger.info(f"Данные калибровки успешно загружены из {file_path}")
            
            # Рассчитываем матрицу гомографии при загрузке
            self._compute_homography()
            return True

        except FileNotFoundError:
            logger.error(f"Файл калибровки не найден: {file_path}")
            return False
        except (KeyError, TypeError) as e:
            logger.error(f"Ошибка в структуре файла калибровки {file_path}: {e}")
            return False

    def _compute_homography(self):
        """
        Рассчитывает матрицу гомографии для преобразования координат.
        Этот метод был значительно упрощен, так как camera_calibration.py
        теперь сохраняет rvecs и tvecs.
        """
        if self.rvecs is None or self.tvecs is None:
            logger.error("Невозможно рассчитать гомографию: отсутствуют векторы вращения или трансляции.")
            return

        # Используем первый вектор вращения и трансляции
        rvec = self.rvecs[0]
        tvec = self.tvecs[0]
        
        # Преобразуем вектор вращения в матрицу вращения
        R, _ = cv2.Rodrigues(rvec)
        
        # Собираем матрицу гомографии.
        # Исключаем Z-компоненту из матрицы вращения, так как мы проецируем на плоскость.
        H = np.hstack((R[:, 0:2], tvec))
        H_inv = np.linalg.inv(self.mtx @ H)
        
        self.inv_homography_matrix = H_inv
        logger.info("Матрица гомографии успешно рассчитана и инвертирована.")


    def pixel_to_world(self, u: int, v: int) -> np.ndarray | None:
        """
        Преобразует 2D пиксельные координаты (u, v) в 3D мировые координаты (X, Y, Z).

        Args:
            u (int): Координата X на изображении (пиксель).
            v (int): Координата Y на изображении (пиксель).

        Returns:
            np.ndarray: Массив [X, Y, Z] в мировых координатах или None.
        """
        if self.inv_homography_matrix is None:
            logger.error("Преобразование невозможно: матрица гомографии не рассчитана.")
            return None

        # Шаг 1: Устраняем дисторсию для входной точки.
        # Гомография корректно работает только на точках без искажений.
        pixel_coords_distorted = np.array([[[u, v]]], dtype=np.float32)
        
        pixel_coords_undistorted = cv2.undistortPoints(
            pixel_coords_distorted, self.mtx, self.dist, P=self.mtx
        )
        
        # Извлекаем исправленные координаты
        u_corr, v_corr = pixel_coords_undistorted[0, 0]

        # Шаг 2: Применяем гомографию к ИСПРАВЛЕННОЙ точке.
        pixel_coords_homogeneous = np.array([u_corr, v_corr, 1], dtype=np.float32)
        world_coords_2d_homogeneous = self.inv_homography_matrix @ pixel_coords_homogeneous
        
        # Нормализуем, чтобы последняя компонента была 1
        if world_coords_2d_homogeneous[2] == 0:
            logger.error("Ошибка при преобразовании координат: деление на ноль.")
            return None
            
        world_x = world_coords_2d_homogeneous[0] / world_coords_2d_homogeneous[2]
        world_y = world_coords_2d_homogeneous[1] / world_coords_2d_homogeneous[2]
        
        world_coords_3d = np.array([world_x, world_y, self.work_plane_z])
        
        logger.debug(f"Pixel ({u}, {v}) -> Corrected ({u_corr:.2f}, {v_corr:.2f}) -> World ({world_coords_3d[0]:.4f}, {world_coords_3d[1]:.4f}, {world_coords_3d[2]:.4f})")
        return world_coords_3d
        
    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Устраняет дисторсию на изображении.
        
        Args:
            frame (np.ndarray): Искаженное изображение.

        Returns:
            np.ndarray: Изображение без дисторсии.
        """
        if self.mtx is None or self.dist is None:
            logger.warning("Невозможно устранить дисторсию: данные калибровки отсутствуют.")
            return frame
            
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (w, h), 1, (w, h))
        
        dst = cv2.undistort(frame, self.mtx, self.dist, None, newcameramtx)
        
        # Обрезаем изображение, чтобы убрать черные поля
        x, y, w, h = roi
        if w > 0 and h > 0:
            dst = dst[y:y+h, x:x+w]
        
        return dst

# Пример использования удален, так как он был устаревшим и нерабочим.
# Для проверки используйте основной цикл приложения. 