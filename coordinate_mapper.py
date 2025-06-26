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
    Преобразует 2D координаты с изображения камеры в 3D мировые координаты робота,
    используя стабильную робото-центричную систему.
    """

    def __init__(self,
                 intrinsic_params_path: str = "camera_params.yaml",
                 extrinsic_params_path: str = "hand_eye_params.yaml"):
        """
        Инициализирует маппер, загружая параметры внутренней и внешней калибровки.

        Args:
            intrinsic_params_path (str): Путь к файлу с внутренними параметрами камеры.
            extrinsic_params_path (str): Путь к файлу с параметрами калибровки "Рука-Глаз".
        """
        self.camera_matrix = None
        self.dist_coeffs = None
        self.transform_robot_to_camera = None
        self.transform_camera_to_robot = None

        if not self._load_intrinsic_params(intrinsic_params_path):
            raise ValueError(f"Критическая ошибка: не удалось загрузить файл внутренней калибровки: {intrinsic_params_path}")
        
        if not self._load_extrinsic_params(extrinsic_params_path):
            raise ValueError(f"Критическая ошибка: не удалось загрузить файл внешней калибровки: {extrinsic_params_path}")

    def _load_intrinsic_params(self, file_path: str) -> bool:
        """Загружает внутренние параметры камеры (матрица, дисторсия)."""
        try:
            with open(file_path, 'r') as f:
                params = yaml.safe_load(f)
            self.camera_matrix = np.array(params["camera_matrix"])
            self.dist_coeffs = np.array(params["dist_coeff"])
            logger.info(f"Внутренние параметры камеры успешно загружены из '{file_path}'.")
            return True
        except FileNotFoundError:
            logger.error(f"Файл внутренней калибровки не найден: {file_path}")
            return False
        except (KeyError, TypeError) as e:
            logger.error(f"Ошибка в структуре файла внутренней калибровки {file_path}: {e}")
            return False

    def _load_extrinsic_params(self, file_path: str) -> bool:
        """Загружает внешние параметры (трансформация Робот -> Камера)."""
        try:
            with open(file_path, 'r') as f:
                params = yaml.safe_load(f)
            
            self.transform_robot_to_camera = np.array(params["transform_robot_to_camera"])
            
            # Предварительно вычисляем обратную матрицу, так как она используется чаще
            R = self.transform_robot_to_camera[:3, :3]
            t = self.transform_robot_to_camera[:3, 3]
            R_inv = R.T
            t_inv = -R_inv @ t
            
            self.transform_camera_to_robot = np.eye(4)
            self.transform_camera_to_robot[:3, :3] = R_inv
            self.transform_camera_to_robot[:3, 3] = t_inv

            logger.info(f"Внешние параметры калибровки ('Рука-Глаз') успешно загружены из '{file_path}'.")
            return True
        except FileNotFoundError:
            logger.error(f"Файл внешней калибровки не найден: {file_path}")
            return False
        except (KeyError, TypeError) as e:
            logger.error(f"Ошибка в структуре файла внешней калибровки {file_path}: {e}")
            return False

    def pixel_to_robot_coords(self, u: int, v: int, z_robot: float = 0.0) -> np.ndarray | None:
        """
        Преобразует 2D пиксельные координаты (u, v) в 3D координаты (X, Y, Z) 
        в системе координат РОБОТА на заданной высоте Z.

        Args:
            u (int): Координата X на изображении (пиксель).
            v (int): Координата Y на изображении (пиксель).
            z_robot (float): Целевая высота Z в системе координат робота (например, высота стола).

        Returns:
            np.ndarray: Массив [X, Y, Z] в координатах робота или None, если преобразование невозможно.
        """
        if self.camera_matrix is None or self.transform_camera_to_robot is None:
            logger.error("Преобразование невозможно: параметры калибровки не загружены.")
            return None

        # Шаг 1: Устраняем дисторсию для входной точки.
        pixel_distorted = np.array([[[u, v]]], dtype=np.float32)
        pixel_undistorted = cv2.undistortPoints(pixel_distorted, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        u_corr, v_corr = pixel_undistorted[0, 0]

        # Шаг 2: Создаем точку в однородных координатах и нормализуем ее,
        # чтобы получить вектор направления в системе координат КАМЕРЫ.
        cam_fx = self.camera_matrix[0, 0]
        cam_fy = self.camera_matrix[1, 1]
        cam_cx = self.camera_matrix[0, 2]
        cam_cy = self.camera_matrix[1, 2]

        # Вектор, исходящий из центра камеры и проходящий через пиксель
        vector_in_camera_frame = np.array([(u_corr - cam_cx) / cam_fx,
                                           (v_corr - cam_cy) / cam_fy,
                                           1.0])
        
        # Нормализуем вектор (хотя это не строго обязательно для этого метода)
        vector_in_camera_frame /= np.linalg.norm(vector_in_camera_frame)

        # Шаг 3: Трансформируем начало координат камеры и вектор направления
        # из системы камеры в систему робота.
        R_cam_to_robot = self.transform_camera_to_robot[:3, :3]
        t_cam_to_robot = self.transform_camera_to_robot[:3, 3] # Это позиция камеры в мире робота

        vector_in_robot_frame = R_cam_to_robot @ vector_in_camera_frame
        camera_pos_in_robot_frame = t_cam_to_robot

        # Шаг 4: Находим точку пересечения луча с горизонтальной плоскостью z = z_robot.
        # Уравнение луча: P = camera_pos + t * vector
        # P.z = z_robot => camera_pos.z + t * vector.z = z_robot
        # t = (z_robot - camera_pos.z) / vector.z
        
        vec_z = vector_in_robot_frame[2]
        if abs(vec_z) < 1e-6:
            # Луч параллелен плоскости XY, пересечения не будет (или их бесконечно много)
            logger.warning("Луч параллелен рабочей плоскости. Невозможно найти пересечение.")
            return None

        t = (z_robot - camera_pos_in_robot_frame[2]) / vec_z

        if t < 0:
            # Пересечение находится позади камеры, что физически невозможно.
            logger.warning(f"Точка пересечения находится позади камеры (t={t:.2f}). Проверьте калибровку.")
            return None

        # Находим финальные координаты точки в мире робота
        target_point_robot = camera_pos_in_robot_frame + t * vector_in_robot_frame
        
        logger.debug(f"Pixel ({u}, {v}) -> Robot Coords ({target_point_robot[0]:.4f}, {target_point_robot[1]:.4f}, {target_point_robot[2]:.4f})")
        return target_point_robot

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Устраняет дисторсию на изображении.
        
        Args:
            frame (np.ndarray): Искаженное изображение.

        Returns:
            np.ndarray: Изображение без дисторсии.
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            logger.warning("Невозможно устранить дисторсию: данные калибровки отсутствуют.")
            return frame
            
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        
        dst = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
        
        # Обрезаем изображение, чтобы убрать черные поля
        x, y, w, h = roi
        if w > 0 and h > 0:
            dst = dst[y:y+h, x:x+w]
        
        return dst

# Пример использования удален, так как он был устаревшим и нерабочим.
# Для проверки используйте основной цикл приложения. 