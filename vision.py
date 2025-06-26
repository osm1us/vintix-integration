"""
This module is responsible for all computer vision tasks,
including camera handling and object detection.
"""
import cv2
import numpy as np
import logging
from config import settings

logger = logging.getLogger(__name__)


class Vision:
    """
    Отвечает за получение изображений с камеры, их обработку и поиск объектов.
    """

    def __init__(self):
        """
        Инициализирует модуль зрения, используя глобальные настройки.
        """
        self.config = settings.vision.Camera
        self.color_ranges_hsv = settings.vision.COLOR_RANGES_HSV
        self.min_contour_area = settings.vision.MIN_CONTOUR_AREA
        self.cap = None

        if not self._initialize_camera():
            raise RuntimeError("Не удалось инициализировать камеру. Проверьте ID и настройки.")
        
        logger.info("Модуль Vision успешно инициализирован.")

    def _initialize_camera(self) -> bool:
        """Инициализирует захват видео с камеры."""
        logger.info(f"Попытка подключения к камере ID: {self.config.ID} "
                    f"с разрешением {self.config.RESOLUTION_WIDTH}x{self.config.RESOLUTION_HEIGHT}")
        
        self.cap = cv2.VideoCapture(self.config.ID)
        if not self.cap.isOpened():
            logger.critical(f"Не удалось открыть камеру с ID {self.config.ID}.")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.RESOLUTION_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.RESOLUTION_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.BUFFER_SIZE)
        
        # Проверка установки параметров
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w != self.config.RESOLUTION_WIDTH or h != self.config.RESOLUTION_HEIGHT:
             logger.warning(
                f"Не удалось установить желаемое разрешение. "
                f"Запрошено: {self.config.RESOLUTION_WIDTH}x{self.config.RESOLUTION_HEIGHT}, "
                f"Установлено: {w}x{h}"
            )
        else:
            logger.info(f"Разрешение камеры успешно установлено: {w}x{h}")
            
        return True

    def get_frame(self) -> np.ndarray | None:
        """Захватывает один кадр с камеры."""
        if not self.cap.isOpened():
            logger.error("Попытка получить кадр, но камера не инициализирована.")
            return None
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Не удалось захватить кадр с камеры.")
            return None
        return frame

    def find_object_by_color(self, frame: np.ndarray, color_name: str, draw_debug: bool = False) -> tuple[dict | None, np.ndarray]:
        """
        Находит центр самого большого объекта заданного цвета на изображении.

        Args:
            frame (np.ndarray): Кадр для анализа.
            color_name (str): Название цвета (ключ в self.color_ranges_hsv).
            draw_debug (bool): Если True, рисует контуры на возвращаемом кадре.

        Returns:
            tuple[dict | None, np.ndarray]:
                - Словарь с координатами {'x': int, 'y': int} или None, если объект не найден.
                - Кадр (оригинальный или с отладочной информацией).
        """
        if color_name not in self.color_ranges_hsv:
            logger.warning(f"Цвет '{color_name}' не найден в конфигурации.")
            return None, frame

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Объединяем маски, если для цвета задано несколько диапазонов (например, для красного)
        color_masks = []
        for (lower, upper) in self.color_ranges_hsv[color_name]:
            color_masks.append(cv2.inRange(hsv_frame, lower, upper))
        
        mask = color_masks[0]
        for i in range(1, len(color_masks)):
            mask = cv2.bitwise_or(mask, color_masks[i])

        # Улучшение маски
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, frame

        largest_contour = max(contours, key=cv2.contourArea)
        
        # Проверяем, что площадь контура больше минимального порога
        if cv2.contourArea(largest_contour) < self.min_contour_area:
            return None, frame

        M = cv2.moments(largest_contour)
        coords = None
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            coords = {'x': cx, 'y': cy}
            logger.debug(f"Объект цвета '{color_name}' найден в координатах ({cx}, {cy}).")
        
        output_frame = frame
        if draw_debug:
            output_frame = frame.copy()
            cv2.drawContours(output_frame, [largest_contour], -1, (0, 255, 0), 2)
            if coords:
                cv2.circle(output_frame, (coords['x'], coords['y']), 7, (0, 0, 255), -1)
            
        return coords, output_frame

    @staticmethod
    def display_frame(frame: np.ndarray, window_name: str = "Vision"):
        """Отображает кадр в окне."""
        if frame is not None:
            cv2.imshow(window_name, frame)

    def release(self):
        """Освобождает ресурс камеры."""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Ресурс камеры освобожден.")