"""
Тестовый скрипт для модуля Vision.
"""
import cv2
import logging
import sys
import os

# Добавляем корневую директорию проекта в Python path для корректного импорта
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем необходимые классы из проекта
from vision import Vision
from config import settings
from utils import setup_logging

def main():
    """
    Инициализирует камеру, используя модуль Vision, находит объекты
    всех заданных в конфиге цветов и отображает результат в реальном времени.
    """
    # Используем общую функцию настройки логгера
    setup_logging(name="VisionTest", level=logging.INFO)
    logger = logging.getLogger("VisionTest")

    try:
        vision_system = Vision()
        logger.info("Модуль Vision успешно инициализирован.")
    except RuntimeError as e:
        logger.critical(f"Не удалось инициализировать модуль Vision: {e}")
        return

    logger.info("Камера работает. Нажмите 'q' в окне с видео, чтобы выйти.")

    # Получаем цвета для поиска из главного файла конфигурации
    colors_to_detect = list(settings.vision.COLOR_RANGES_HSV.keys())
    if not colors_to_detect:
        logger.error("В файле конфигурации не заданы цвета для поиска. Выход.")
        return

    logger.info(f"Идет поиск следующих цветов: {', '.join(colors_to_detect)}")

    # Словарь с цветами для отрисовки рамок (BGR формат)
    display_colors = {
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0)
    }

    while True:
        # 1. Получаем кадр с камеры
        frame = vision_system.get_frame()
        if frame is None:
            logger.warning("Не удалось получить кадр. Повторная попытка...")
            continue

        display_frame = frame.copy()

        # 2. Проходим по всем цветам и ищем для каждого самый большой объект
        for color_name in colors_to_detect:
            # Ищем объект, но не просим модуль Vision рисовать (draw_debug=False)
            coords, _ = vision_system.find_object_by_color(display_frame, color_name)
            
            # Если объект найден, рисуем на кадре сами
            if coords:
                x, y = coords['x'], coords['y']
                draw_color = display_colors.get(color_name, (255, 255, 255)) # Белый по-умолчанию
                cv2.circle(display_frame, (x, y), 15, draw_color, 2)
                cv2.putText(display_frame, color_name, (x + 20, y + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)

        # 3. Отображаем итоговый кадр со всеми найденными объектами
        cv2.imshow("Vision Module Test", display_frame)

        # 4. Проверяем, не нажата ли клавиша 'q' для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Получена команда на выход.")
            break

    # 5. Освобождаем ресурсы
    vision_system.release()
    cv2.destroyAllWindows()
    logger.info("Тест успешно завершен.")

if __name__ == "__main__":
    main() 