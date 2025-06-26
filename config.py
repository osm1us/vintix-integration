"""
Центральный конфигурационный файл проекта Vintix Corobot.

Этот файл должен быть единственным источником истины для всех настраиваемых
параметров в проекте. Использование классов помогает структурировать
конфигурацию и обеспечивает удобный доступ через точки.

Пример использования:
from config import settings
print(settings.robot.urdf_path)
print(settings.vision.camera.resolution_width)
"""
import logging
import numpy as np


class SystemSettings:
    """Общие системные настройки."""
    LOG_LEVEL: int = logging.INFO
    # Путь к файлу для сохранения траектории калибровки гомографии
    HOMOGRAPHY_PATH: str = "homography_matrix.json"
    # Путь к файлу с параметрами внутренней калибровки камеры
    CAMERA_PARAMS_PATH: str = "camera_params.yaml"


class RobotSettings:
    """Настройки, связанные с роботом-манипулятором."""
    # Путь к URDF-файлу, который является основным описанием робота
    URDF_PATH: str = "manipulator.urdf"
    # Имя конечного звена (эффектора) в URDF
    END_EFFECTOR_LINK: str = "gripper_link"
    
    # Конфигурация для ikpy. Маска активных звеньев.
    # Длина должна соответствовать количеству <joint> в URDF.
    # Основываясь на manipulator.urdf, у нас 6 активных суставов.
    # [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6]
    # base_link не имеет активного сустава, поэтому он False.
    # В URDF 7 звеньев, но ikpy смотрит на суставы.
    ACTIVE_LINKS_MASK: list[bool] = [False, True, True, True, True, True, True]
    
    # Домашняя позиция робота в радианах.
    # Длина должна соответствовать количеству True в ACTIVE_LINKS_MASK.
    HOME_ANGLES_RAD: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Физические параметры, необходимые для преобразования радиан в шаги.
    # ЗАМЕНИТЬ НА РЕАЛЬНЫЕ ЗНАЧЕНИЯ ВАШЕГО РОБОТА!
    # (шаги на оборот * передаточное число) / (2 * PI)
    STEPS_PER_RADIAN: list[float] = [
        88.88,  # Пример для joint_1
        88.88,  # Пример для joint_2
        88.88,  # Пример для joint_3
        88.88,  # Пример для joint_4
        88.88,  # Пример для joint_5
        88.88,  # Пример для joint_6
    ]

    class Gripper:
        """Настройки для захвата."""
        # Углы для сервопривода
        OPEN_ANGLE: int = 180
        CLOSED_ANGLE: int = 60
        # Скорости (будут использоваться в прошивке)
        OPEN_SPEED: float = 0.5
        CLOSE_SPEED: float = 1.0
        # Задержка после действия с захватом или возвращения домой (в секундах)
        ACTION_DELAY_SEC: float = 1.0


class ControllerSettings:
    """Настройки для контроллера (ESP32)."""
    IP: str = "192.168.1.10"  # IP-адрес ESP32
    PORT: int = 80
    TIMEOUT: int = 3  # Таймаут запроса в секундах
    RETRIES: int = 2  # Количество попыток переподключения


class VisionSettings:
    """Настройки для системы компьютерного зрения."""
    class Camera:
        ID: int = 0
        RESOLUTION_WIDTH: int = 1280
        RESOLUTION_HEIGHT: int = 720
        BUFFER_SIZE: int = 1

    # Минимальная площадь контура, чтобы он считался объектом
    MIN_CONTOUR_AREA: int = 100

    # При преобразовании 2D-координат цели в 3D, эта Z-координата будет использоваться.
    # Это упрощение, в идеале Z-координата должна определяться из 3D-зрения.
    TARGET_Z_COORD_M: float = 0.05  # 5 см над столом

    # Цветовые диапазоны HSV для обнаружения объектов
    # Формат: "имя": [(нижняя_граница_hsv), (верхняя_граница_hsv)]
    # Для красного цвета используется два диапазона.
    COLOR_RANGES_HSV: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        'red': [
            (np.array([0, 120, 70]), np.array([10, 255, 255])),
            (np.array([170, 120, 70]), np.array([180, 255, 255]))
        ],
        'green': [
            (np.array([35, 80, 40]), np.array([85, 255, 255]))
        ],
        'blue': [
            (np.array([90, 80, 40]), np.array([130, 255, 255]))
        ],
    }


class AgentSettings:
    """Настройки для агента Vintix."""
    MODEL_PATH: str = "models/vintix_checkpoint"
    # Длина контекста (истории), подаваемой в модель
    CONTEXT_LENGTH: int = 10
    # Имя задачи из metadata.json. ДОЛЖНО БЫТЬ ЗАМЕНЕНО НА РЕАЛЬНОЕ!
    TASK_NAME: str = "xarm_pick_place_task_name_placeholder"

    class Episode:
        """Параметры одного эпизода (попытки)."""
        # Максимальное количество шагов в эпизоде
        MAX_STEPS: int = 200
        # Порог успеха: если эффектор ближе этого расстояния (в метрах), эпизод успешен
        SUCCESS_THRESHOLD: float = 0.02  # 2 см


class VoiceSettings:
    """Настройки для голосового управления."""
    MODEL_PATH: str = "models/vosk/vosk-model-small-ru-0.22"
    SAMPLE_RATE: int = 16000
    # Ключевые слова можно вынести в отдельный YAML, но для простоты оставим здесь
    KEYWORDS: dict[str, dict[str, str]] = {
        'action': {
            'захвати': 'grab', 'возьми': 'grab', 'подбери': 'grab', 'подними': 'grab',
            'схвати': 'grab', 'хватай': 'grab', 'взять': 'grab', 'поднять': 'grab'
        },
        'target_color': {
            'красный': 'red', 'красного': 'red', 'зелёный': 'green', 'зеленый': 'green',
            'синий': 'blue', 'синего': 'blue'
        },
        'command': {
            'стоп': 'stop', 'остановись': 'stop', 'отмена': 'stop',
            'домой': 'home'
        }
    }


class DataLoggerSettings:
    """Настройки для логгера данных."""
    LOG_DIR: str = "data/training_logs"


# Создаем единый объект настроек для импорта в других частях проекта
class Settings:
    system = SystemSettings()
    robot = RobotSettings()
    controller = ControllerSettings()
    vision = VisionSettings()
    agent = AgentSettings()
    voice = VoiceSettings()
    datalogger = DataLoggerSettings()

settings = Settings()