"""
Module for low-level motor control.
Handles the communication with the ESP32 bridge to send commands
to the Arduino Nano controlling the stepper motors and servo.
"""
import requests
import logging
import time
from config import settings

logger = logging.getLogger(__name__)


class MotorController:
    """
    Управляет отправкой команд на контроллер ESP32 по Wi-Fi.
    """

    def __init__(self):
        """
        Инициализирует контроллер, используя настройки из config.py.
        """
        controller_settings = settings.controller
        self.ip_address = controller_settings.IP
        self.port = controller_settings.PORT
        self.timeout = controller_settings.TIMEOUT
        self.retries = controller_settings.RETRIES
        
        self.api_endpoint = f"http://{self.ip_address}:{self.port}/api/command"
        self.ping_endpoint = f"http://{self.ip_address}:{self.port}/ping"
        self.status_endpoint = f"http://{self.ip_address}:{self.port}/api/status"

        self.session = requests.Session()
        self._is_expecting_movement = False # Флаг, показывающий, что мы отправили команду и ожидаем движения
        logger.info(f"MotorController инициализирован для работы с {self.api_endpoint}")

    def _send_request(self, payload: dict) -> bool:
        """
        Внутренний метод для отправки JSON-запроса на ESP32 с логикой повторных попыток.
        """
        for attempt in range(self.retries + 1):
            try:
                response = self.session.post(self.api_endpoint, json=payload, timeout=self.timeout)
                response.raise_for_status()  # Вызовет исключение для статусов 4xx/5xx

                response_json = response.json()
                if response_json.get("status") == "success":
                    logger.debug(f"Команда {payload.get('command')} успешно выполнена: {response_json.get('message')}")
                    return True
                else:
                    error_message = response_json.get('error', 'Неизвестная ошибка от контроллера')
                    logger.error(f"Контроллер вернул ошибку для команды {payload.get('command')}: {error_message}")
                    return False

            except requests.exceptions.RequestException as e:
                logger.warning(f"Попытка {attempt + 1} из {self.retries + 1}: не удалось отправить команду на {self.api_endpoint}. Ошибка: {e}")
                if attempt == self.retries:
                    logger.critical(f"Не удалось подключиться к контроллеру после {self.retries + 1} попыток.")
                    return False
        return False

    def check_connection(self) -> bool:
        """
        Проверяет соединение с ESP32, отправляя ping-запрос.
        """
        logger.debug(f"Проверка соединения с {self.ping_endpoint}...")
        try:
            response = requests.get(self.ping_endpoint, timeout=self.timeout)
            if response.status_code == 200 and response.text == "pong":
                logger.info("Соединение с ESP32 успешно установлено.")
                return True
            else:
                logger.warning(f"Получен неожиданный ответ от {self.ping_endpoint}: {response.text}")
                return False
        except requests.exceptions.RequestException:
            logger.error(f"Не удалось подключиться к ESP32 по адресу {self.ping_endpoint}.")
            return False

    def is_moving(self) -> bool:
        """
        Проверяет, движется ли робот в данный момент, опрашивая эндпоинт статуса.

        Returns:
            bool: True, если хотя бы один мотор движется; False в противном случае.
                  В случае ошибки связи возвращает False.
        """
        if not self._is_expecting_movement:
            return False # Если мы не отправляли команду на движение, то и проверять нечего

        try:
            # Используем более короткий таймаут для частых опросов статуса
            response = self.session.get(self.status_endpoint, timeout=self.timeout * 0.5)
            response.raise_for_status()
            status_data = response.json()
            
            motors_status = status_data.get("motors", [])
            if not isinstance(motors_status, list):
                logger.warning(f"Получен некорректный формат статуса моторов: {motors_status}, считаем движение завершенным.")
                self._is_expecting_movement = False
                return False

            is_any_motor_moving = any(
                isinstance(motor, dict) and motor.get("moving", False)
                for motor in motors_status
            )

            if not is_any_motor_moving:
                self._is_expecting_movement = False # Сбрасываем флаг, движение завершено
            
            return is_any_motor_moving

        except requests.exceptions.RequestException as e:
            logger.warning(f"Не удалось получить статус с контроллера: {e}. "
                           "Считаем, что движение прекращено из-за потери связи.")
            self._is_expecting_movement = False
            return False
        except (ValueError, KeyError) as e:
            logger.warning(f"Не удалось разобрать JSON ответа статуса: {e}. "
                           "Считаем, что движение завершено.")
            self._is_expecting_movement = False
            return False

    def move_joints_by_steps(self, steps: list[int]) -> bool:
        """
        Отправляет команду на перемещение шаговых двигателей в абсолютные позиции (шаги).

        Args:
            steps (list[int]): Список абсолютных позиций в шагах для каждого сустава.
        
        Returns:
            bool: True, если команда отправлена успешно.
        """
        command = "move_steppers"
        # Прошивка ожидает параметры с именами p1, p2, p3...
        params = {f"p{i+1}": step for i, step in enumerate(steps)}
        payload = {"command": command, "params": params}
        
        logger.info(f"Отправка команды '{command}' с параметрами: {params}")
        success = self._send_request(payload)
        if success:
            self._is_expecting_movement = True # Взводим флаг ожидания движения
        return success

    def set_gripper_angle(self, angle: int) -> bool:
        """
        Отправляет команду для установки угла сервопривода захвата.

        Args:
            angle (int): Целевой угол для захвата (например, 0-180).

        Returns:
            bool: True, если команда отправлена успешно.
        """
        command = "set_gripper"
        params = {"angle": angle}
        payload = {"command": command, "params": params}

        logger.info(f"Отправка команды '{command}' с параметрами: {params}")
        return self._send_request(payload)

    def close_session(self):
        """
        Закрывает сессию requests.
        """
        self.session.close()
        logger.info("Сессия MotorController закрыта.") 