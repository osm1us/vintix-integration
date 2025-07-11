"""
Main executable file for the Vintix-powered robot control loop.
Orchestrates all modules to run the robot.
"""
import logging
import time
import cv2
import numpy as np
from enum import Enum, auto

from config import settings
from vision import Vision
from vintix_agent import VintixAgent
from robot_controller import RobotController
from coordinate_mapper import CoordinateMapper
from voice_control import VoiceControl, Command
from utils import setup_logging, GracefulShutdown
from datalogger import HDF5Logger

setup_logging(name="VintixRunner", level=settings.system.LOG_LEVEL)
logger = logging.getLogger(__name__)


class RobotState(Enum):
    """Состояния конечного автомата для управления роботом."""
    IDLE = auto()               # Ожидание команд
    PICK_PLACE_PLANNING = auto() # Поиск цели и планирование следующего шага
    PICK_PLACE_MOVING = auto()   # Ожидание завершения движения робота
    
    # --- Новые состояния для неблокирующей последовательности захвата ---
    GRASP_START = auto()         # Начало последовательности захвата
    GRASP_CLOSING = auto()       # Ожидание закрытия захвата
    GRASP_RAISING = auto()       # Ожидание подъема объекта
    GRASP_GOING_HOME = auto()    # Ожидание возвращения домой
    GRASP_OPENING = auto()       # Ожидание открытия захвата (финал)


class VintixRunner:
    """
    Главный класс, который инициализирует все компоненты и управляет основным циклом.
    """

    def __init__(self):
        logger.info("Инициализация Vintix Corobot...")
        self.shutdown_manager = GracefulShutdown()
        
        try:
            # --- Инициализация компонентов ---
            self.robot_controller = RobotController()
            self.vision = Vision()
            self.agent = VintixAgent(
                model_path=settings.agent.MODEL_PATH
            )
            self.mapper = CoordinateMapper(
                work_plane_z=settings.vision.TARGET_Z_COORD_M
            )
            self.voice_control = VoiceControl(
                model_path=settings.voice.MODEL_PATH
            )
            self.datalogger = HDF5Logger(
                log_dir=settings.datalogger.LOG_DIR
            )
            
            # --- Инициализация конечного автомата ---
            self.state = RobotState.IDLE
            self.running = True
            self.pick_place_task = {} # Словарь для хранения параметров текущей задачи

            logger.info("Все компоненты успешно инициализированы.")

        except Exception as e:
            logger.critical(f"Критическая ошибка при инициализации: {e}", exc_info=True)
            self.running = False

    def run(self):
        """
        Запускает основной рабочий цикл.
        """
        if not self.running:
            logger.error("Запуск невозможен из-за ошибки инициализации.")
            return

        logger.info("Vintix Corobot запущен. Запуск фонового прослушивания...")
        self.voice_control.start()
        
        try:
            while self.running and not self.shutdown_manager.is_shutting_down():
                # --- Шаг 0: Обновление состояний ---
                self.robot_controller.update_state() # Важно: обновляем состояние робота в начале каждого цикла

                # --- Шаг 1: Получение данных извне ---
                frame = self.vision.get_frame()
                if frame is None:
                    logger.warning("Не удалось получить кадр, пропуск итерации.")
                    time.sleep(0.5)
                    continue

                voice_command = self.voice_control.get_command()

                # --- Шаг 2: Обработка команд и обновление состояния ---
                if voice_command:
                    self._handle_voice_command(voice_command)

                # --- Шаг 3: Выполнение действий в зависимости от состояния ---
                if self.state == RobotState.PICK_PLACE_PLANNING:
                    frame = self._handle_pick_place_planning(frame)
                elif self.state == RobotState.PICK_PLACE_MOVING:
                    frame = self._handle_pick_place_moving(frame)
                elif self.state in [
                    RobotState.GRASP_START, RobotState.GRASP_CLOSING, 
                    RobotState.GRASP_RAISING, RobotState.GRASP_GOING_HOME,
                    RobotState.GRASP_OPENING
                ]:
                    self._handle_grasp_sequence()

                # --- Шаг 4: Отображение и задержка ---
                self.vision.display_frame(frame, "Vintix Control")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Нажата клавиша 'q', завершение работы.")
                    self.running = False
                
                time.sleep(0.05) # Небольшая задержка, чтобы не загружать CPU

        finally:
            self.shutdown()

    def _handle_voice_command(self, command: dict):
        """Обрабатывает команды из голосового модуля и меняет состояние."""
        command_type = command.get('type')

        if command_type == Command.PICK_UP:
            if self.state == RobotState.IDLE:
                target_color = command.get('data')
                logger.info(f"Получена команда на захват объекта цвета: {target_color}. Переход в состояние планирования.")
                self._start_pick_place_task(target_color)
            else:
                logger.warning("Получена команда на захват, но робот уже занят. Команда проигнорирована.")

        elif command_type == Command.GO_HOME:
            logger.info("Выполняется команда 'домой'.")
            if self.state == RobotState.PICK_PLACE_MOVING or self.state == RobotState.PICK_PLACE_PLANNING:
                logger.warning("Задача прервана командой 'домой'.")
            self.state = RobotState.IDLE
            self.robot_controller.go_home()
        
        elif command_type == Command.STOP:
            logger.info("Получена команда 'стоп', завершение работы.")
            self.running = False

    def _start_pick_place_task(self, target_color: str):
        """Инициализирует новую задачу 'взять и положить'."""
        self.state = RobotState.PICK_PLACE_PLANNING
        self.agent.reset()
        self.datalogger.reset_episode_buffer()

        self.pick_place_task = {
            "target_color": target_color,
            "step": 0,
            "last_reward": 0.0,
            "last_action_delta": np.zeros(3),
            "target_world_coords": None,
            "next_joint_angles": None
        }

    def _handle_pick_place_planning(self, frame: np.ndarray) -> np.ndarray:
        """
        Выполняет один шаг планирования: поиск цели, получение действия от агента
        и отправка команды на движение.
        """
        task = self.pick_place_task
        step = task["step"]
        
        # Найти объект и получить его 3D координаты
        pixel_coords, debug_frame = self.vision.find_object_by_color(frame, task["target_color"], draw_debug=True)
        
        if pixel_coords:
            target_world_coords = self.mapper.pixel_to_world(pixel_coords['x'], pixel_coords['y'])
            task["target_world_coords"] = target_world_coords
        else:
            logger.warning(f"На шаге {step} не удалось найти цель цвета {task['target_color']}.")
            target_world_coords = None

        # Получаем текущее состояние робота
        current_angles_rad = self.robot_controller.get_current_angles_rad()

        # Формируем вектор наблюдения (только проприоцепция).
        observation = np.array(current_angles_rad)
        
        # Получение действия от агента
        action_delta = self.agent.get_next_action(observation, task["last_reward"])
        task["last_action_delta"] = action_delta  # Сохраняем действие для логирования
        
        # Применяем дельту к текущим углам
        new_target_angles = np.array(current_angles_rad)
        new_target_angles[:3] += action_delta
        
        # Отправляем команду на движение, но НЕ ждем ее выполнения
        self.robot_controller.move_to_angles_rad(list(new_target_angles))
        
        # Сохраняем целевые углы для последующего обновления состояния
        task["next_joint_angles"] = list(new_target_angles)
        
        # Переключаем состояние на ожидание движения
        self.state = RobotState.PICK_PLACE_MOVING
        logger.debug(f"Шаг {step}: команда на движение отправлена, переход в состояние MOVING.")

        return debug_frame

    def _handle_pick_place_moving(self, frame: np.ndarray) -> np.ndarray:
        """
        Ожидает завершения движения робота, затем оценивает результат.
        """
        if self.robot_controller.is_moving():
            # Движение еще не завершено, ничего не делаем
            return frame

        # Движение завершено, можно оценивать результат
        # Явно обновляем состояние контроллера, чтобы получить актуальные углы
        self.robot_controller.update_state()
        
        task = self.pick_place_task
        step = task["step"]
        logger.debug(f"Шаг {step}: движение завершено, оценка результата.")

        # 1. Получаем новую позицию эффектора
        end_effector_pos = self.robot_controller.get_end_effector_position()
        if end_effector_pos is None:
            logger.error("Не удалось получить позицию эффектора, прерывание задачи.")
            self.state = RobotState.IDLE
            return frame

        # 2. Вычисляем вознаграждение и проверяем условие успеха
        is_success = False
        target_world_coords = task.get("target_world_coords")

        if target_world_coords is not None:
            distance_to_target = np.linalg.norm(target_world_coords - end_effector_pos)
            task["last_reward"] = -distance_to_target
            
            if distance_to_target < settings.agent.episode.SUCCESS_THRESHOLD:
                is_success = True
                task["last_reward"] = 1.0
        else:
            task["last_reward"] = -0.1

        # 3. Логируем шаг
        observation = np.array(task["next_joint_angles"])
        
        self.datalogger.log_step(
            observation, 
            task["last_action_delta"],  # Используем сохраненное действие
            task["last_reward"], 
            step
        )
        
        # 4. Обработка завершения эпизода
        task["step"] += 1
        if is_success:
            logger.info(f"УСПЕХ! Эпизод завершен на шаге {task['step']}.")
            self.datalogger.finish_episode(final_reward=1.0)
            self.state = RobotState.GRASP_START # Запускаем неблокирующую последовательность
        elif task["step"] >= settings.agent.episode.MAX_STEPS:
            logger.warning(f"ПРОВАЛ. Эпизод не завершился за {settings.agent.episode.MAX_STEPS} шагов.")
            self.datalogger.finish_episode(final_reward=-1.0)
            self.state = RobotState.IDLE
        else:
            # Если эпизод продолжается, возвращаемся к планированию
            self.state = RobotState.PICK_PLACE_PLANNING
        
        return frame
    
    def _handle_grasp_sequence(self):
        """
        Обрабатывает шаги неблокирующей последовательности захвата,
        проверяя состояние и переключая его по завершении каждого шага.
        """
        if self.robot_controller.is_moving():
            # Если робот все еще движется, мы просто ждем следующей итерации цикла.
            return

        # Если робот не движется, значит, предыдущая команда завершена, и можно выполнять следующую.
        
        if self.state == RobotState.GRASP_START:
            logger.info("Начало последовательности захвата: закрытие захвата.")
        self.robot_controller.close_gripper()
            # Даем небольшую задержку для сервопривода, так как он не отслеживается is_moving()
            # Это единственная допустимая короткая задержка.
            time.sleep(settings.robot.gripper.ACTION_DELAY_SEC)
            self.state = RobotState.GRASP_CLOSING
            # Сразу переходим к следующему состоянию, не дожидаясь здесь.
            
        elif self.state == RobotState.GRASP_CLOSING:
            logger.info("Шаг захвата: подъем объекта.")
        current_pos = self.robot_controller.get_end_effector_position()
        if current_pos is not None:
            target_pos = current_pos + np.array([0, 0, 0.05]) # Поднять на 5 см
            ik_solution = self.robot_controller.calculate_ik(target_pos.tolist())
            if ik_solution:
                self.robot_controller.move_to_angles_rad(ik_solution)
                    self.state = RobotState.GRASP_RAISING
                else:
                    logger.error("Не удалось рассчитать IK для подъема, перехожу к движению домой.")
                    self.state = RobotState.GRASP_GOING_HOME
            else:
                logger.error("Не удалось получить позицию для подъема, перехожу к движению домой.")
                self.state = RobotState.GRASP_GOING_HOME
        
        elif self.state == RobotState.GRASP_RAISING:
            logger.info("Шаг захвата: возвращение домой.")
        self.robot_controller.go_home()
            self.state = RobotState.GRASP_GOING_HOME

        elif self.state == RobotState.GRASP_GOING_HOME:
            logger.info("Шаг захвата: открытие захвата.")
        self.robot_controller.open_gripper()
            time.sleep(settings.robot.gripper.ACTION_DELAY_SEC) # Задержка для серво
            self.state = RobotState.GRASP_OPENING
            
        elif self.state == RobotState.GRASP_OPENING:
            logger.info("Последовательность захвата успешно завершена.")
            self.state = RobotState.IDLE # Финальное состояние - ожидание


    def shutdown(self):
        """
        Освобождает все ресурсы.
        """
        logger.info("Завершение работы Vintix Corobot...")
        if hasattr(self, 'voice_control'):
            self.voice_control.stop()
        if hasattr(self, 'robot_controller'):
            self.robot_controller.shutdown()
        if hasattr(self, 'vision'):
            self.vision.release()
        
        cv2.destroyAllWindows()
        logger.info("Все ресурсы освобождены. Выход.")


if __name__ == '__main__':
    runner = VintixRunner()
    if runner.running:
        runner.run()