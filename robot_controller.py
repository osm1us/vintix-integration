"""
Высокоуровневый контроллер для Corobot.
Оркестрирует модули кинематики и управления двигателями.
"""

import logging
import numpy as np
from config import settings
from kinematics import Kinematics
from motor_control import MotorController

logger = logging.getLogger(__name__)


class RobotController:
    """
    Оркестрирует движения робота, преобразуя высокоуровневые команды
    в низкоуровневые сигналы для двигателей.
    """

    def __init__(self):
        """
        Инициализирует контроллер робота, используя глобальные настройки.
        """
        logger.info("Инициализация RobotController...")
        try:
            robot_settings = settings.robot
            self.kinematics = Kinematics(
                urdf_path=robot_settings.URDF_PATH,
                active_links_mask=robot_settings.ACTIVE_LINKS_MASK
            )
            logger.info("Модуль кинематики инициализирован.")
        except Exception as e:
            logger.critical(f"Не удалось инициализировать Kinematics: {e}", exc_info=True)
            raise

        self.motor_controller = MotorController()
        if not self.motor_controller.check_connection():
            logger.warning("Не удалось установить соединение с контроллером двигателя при инициализации.")
        else:
            logger.info("MotorController инициализирован, соединение проверено.")

        self.num_joints = self.kinematics.num_active_joints
        self.current_angles_rad = [0.0] * self.num_joints
        
        # Перемещаем робота в домашнюю позицию при старте
        self.go_home()

    def get_current_angles_rad(self) -> list[float]:
        """
        Возвращает последние заданные углы суставов.
        ВНИМАНИЕ: Это последнее заданное состояние, а не данные с энкодеров.
        """
        return self.current_angles_rad

    def set_current_angles_rad(self, joint_angles_rad: list[float]):
        """
        Принудительно устанавливает внутреннее состояние углов робота.
        Вызывать только после подтверждения завершения движения!
        """
        if len(joint_angles_rad) != self.num_joints:
            logger.error(f"Попытка установить неверное количество углов. "
                         f"Ожидалось {self.num_joints}, получено {len(joint_angles_rad)}.")
            return
        self.current_angles_rad = joint_angles_rad

    def is_moving(self) -> bool:
        """
        Проверяет, движется ли робот в данный момент.
        """
        return self.motor_controller.is_moving()

    def get_end_effector_position(self) -> np.ndarray | None:
        """
        Возвращает текущее положение конечного эффектора (XYZ) на основе последних заданных углов.
        """
        fk_matrix = self.kinematics.forward_kinematics(self.current_angles_rad)
        if fk_matrix is not None:
            # Позиция (трансляция) находится в последнем столбце матрицы
            return fk_matrix[:3, 3]
        
        logger.error("Не удалось рассчитать положение конечного эффектора.")
        return None

    def move_to_angles_rad(self, joint_angles_rad: list[float]) -> bool:
        """
        Перемещает суставы робота в указанные углы (в радианах).

        Args:
            joint_angles_rad (list[float]): Список целевых углов для активных суставов.

        Returns:
            bool: True, если команда на движение была успешно отправлена.
        """
        if len(joint_angles_rad) != self.num_joints:
            logger.error(f"Неверное количество углов. Ожидалось {self.num_joints}, получено {len(joint_angles_rad)}.")
            return False

        try:
            steps = self.kinematics.radians_to_steps(joint_angles_rad)
            if steps is None:
                logger.error("Не удалось преобразовать радианы в шаги.")
                return False

            logger.info(f"Преобразованы радианы {joint_angles_rad} в шаги {steps}.")

            success = self.motor_controller.move_joints_by_steps(steps)
            if success:
                logger.info("Команда на движение успешно отправлена.")
                # ВНИМАНИЕ: Состояние self.current_angles_rad здесь больше не обновляется.
                # Это должен делать вызывающий код после подтверждения завершения движения.
            else:
                logger.error("Не удалось отправить команду на движение.")
            return success

        except Exception as e:
            logger.error(f"Произошла ошибка при попытке движения: {e}", exc_info=True)
            return False

    def set_gripper(self, angle: int) -> bool:
        """
        Устанавливает захват в определенный угол.

        Args:
            angle (int): Целевой угол (0-180).

        Returns:
            bool: True, если команда была отправлена успешно.
        """
        logger.info(f"Установка захвата в угол {angle}.")
        return self.motor_controller.set_gripper_angle(angle)

    def open_gripper(self) -> bool:
        """Полностью открывает захват."""
        angle = settings.robot.gripper.OPEN_ANGLE
        logger.info(f"Открытие захвата (угол {angle}).")
        return self.set_gripper(angle)

    def close_gripper(self) -> bool:
        """Полностью закрывает захват."""
        angle = settings.robot.gripper.CLOSED_ANGLE
        logger.info(f"Закрытие захвата (угол {angle}).")
        return self.set_gripper(angle)
    
    def go_home(self) -> bool:
        """
        Перемещает робота в домашнюю позицию, определенную в настройках.
        """
        logger.info("Отправка робота в домашнюю позицию.")
        home_angles_rad = settings.robot.HOME_ANGLES_RAD
        if len(home_angles_rad) != self.num_joints:
            logger.error(f"Конфигурация HOME_ANGLES_RAD ({len(home_angles_rad)} углов) "
                         f"не соответствует количеству суставов робота ({self.num_joints}).")
            return False
        return self.move_to_angles_rad(home_angles_rad)

    def calculate_ik(self, target_position: list[float]) -> list[float] | None:
        """
        Решает обратную задачу кинематики для заданной цели.
        Является оберткой над модулем кинематики для улучшения инкапсуляции.
        
        Args:
            target_position (list[float]): Целевые координаты [X, Y, Z].

        Returns:
            list[float] | None: Список углов суставов в радианах или None.
        """
        logger.info(f"Расчет обратной кинематики для цели: {target_position}")
        initial_angles = self.get_current_angles_rad()
        
        return self.kinematics.inverse_kinematics(
            target_position=target_position,
            initial_position_rad=initial_angles
        )

    def shutdown(self):
        """Безопасно завершает работу с контроллером."""
        logger.info("Завершение работы RobotController.")
        self.motor_controller.close_session() 