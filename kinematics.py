import logging
import ikpy.chain
import numpy as np
from config import settings

logger = logging.getLogger(__name__)


class Kinematics:
    """
    Класс для работы с кинематикой манипулятора на основе URDF файла.
    Предоставляет методы для прямой и обратной кинематики.
    """
    def __init__(self, urdf_path: str, active_links_mask: list[bool]):
        """
        Инициализирует кинематическую цепь из URDF файла.

        Args:
            urdf_path (str): Путь к URDF файлу робота.
            active_links_mask (list[bool]): Маска, указывающая, какие звенья активны.
        """
        try:
            self.chain = ikpy.chain.Chain.from_urdf_file(
                urdf_path,
                active_links_mask=active_links_mask
            )
            logger.info(f"Кинематическая цепь успешно загружена из {urdf_path}")

            self.num_active_joints = sum(1 for active in active_links_mask if active)
            logger.info(f"Количество активных суставов: {self.num_active_joints}")
            
            # Получаем и кешируем индексы активных звеньев для использования в других методах
            self.active_link_indices = [i for i, active in enumerate(active_links_mask) if active]

        except Exception as e:
            logger.critical(f"Ошибка при загрузке URDF файла '{urdf_path}': {e}", exc_info=True)
            raise

    def forward_kinematics(self, angles_rad: list[float]) -> np.ndarray | None:
        """
        Решает прямую задачу кинематики.

        Args:
            angles_rad (list of float): Список углов для каждого активного сустава в радианах.

        Returns:
            np.ndarray: 4x4 матрица преобразования конечного эффектора или None.
        """
        if len(angles_rad) != self.num_active_joints:
            logger.error(f"Неверное количество углов. Ожидалось {self.num_active_joints}, получено {len(angles_rad)}")
            return None

        full_angles = [0.0] * len(self.chain.links)
        for i, angle in zip(self.active_link_indices, angles_rad):
            full_angles[i] = angle

        return self.chain.forward_kinematics(full_angles)

    def inverse_kinematics(self, target_position: list[float], initial_position_rad: list[float] | None = None) -> list[float] | None:
        """
        Решает обратную задачу кинематики.

        Args:
            target_position (list of float): Целевые координаты [X, Y, Z] в метрах.
            initial_position_rad (list of float, optional): Начальное положение суставов.

        Returns:
            list of float: Список углов суставов в радианах или None.
        """
        if initial_position_rad is None:
            initial_position_rad = [0.0] * self.num_active_joints

        if len(initial_position_rad) != self.num_active_joints:
             logger.error(f"Неверное количество начальных углов. Ожидалось {self.num_active_joints}, получено {len(initial_position_rad)}")
             return None

        full_initial_position = [0.0] * len(self.chain.links)
        for i, angle in zip(self.active_link_indices, initial_position_rad):
            full_initial_position[i] = angle

        ik_solution = self.chain.inverse_kinematics(
            target_position=target_position,
            initial_position=full_initial_position
        )

        active_ik_solution = [ik_solution[i] for i in self.active_link_indices]
        return active_ik_solution

    def radians_to_steps(self, angles_rad: list[float]) -> list[int] | None:
        """
        Преобразует углы в радианах в шаги для шаговых двигателей.
        
        Args:
            angles_rad (list[float]): Список углов в радианах.

        Returns:
            list[int]: Список соответствующих позиций в шагах или None.
        """
        if len(angles_rad) != len(settings.robot.STEPS_PER_RADIAN):
            logger.error(f"Несоответствие между количеством углов ({len(angles_rad)}) и "
                         f"количеством настроенных коэффициентов шагов ({len(settings.robot.STEPS_PER_RADIAN)}).")
            return None
        
        steps = [int(angle * ratio) for angle, ratio in zip(angles_rad, settings.robot.STEPS_PER_RADIAN)]
        return steps


# Пример использования
if __name__ == '__main__':
    # Создаем экземпляр класса
    kinematics_solver = Kinematics(urdf_path='manipulator.urdf', active_links_mask=[False, True, True, True, True, True, True])

    if kinematics_solver.chain:
        # --- Пример прямой кинематики ---
        # Углы для 6 суставов в радианах (например, все по 0.5 радиан)
        joint_angles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        print(f"\nТестирование прямой кинематики с углами: {joint_angles}")
        
        fk_matrix = kinematics_solver.forward_kinematics(joint_angles)
        
        if fk_matrix is not None:
            position = fk_matrix[:3, 3]
            orientation = fk_matrix[:3, :3]
            print(f"Положение конечного эффектора (X, Y, Z): {position}")
            # print(f"Матрица ориентации:\n{orientation}")

        # --- Пример обратной кинематики ---
        # Целевая точка в пространстве (в метрах)
        target_xyz = [0.1, 0.1, 0.3]
        print(f"\nТестирование обратной кинематики для цели: {target_xyz}")

        # Начальное положение для поиска решения (необязательно, но рекомендуется)
        initial_angles = [0, 0, 0, 0, 0, 0]

        ik_angles = kinematics_solver.inverse_kinematics(target_xyz, initial_position_rad=initial_angles)

        if ik_angles is not None:
            print(f"Найденные углы суставов (радианы): {np.round(np.rad2deg(ik_angles), 2)} (в градусах)")

            # Проверка: применим прямую кинематику к найденным углам
            # чтобы увидеть, попали ли мы в цель.
            validation_matrix = kinematics_solver.forward_kinematics(ik_angles)
            validation_position = validation_matrix[:3, 3]
            print(f"Положение после IK (проверка): {validation_position}")
            
            error = np.linalg.norm(np.array(target_xyz) - validation_position)
            print(f"Ошибка позиционирования: {error * 1000:.2f} мм")
        else:
            print("Решение для обратной кинематики не найдено.")
