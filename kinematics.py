import ikpy.chain
import numpy as np

class Kinematics:
    """
    Класс для работы с кинематикой манипулятора на основе URDF файла.
    Предоставляет методы для прямой и обратной кинематики.
    """
    def __init__(self, urdf_file_path='manipulator.urdf'):
        """
        Инициализирует кинематическую цепь из URDF файла.

        Args:
            urdf_file_path (str): Путь к URDF файлу робота.
        """
        try:
            # Создаем маску, которая явно указывает, какие звенья активны.
            # Длина маски должна точно совпадать с количеством <link> в URDF (7).
            # [base_link, link_1, link_2, link_3, link_4, link_5, gripper_link]
            active_links_mask = [False, True, True, True, True, True, True]

            self.chain = ikpy.chain.Chain.from_urdf_file(
                urdf_file_path,
                active_links_mask=active_links_mask
            )
            print(f"Кинематическая цепь успешно загружена из {urdf_file_path}")
            # Используем новое свойство для получения имен активных звеньев
            active_links_names = [self.chain.links[i].name for i in self.active_link_indices]
            print(f"Активные звенья: {active_links_names}")
        except Exception as e:
            print(f"Ошибка при загрузке URDF файла: {e}")
            self.chain = None
            # Инициализируем кеш даже в случае ошибки, чтобы избежать проблем
            self._active_link_indices_cache = None

    @property
    def active_link_indices(self):
        """
        Возвращает и кэширует индексы активных звеньев (тех, у которых есть сустав).
        Использует hasattr для безопасной проверки, избегая ошибок с OriginLink.
        """
        # Проверяем, есть ли у нас уже кэшированный результат
        if hasattr(self, '_active_link_indices_cache') and self._active_link_indices_cache is not None:
            return self._active_link_indices_cache

        if not self.chain:
            self._active_link_indices_cache = []
            return self._active_link_indices_cache

        # Финальная, единственно верная логика, основанная на исходном коде ikpy.
        # Мы берем `self.chain.active_links_mask` (который мы сами же и задаем)
        # и находим в нем индексы всех активных звеньев (где значение True).
        # np.where возвращает кортеж, поэтому берем первый элемент.
        indices = np.where(self.chain.active_links_mask)[0].tolist()
        self._active_link_indices_cache = indices
        return indices

    def forward_kinematics(self, angles):
        """
        Решает прямую задачу кинематики.

        Args:
            angles (list of float): Список углов для каждого сустава в радианах.
                                   Длина списка должна соответствовать количеству активных суставов.
                                   Первый элемент списка - для первого сустава (base_link), и т.д.

        Returns:
            np.ndarray: 4x4 матрица преобразования (положение и ориентация) конечного эффектора.
                        Возвращает None в случае ошибки.
        """
        if not self.chain:
            print("Ошибка: кинематическая цепь не инициализирована.")
            return None

        # ikpy ожидает, что в массиве углов будет значение для каждого звена,
        # включая пассивные. Мы добавляем 0 для всех звеньев.
        full_angles = [0] * len(self.chain.links)
        
        # Заменяем дублирующийся код на вызов свойства
        active_link_indices = self.active_link_indices

        if len(angles) != len(active_link_indices):
            print(f"Ошибка: ожидалось {len(active_link_indices)} углов, но получено {len(angles)}.")
            return None

        for i, angle in zip(active_link_indices, angles):
            full_angles[i] = angle

        return self.chain.forward_kinematics(full_angles)

    def inverse_kinematics(self, target_position, initial_position=None):
        """
        Решает обратную задачу кинематики.

        Args:
            target_position (list of float): Целевые координаты [X, Y, Z] в метрах.
            initial_position (list of float, optional): Начальное положение суставов в радианах.
                                                     Используется для более быстрого и точного решения.
                                                     Если None, используется нулевое положение.

        Returns:
            list of float: Список углов суставов в радианах для достижения цели.
                           Возвращает None в случае ошибки или если решение не найдено.
        """
        if not self.chain:
            print("Ошибка: кинематическая цепь не инициализирована.")
            return None

        # Используем свойство для получения активных индексов
        active_link_indices = self.active_link_indices
        
        if initial_position is None:
            # Создаем начальное положение (все суставы в 0)
            # Упрощаем код, используя длину списка индексов
            num_active_joints = len(active_link_indices)
            initial_position = [0.0] * num_active_joints
        
        # ikpy ожидает, что в initial_position будет значение для каждого звена
        full_initial_position = [0] * len(self.chain.links)
        for i, angle in zip(active_link_indices, initial_position):
            full_initial_position[i] = angle

        ik_solution = self.chain.inverse_kinematics(
            target_position=target_position,
            initial_position=full_initial_position
        )

        # Возвращаем только углы для активных суставов
        active_ik_solution = [ik_solution[i] for i in active_link_indices]
        return active_ik_solution


# Пример использования
if __name__ == '__main__':
    # Создаем экземпляр класса
    kinematics_solver = Kinematics()

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

        ik_angles = kinematics_solver.inverse_kinematics(target_xyz, initial_position=initial_angles)

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
