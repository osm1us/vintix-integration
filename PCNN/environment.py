import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
import gymnasium as gym

class ManipulatorEnv:
    """
    Класс окружения для обучения робота-манипулятора в PyBullet.
    Эта среда разработана для задач обучения с подкреплением (RL).
    """
    def __init__(self, render=True):
        """
        Инициализация симуляции.

        :param render: True, если нужно запустить с графическим интерфейсом (GUI).
        """
        # --- Телепортация рабочего каталога и самонастройка ассетов ---
        self.original_cwd = os.getcwd()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        os.chdir(script_dir)
        print(f"INFO: Рабочий каталог изменен на '{script_dir}' для обхода бага с кодировкой.")

        # --- 1. Подключение к симулятору ---
        # ВАЖНО: Мы всегда используем p.GUI, т.к. аппаратный рендеринг
        # (p.ER_BULLET_HARDWARE_OPENGL) для getCameraImage требует активного
        # OpenGL контекста, который создается только в этом режиме.
        self.physics_client = p.connect(p.GUI, options="--opengl2")

        if not render:
            # Если обучение "безголовое", отключаем отрисовку основного окна для макс. скорости.
            # Это не повлияет на рендеринг виртуальной камеры, который идет на GPU.
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # --- 2. Основные параметры ---
        # Пути к ассетам теперь относительные, чтобы избежать проблем с кодировкой
        self.robot_urdf_path = "manipulator.urdf"
        self.plane_urdf_path = "assets/plane.urdf"
        self.cube_urdf_path = "assets/cube.urdf"

        self.robot_id = None
        self.cube_id = None

        self.arm_joints = []
        self.gripper_joints = []
        
        self.num_arm_joints = 6
        self.num_gripper_joints = 2 # Один ведущий, один ведомый
        
        # Настройка камеры для лучшего обзора
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.5, 0, 0.5])

        # --- Предзагрузка ассетов для Domain Randomization ---
        # Загружаем текстуру один раз, чтобы не обращаться к диску в цикле обучения
        try:
            self.desk_texture_id = p.loadTexture("assets/parta.png")
        except p.error:
            self.desk_texture_id = -1 # Обозначим, что текстура не загрузилась
            print("WARNING: Не удалось предзагрузить текстуру 'assets/parta.png'.")

        # --- Инициализация матриц камеры (будут перезаписаны в reset) ---
        self.camera_view_matrix = None
        self.camera_proj_matrix = None
        # Индекс звена-схвата. В нашем URDF 6-й сустав (индекс 5) двигает последнее звено.
        self.end_effector_link_index = 5
        self.link_name_to_index = {}
        self.finger_link_indices = []

        # --- 3. Определение пространств действий и состояний (для RL) ---
        # Размерность вектора действий: 6 (рука) + 1 (клешня) = 7
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,), dtype=np.float32)

        # Размерность вектора наблюдений: 16
        # 2 (коорд. куба) + 6 (поз. руки) + 1 (поз. клешни) + 6 (скор. руки) + 1 (скор. клешни)
        obs_low = np.array(
            [-1.0] * 2 +       # Координаты кубика XY (нормализованные)
            [-np.pi] * 6 +     # Положения суставов руки
            [0.0] +            # Положение клешни
            [-20] * 6 +        # Скорости суставов руки
            [-5],              # Скорость клешни
            dtype=np.float32
        )
        obs_high = np.array(
            [1.0] * 2 +        # Координаты кубика XY (нормализованные)
            [np.pi] * 6 +      # Положения суставов руки
            [0.055] +          # Положение клешни (макс. раскрытие)
            [20] * 6 +         # Скорости суставов руки
            [5],               # Скорость клешни
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)


    def _randomize_domain(self):
        """
        Применяет случайные изменения к параметрам симуляции.
        Вызывается в начале каждого эпизода (в reset).
        """
        # --- 1. Случайное положение камеры ---
        # Небольшие смещения от "идеальной" позиции
        eye_pos_x = np.random.uniform(-0.1, 0.1)
        eye_pos_y = np.random.uniform(-0.1, 0.1)
        eye_pos_z = np.random.uniform(1.0, 1.2)
        
        self.camera_view_matrix = p.computeViewMatrix(
            cameraEyePosition=[eye_pos_x, eye_pos_y, eye_pos_z],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 1, 0]
        )

        # --- 2. Случайный "зум" (угол обзора) ---
        fov = np.random.uniform(65.0, 85.0)
        self.camera_proj_matrix = p.computeProjectionMatrixFOV(
            fov=fov,
            aspect=1.0,
            nearVal=0.1,
            farVal=100.0
        )

    def _log_robot_info(self):
        """Выводит в консоль информацию о суставах робота."""
        print("="*50)
        print(f"Информация о роботе ID: {self.robot_id}")
        num_joints = p.getNumJoints(self.robot_id)
        print(f"Всего суставов: {num_joints}")
        print("="*50)
        
        self.arm_joints = []
        self.gripper_joints = []
        self.link_name_to_index = {} # Очищаем перед заполнением
        self.finger_link_indices = [] # ОЧИЩАЕМ СПИСОК ПРИ КАЖДОМ СБРОСЕ
        
        # Ожидаемые имена суставов
        gripper_joint_names = [b'finger_joint1', b'finger_joint2']

        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1]
            link_name = info[12].decode('utf-8')
            joint_type = info[2]
            
            self.link_name_to_index[link_name] = i
            
            if joint_type == p.JOINT_REVOLUTE and len(self.arm_joints) < self.num_arm_joints:
                self.arm_joints.append(i)
            elif joint_name in gripper_joint_names and joint_type == p.JOINT_PRISMATIC:
                 self.gripper_joints.append(i)

        # Ведущий сустав должен быть первым в списке
        if p.getJointInfo(self.robot_id, self.gripper_joints[0])[1] != b'finger_joint1':
            self.gripper_joints.reverse()

        # Находим и сохраняем индексы звеньев пальцев
        if 'finger_link1' in self.link_name_to_index:
            self.finger_link_indices.append(self.link_name_to_index['finger_link1'])
        if 'finger_link2' in self.link_name_to_index:
            self.finger_link_indices.append(self.link_name_to_index['finger_link2'])

        print(f"Найдено суставов руки: {len(self.arm_joints)}, Индексы: {self.arm_joints}")
        print(f"Найдено суставов клешни: {len(self.gripper_joints)}, Индексы: {self.gripper_joints}")
        print(f"Найдены индексы звеньев пальцев: {self.finger_link_indices}")
        print("="*50)


    def reset(self):
        """
        Сбрасывает среду в начальное состояние.
        - Перезагружает объекты.
        - Устанавливает кубик в случайное положение.
        - Возвращает начальное наблюдение (observation).
        """
        p.resetSimulation()
        self._randomize_domain()
        
        p.setGravity(0, 0, -9.8)
        
        # --- Тонкая настройка физики для надежного захвата ---
        # Уменьшаем шаг симуляции для большей точности
        p.setTimeStep(1/500.0)
        # Увеличиваем число итераций решателя контактов
        p.setPhysicsEngineParameter(numSolverIterations=200)
        # Уменьшаем "зазор" при контакте, чтобы убрать эффект левитации
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)
        
        # --- Жесткое подавление вывода C++ ядра PyBullet ---
        original_stdout_fd = sys.stdout.fileno()
        original_stderr_fd = sys.stderr.fileno()
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)

        with open(os.devnull, 'wb') as devnull:
            os.dup2(devnull.fileno(), original_stdout_fd)
            os.dup2(devnull.fileno(), original_stderr_fd)
        
            try:
                p.loadURDF(self.plane_urdf_path)
                self.robot_id = p.loadURDF(self.robot_urdf_path, [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

                # Возвращаем рассчитанную область и правильную загрузку куба
                
                # --- НОВАЯ ЛОГИКА: спавн в полукруге, чтобы заставить вращаться ---
                radius = np.random.uniform(0.25, 0.45) # Спавним на разном расстоянии
                angle = np.random.uniform(-np.pi / 2, np.pi / 2) # Спавним в секторе 180 градусов перед роботом
                
                xpos = radius * np.cos(angle)
                ypos = radius * np.sin(angle)

                self.initial_cube_pos = [xpos, ypos, 0.025] 
                self.cube_id = p.loadURDF(self.cube_urdf_path, self.initial_cube_pos)

            except p.error as e:
                os.dup2(saved_stdout_fd, original_stdout_fd)
                os.dup2(saved_stderr_fd, original_stderr_fd)
                print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить один из URDF-файлов.")
                print(f"Ошибка PyBullet: {e}")
                self.close()
                raise
            finally:
                os.dup2(saved_stdout_fd, original_stdout_fd)
                os.dup2(saved_stderr_fd, original_stderr_fd)
                os.close(saved_stdout_fd)
                os.close(saved_stderr_fd)
        # --- Domain Randomization: Фон ---
        if self.desk_texture_id != -1:
            p.changeVisualShape(p.getBodyUniqueId(0), -1, textureUniqueId=self.desk_texture_id)
        
        # Устанавливаем цвет кубика - красный
        p.changeVisualShape(self.cube_id, -1, rgbaColor=[1, 0, 0, 1])

        # Логируем информацию о роботе после его загрузки
        self._log_robot_info()
        
        print(f"Среда сброшена. Робот ID: {self.robot_id}, Кубик ID: {self.cube_id}")
        
        return self._get_observation()

    def _get_observation(self):
        """
        Собирает и возвращает текущее состояние среды (наблюдение).
        Ключевой момент: мы симулируем РЕАЛЬНУЮ систему зрения.
        Агент получает 2D пиксельные координаты цели, а не "читерские" 3D.
        """
        # --- 1. Симуляция компьютерного зрения ---
        # Получаем изображение и маску сегментации с виртуальной камеры
        _, _, _, _, seg_mask_flat = p.getCameraImage(
            width=320,
            height=320,
            viewMatrix=self.camera_view_matrix,
            projectionMatrix=self.camera_proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Преобразуем плоский массив в 2D матрицу
        seg_mask = np.reshape(seg_mask_flat, (320, 320))

        # Ищем пиксели, принадлежащие кубику
        cube_pixels = np.where(seg_mask == self.cube_id)
        
        # Если кубик виден, находим его центр в пикселях
        if len(cube_pixels[0]) > 0:
            # y-координата (среднее по строкам), x-координата (среднее по столбцам)
            center_y = np.mean(cube_pixels[0]) 
            center_x = np.mean(cube_pixels[1])
            # Нормализуем координаты к диапазону [0, 1] с центром в левом-верхнем углу (как в OpenCV)
            norm_x = center_x / 320.0
            norm_y = center_y / 320.0
        else:
            # Если кубик не виден, возвращаем "сигнальное" значение за пределами [0, 1]
            norm_x = -1.0
            norm_y = -1.0
        
        cube_coords_2d = np.array([norm_x, norm_y])

        # --- 2. Состояние робота ---
        arm_states = p.getJointStates(self.robot_id, self.arm_joints)
        arm_positions = [state[0] for state in arm_states]
        arm_velocities = [state[1] for state in arm_states]
        
        # Состояние 1 сустава клешни (ведущего)
        gripper_state = p.getJointState(self.robot_id, self.gripper_joints[0])
        gripper_position = [gripper_state[0]]
        gripper_velocity = [gripper_state[1]]
        
        # --- 3. Собираем финальное наблюдение ---
        observation = np.concatenate([
            cube_coords_2d,
            arm_positions, gripper_position,
            arm_velocities, gripper_velocity,
        ]).astype(np.float32)

        return observation

    def step(self, action):
        """
        Выполняет один шаг в среде.

        :param action: Действие, которое должен выполнить агент.
        :return: observation, reward, done, info
        """
        # Первые 6 действий для руки, 7-е для клешни
        arm_action = action[:self.num_arm_joints]
        # Возвращаем корректный диапазон для клешни [0, 0.055]
        gripper_action_normalized = (action[self.num_arm_joints] + 1) / 2 
        gripper_target_pos = gripper_action_normalized * 0.055

        # --- Деликатное управление ---
        # Ограничиваем максимальное усилие и скорость, чтобы избежать "ударов"
        # Для руки оставляем большое усилие, чтобы она была сильной
        arm_forces = [100.0] * self.num_arm_joints
        # Для клешни ставим усилие, достаточное для уверенного захвата, но не "взрывное"
        gripper_force = 2.5
        
        # Применяем действие к руке
        p.setJointMotorControlArray(
            self.robot_id, 
            self.arm_joints, 
            p.POSITION_CONTROL, 
            targetPositions=arm_action,
            forces=arm_forces
        )
        # Применяем действие к обоим пальцам клешни для идеальной синхронизации
        p.setJointMotorControlArray(
            self.robot_id, 
            self.gripper_joints, # Управляем обоими суставами
            p.POSITION_CONTROL, 
            targetPositions=[gripper_target_pos] * self.num_gripper_joints,
            forces=[gripper_force] * self.num_gripper_joints
            )
        
        p.stepSimulation()
        
        # Получаем новое наблюдение после шага симуляции
        obs = self._get_observation()
        
        # --- Новая функция вознаграждения ---
        ee_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
        ee_pos = np.array(ee_state[0])
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        cube_pos = np.array(cube_pos)

        # 1. Базовая награда за приближение (всегда активна)
        dist_to_cube = np.linalg.norm(ee_pos - cube_pos)
        reward = -dist_to_cube

        # 2. Проверяем, есть ли контакт именно с пальцами
        is_gripping_properly = False
        if self.finger_link_indices:
            contact_points_finger1 = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id, linkIndexA=self.finger_link_indices[0])
            contact_points_finger2 = p.getContactPoints(bodyA=self.robot_id, bodyB=self.cube_id, linkIndexA=self.finger_link_indices[1])
            if contact_points_finger1 or contact_points_finger2:
                 is_gripping_properly = True

        is_closing_gripper = gripper_target_pos < 0.015

        # 3. Структурированная система бонусов
        if dist_to_cube < 0.05: # Если схват очень близко к цели
            reward += 0.1 # Небольшой бонус за нахождение в "зоне захвата"

            if is_closing_gripper:
                reward += 0.2 # Бонус за попытку закрыть клешню в правильном месте

            if is_gripping_properly:
                reward += 1.0 # Большой бонус за правильный контакт пальцами!

                # 4. Награда за подъем (только если кубик правильно схвачен)
                lift_height = cube_pos[2] - self.initial_cube_pos[2]
                if lift_height > 0.01: # Если есть хоть какой-то отрыв от стола
                     reward += lift_height * 30 # Увеличим множитель
                     
        else:
            # Штраф за закрытие клешни далеко от кубика, чтобы избежать лишних движений
            if is_closing_gripper:
                reward -= 0.1
        
        # --- Условие завершения эпизода (Done) ---
        done = False
        # 5. Финальный бонус за успех - ТЕПЕРЬ С ПРОВЕРКОЙ ЗАХВАТА
        if is_gripping_properly and cube_pos[2] > self.initial_cube_pos[2] + 0.1:
            done = True
            reward += 50  # Огромный бонус за успех
            print("INFO: Цель достигнута! Кубик поднят ПРАВИЛЬНО.")

        # Условие провала: если кубик упал со стола
        if cube_pos[2] < -0.1: # Дадим небольшой запас на случай проваливания сквозь пол
            reward -= 10
            done = True
            print("INFO: Провал! Кубик уронили.")

        info = {'distance': dist_to_cube, 'is_grasping': is_gripping_properly}
        
        return obs, reward, done, info

    def close(self):
        """Закрывает соединение с симулятором."""
        if self.physics_client is not None:
            p.disconnect()
            self.physics_client = None
            print("Симуляция остановлена.")
        
        # Возвращаем рабочий каталог в исходное состояние
        os.chdir(self.original_cwd)
        print(f"INFO: Рабочий каталог восстановлен на '{self.original_cwd}'.")

# --- Блок для тестирования среды ---
def main():
    """
    Основная функция для демонстрации и тестирования окружения.
    Запускает симуляцию и поддерживает ее активной.
    """
    env = None
    try:
        env = ManipulatorEnv(render=True)
        env.reset()
        
        # --- Настройка для живой визуализации камеры ---
        plt.ion() # Включаем интерактивный режим
        fig, ax = plt.subplots()
        # Создаем пустой объект изображения для последующего обновления
        img_plot = ax.imshow(np.zeros((320, 320, 4))) 
        ax.set_title("Simulated Camera View")

        # --- Создаем слайдеры для ручного управления ---
        joint_limits = []
        for joint_id in env.arm_joints:
            info = p.getJointInfo(env.robot_id, joint_id)
            joint_limits.append({'low': info[8], 'high': info[9]})

        sliders = []
        for i in range(env.num_arm_joints):
            slider = p.addUserDebugParameter(
                paramName=f"Joint {i}",
                rangeMin=joint_limits[i]['low'],
                rangeMax=joint_limits[i]['high'],
                startValue=0
            )
            sliders.append(slider)

        # Добавляем слайдер для клешни
        gripper_slider = p.addUserDebugParameter(
            paramName="Gripper",
            rangeMin=-1, # Закрыто
            rangeMax=1,  # Открыто
            startValue=1
        )
        sliders.append(gripper_slider)

        print("Запуск тестового цикла. Окно симуляции активно.")
        print("Используйте слайдеры для управления роботом.")
        print("Закройте окно или нажмите Ctrl+C в терминале для выхода.")
        
        # Поддерживаем симуляцию активной, пока пользователь не закроет окно
        while p.isConnected(): 
            try:
                # Считываем значения со слайдеров
                action = [p.readUserDebugParameter(slider) for slider in sliders]
                
                # Передаем действие в среду
                obs, reward, done, info = env.step(action)
                
                if done:
                    print("Эпизод завершен, сбрасываю среду...")
                    env.reset()

                # Обновляем заголовок окна с информацией
                distance = info.get('distance', 0)
                is_grasping = "YES" if info.get('is_grasping') else "NO"
                cube_pixel_coords = obs[:2] 
                title = (f"Dist: {distance:.3f} | Grasping: {is_grasping} | "
                         f"Coords: ({cube_pixel_coords[0]:.2f}, {cube_pixel_coords[1]:.2f})")
                ax.set_title(title)

                # --- Рендерим изображение с камеры для отладки ---
                _, _, rgba_img_flat, _, _ = p.getCameraImage(
                    width=320,
                    height=320,
                    viewMatrix=env.camera_view_matrix,
                    projectionMatrix=env.camera_proj_matrix,
                    renderer=p.ER_BULLET_HARDWARE_OPENGL
                )
                
                rgba_img = np.reshape(rgba_img_flat, (320, 320, 4))
                img_plot.set_data(rgba_img)
                plt.pause(1./240.)
            except p.error:
                # Если окно было закрыто, PyBullet может выдать ошибку
                break

    except KeyboardInterrupt:
        print("\nПолучен сигнал на завершение работы (Ctrl+C).")
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
    finally:
        if env:
            env.close()


if __name__ == "__main__":
    main()
