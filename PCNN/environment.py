import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

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
        if render:
            # Попытка запустить с принудительным использованием OpenGL 2
            # Это известный способ обхода проблем с драйверами на некоторых системах
            self.physics_client = p.connect(p.GUI, options="--opengl2")
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # --- 2. Основные параметры ---
        # Пути к ассетам теперь относительные, чтобы избежать проблем с кодировкой
        self.robot_urdf_path = "../manipulator.urdf"
        self.plane_urdf_path = "assets/plane.urdf"
        self.cube_urdf_path = "assets/cube.urdf"

        self.start_pos = [0, 0, 0]
        self.start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        
        self.plane_id = None
        self.robot_id = None
        self.cube_id = None

        self.controllable_joints = []
        self.num_controllable_joints = 0
        
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
        
        self.controllable_joints = []
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_id = info[0]
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            
            # Нас интересуют только управляемые суставы (не фиксированные)
            if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
                self.controllable_joints.append(joint_id)

            print(f"  Сустав {joint_id}: {joint_name} (Тип: {joint_type})")

        self.num_controllable_joints = len(self.controllable_joints)
        print(f"\nНайдено управляемых суставов: {self.num_controllable_joints}")
        print(f"Индексы управляемых суставов: {self.controllable_joints}")
        print("="*50)


    def reset(self):
        """
        Сбрасывает среду в начальное состояние.
        - Перезагружает объекты.
        - Устанавливает кубик в случайное положение.
        - Возвращает начальное наблюдение (observation).
        """
        p.resetSimulation()
        # Применяем Domain Randomization в начале каждого эпизода
        self._randomize_domain()
        
        p.setGravity(0, 0, -9.8)
        
        # --- Жесткое подавление вывода C++ ядра PyBullet ---
        # Сохраняем оригинальные файловые дескрипторы
        original_stdout_fd = sys.stdout.fileno()
        original_stderr_fd = sys.stderr.fileno()
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)

        # Перенаправляем вывод в "черную дыру"
        with open(os.devnull, 'wb') as devnull:
            os.dup2(devnull.fileno(), original_stdout_fd)
            os.dup2(devnull.fileno(), original_stderr_fd)
        
            try:
                # Загрузка всех объектов сцены, которые создают шум
                self.plane_id = p.loadURDF(self.plane_urdf_path)
                self.robot_id = p.loadURDF(self.robot_urdf_path, self.start_pos, self.start_orientation, useFixedBase=True)

                xpos = np.random.uniform(0.3, 0.6)
                ypos = np.random.uniform(-0.2, 0.2)
                cube_start_pos = [xpos, ypos, 0.05] 
                cube_start_orientation = p.getQuaternionFromEuler([0, 0, np.random.uniform(0, np.pi)])
                self.cube_id = p.loadURDF(self.cube_urdf_path, cube_start_pos, cube_start_orientation, globalScaling=0.1)

            except p.error as e:
                # В случае ошибки, сначала восстанавливаем вывод, чтобы увидеть сообщение
                os.dup2(saved_stdout_fd, original_stdout_fd)
                os.dup2(saved_stderr_fd, original_stderr_fd)
                print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить один из URDF-файлов.")
                print(f"Ошибка PyBullet: {e}")
                self.close()
                raise
            finally:
                # В любом случае восстанавливаем оригинальный вывод
                os.dup2(saved_stdout_fd, original_stdout_fd)
                os.dup2(saved_stderr_fd, original_stderr_fd)
                os.close(saved_stdout_fd)
                os.close(saved_stderr_fd)

        # --- Domain Randomization: Фон ---
        if self.desk_texture_id != -1 and np.random.rand() > 0.5:
            p.changeVisualShape(self.plane_id, -1, textureUniqueId=self.desk_texture_id)
        else:
            p.changeVisualShape(self.plane_id, -1, rgbaColor=[1, 1, 1, 1])
        
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
            renderer=p.ER_TINY_RENDERER
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
        joint_states = p.getJointStates(self.robot_id, self.controllable_joints)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        # --- 3. Собираем финальное наблюдение ---
        observation = np.concatenate([
            cube_coords_2d,
            joint_positions,
            joint_velocities,
        ]).astype(np.float32)

        return observation

    def step(self, action):
        """
        Выполняет один шаг в среде.

        :param action: Действие, которое должен выполнить агент.
        :return: observation, reward, done, info
        """
        # Применяем действие к управляемым суставам
        if action is not None and len(action) == self.num_controllable_joints:
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.controllable_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=action
            )
        
        p.stepSimulation()
        
        # Получаем новое наблюдение после шага симуляции
        obs = self._get_observation()
        
        # --- Расчет награды (Reward) ---
        # Для награды мы используем "читерские" 3D-координаты, т.к. агенту все равно,
        # как считается награда, ему важен сам сигнал.
        ee_state = p.getLinkState(self.robot_id, self.end_effector_link_index)
        ee_pos = ee_state[0]
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        
        distance = np.linalg.norm(np.array(ee_pos) - np.array(cube_pos))
        
        # Награда обратно пропорциональна расстоянию. Чем ближе, тем лучше.
        reward = -distance
        
        # --- Условие завершения эпизода (Done) ---
        # Пока простой вариант: завершаем, если схват очень близко к кубику
        done = False
        if distance < 0.05:  # Пороговое значение, 5 см
            done = True
            reward += 10  # Даем большую награду за достижение цели
            print("INFO: Цель достигнута!")

        info = {'distance': distance}
        
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
        # plt.show() # Это блокирующий вызов, он нам не нужен в интерактивном режиме

        # --- Создаем слайдеры для ручного управления ---
        # Получаем лимиты для каждого сустава, чтобы задать диапазон слайдера
        joint_limits = []
        for joint_id in env.controllable_joints:
            info = p.getJointInfo(env.robot_id, joint_id)
            joint_limits.append({'low': info[8], 'high': info[9]})

        sliders = []
        for i in range(env.num_controllable_joints):
            slider = p.addUserDebugParameter(
                paramName=f"Joint {i}",
                rangeMin=joint_limits[i]['low'],
                rangeMax=joint_limits[i]['high'],
                startValue=0
            )
            sliders.append(slider)

        print("Запуск тестового цикла. Окно симуляции активно.")
        print("Используйте слайдеры для управления роботом.")
        print("Закройте окно или нажмите Ctrl+C в терминале для выхода.")
        
        # Поддерживаем симуляцию активной, пока пользователь не закроет окно
        while True: 
            # Проверяем, не было ли запроса на закрытие окна симуляции
            if p.getConnectionInfo(env.physics_client)['isConnected'] == 0:
                print("Окно симуляции было закрыто пользователем.")
                break

            # Считываем значения со слайдеров
            action = [p.readUserDebugParameter(slider) for slider in sliders]
            
            # Передаем действие в среду
            obs, reward, done, info = env.step(action)
            
            if done:
                print("Эпизод завершен, сбрасываю среду...")
                env.reset()

            # Обновляем заголовок окна с информацией о дистанции и 2D координатах
            distance = info.get('distance', 0)
            # Координаты кубика теперь берем из наблюдения, которое видит агент
            cube_pixel_coords = obs[:2] 
            title = (f"Dist: {distance:.3f} | "
                     f"Cube Screen Coords (norm): ({cube_pixel_coords[0]:.2f}, {cube_pixel_coords[1]:.2f})")
            ax.set_title(title)

            # --- Рендерим изображение с камеры для отладки ---
            # PyBullet возвращает кортеж, 3-й элемент (индекс 2) - это RGBA массив
            _, _, rgba_img_flat, _, _ = p.getCameraImage(
                width=320,
                height=320,
                viewMatrix=env.camera_view_matrix,
                projectionMatrix=env.camera_proj_matrix,
                renderer=p.ER_TINY_RENDERER # Используем надежный программный рендерер
            )
            
            # PyBullet возвращает плоский массив, преобразуем его в 3D массив (высота, ширина, каналы)
            rgba_img = np.reshape(rgba_img_flat, (320, 320, 4))

            # Обновляем данные в нашем окне визуализации
            img_plot.set_data(rgba_img)
            # fig.canvas.draw()
            # fig.canvas.flush_events()

            # plt.pause - это правильный способ для обновления и небольшой задержки
            plt.pause(1./240.)

    except KeyboardInterrupt:
        print("\nПолучен сигнал на завершение работы (Ctrl+C).")
    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
    finally:
        if env:
            env.close()


if __name__ == "__main__":
    main()
