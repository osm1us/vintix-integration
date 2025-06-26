"""
Калибровка камеры с использованием шахматной доски.

Этот скрипт выполняет калибровку камеры для определения ее внутренних параметров:
- Матрица камеры (camera matrix)
- Коэффициенты дисторсии (distortion coefficients)

Эти параметры необходимы для устранения искажений, вносимых объективом,
что позволяет получать "честную" картину мира и точно сопоставлять
пиксельные координаты с реальными мировыми координатами.

Принцип работы:
1. Скрипт ищет углы на шахматной доске на серии изображений.
2. На основе найденных 2D (пиксельных) и известных 3D (реальных) координат
   углов вычисляются параметры камеры.
3. Результаты сохраняются в файл JSON для дальнейшего использования.

Как использовать:
1. Распечатайте или откройте на экране изображение шахматной доски.
   Стандартный размер 9x6 (внутренних углов) хорошо подходит.
2. Создайте папку (например, 'calibration_images').
3. Сделайте 15-20 фотографий этой доски с разных углов и расстояний,
   стараясь, чтобы доска занимала разные части кадра. Сохраните их в
   созданную папку.
4. Запустите скрипт из терминала:
   python camera_calibration.py --path /путь/к/папке/с/фото --rows 6 --cols 9 --size 0.025

   --path: Путь к папке с изображениями.
   --rows: Количество внутренних углов по вертикали (по короткой стороне).
   --cols: Количество внутренних углов по горизонтали (по длинной стороне).
   --size: Размер одного квадрата доски в метрах (например, 0.025 для 2.5 см).

   Команда для запуска: python camera_calibration.py --path calibration_images --rows 6 --cols 9 --size 0.025

   --rows: Количество внутренних углов по короткой стороне. Для доски 7x10 это будет 6.
    --cols: Количество внутренних углов по длинной стороне. Для доски 7x10 это будет 9.
    --size: Тот самый размер квадрата, который ты измерил линейкой, но в МЕТРАХ (например, 0.025).
"""

import cv2
import numpy as np
import os
import argparse
import yaml
from datetime import datetime
import time

# --- Новые вспомогательные функции для трансформаций ---

def create_transform_matrix(rvec, tvec):
    """Создает матрицу гомогенного преобразования 4x4 из вектора вращения и переноса."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def invert_transform_matrix(T):
    """Инвертирует матрицу гомогенного преобразования."""
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

# --- Основная логика ---

def calibrate_intrinsic(images_path, chessboard_rows, chessboard_cols, square_size, force_save=False):
    """
    Выполняет ВНУТРЕННЮЮ калибровку камеры (матрица и дисторсия).
    """
    print("Начинаем ВНУТРЕННЮЮ калибровку камеры (Intrinsic)...")
    print(f"Параметры шахматной доски: {chessboard_rows}x{chessboard_cols}, размер квадрата: {square_size} м")

    # Критерии для уточнения найденных углов
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Подготовка "объектных точек" - 3D координат углов в мире (X, Y, Z)
    # Мы создаем сетку (0,0,0), (1,0,0), ..., (8,5,0) и умножаем на размер квадрата
    objp = np.zeros((chessboard_rows * chessboard_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_cols, 0:chessboard_rows].T.reshape(-1, 2)
    objp = objp * square_size

    # Массивы для хранения точек из реального мира и из 2D изображения
    objpoints = []  # 3D точки в реальном пространстве
    imgpoints = []  # 2D точки на плоскости изображения

    # Получаем список изображений
    image_files = [f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"Ошибка: В папке '{images_path}' не найдено изображений.")
        return False

    print(f"Найдено изображений: {len(image_files)}")
    
    image_shape = None
    found_corners_count = 0

    for fname in image_files:
        img_path = os.path.join(images_path, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Предупреждение: Не удалось прочитать изображение {fname}. Пропускаем.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None:
            image_shape = gray.shape[::-1]

        # Поиск углов шахматной доски
        ret, corners = cv2.findChessboardCorners(gray, (chessboard_cols, chessboard_rows), None)

        # Если углы найдены, добавляем объектные точки и точки изображения
        if ret:
            found_corners_count += 1
            print(f"  - Найдена доска на изображении: {fname}")
            objpoints.append(objp)

            # Уточняем координаты углов
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
        else:
            print(f"  - Не удалось найти доску на изображении: {fname}")

    if found_corners_count < 10:
        print(f"\nКритическая ошибка: Найдено слишком мало ({found_corners_count}) удачных изображений.")
        print("Для качественной калибровки рекомендуется не менее 10-15 изображений.")
        print("Калибровка прервана. Проверьте параметры доски и качество изображений.")
        return False

    print("\nВыполняется вычисление параметров камеры...")
    # Непосредственно калибровка
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_shape, None, None
    )

    if not ret:
        print("Ошибка: калибровка не удалась.")
        return False

    print("Калибровка успешно завершена!")

    # Расчет ошибки перепроецирования
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    mean_error_avg = mean_error / len(objpoints)
    print(f"Средняя ошибка перепроецирования: {mean_error_avg:.4f} пикселей")
    if mean_error_avg > 1.0:
        print("ОШИБКА: Ошибка перепроецирования > 1.0. Результаты калибровки НЕ будут сохранены.")
        print("Улучшите качество изображений (больше ракурсов, лучшее освещение) и попробуйте снова.")
        if not force_save:
            return False
        print("Принудительное сохранение включено. Файл будет создан, но его использование не рекомендуется.")
    else:
        print("Отличный результат! Ошибка < 1.0 пикселя.")

    # Сохранение результатов в файл
    output_filename = "camera_params.yaml"
    data = {
        "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "calibration_type": "intrinsic",
        "image_resolution": image_shape,
        "chessboard_size": (chessboard_cols, chessboard_rows),
        "square_size_m": square_size,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeff": dist_coeffs.tolist(),
        "reprojection_error": mean_error_avg
        # ВАЖНО: Мы больше не сохраняем rvecs и tvecs, так как они относятся
        # к конкретным калибровочным изображениям и являются источником ошибок.
    }

    try:
        with open(output_filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print(f"\nПараметры внутренней калибровки сохранены в файл: {output_filename}")
    except Exception as e:
        print(f"\nОшибка при сохранении файла: {e}")
        return False
        
    return True


def calibrate_extrinsic(intrinsic_params_file, chessboard_rows, chessboard_cols, square_size, board_position):
    """
    Выполняет ВНЕШНЮЮ калибровку "Рука-Глаз" для определения положения камеры.
    """
    print("Начинаем ВНЕШНЮЮ калибровку 'Рука-Глаз' (Extrinsic)...")

    # 1. Загружаем параметры внутренней калибровки
    try:
        with open(intrinsic_params_file, 'r') as f:
            params = yaml.safe_load(f)
        camera_matrix = np.array(params['camera_matrix'])
        dist_coeffs = np.array(params['dist_coeff'])
        print(f"Параметры внутренней калибровки успешно загружены из '{intrinsic_params_file}'.")
    except Exception as e:
        print(f"Критическая ошибка: Не удалось загрузить файл '{intrinsic_params_file}'. {e}")
        print("Сначала выполните внутреннюю калибровку: python camera_calibration.py --mode intrinsic ...")
        return False

    # 2. Подготовка объектных точек (как в intrinsic)
    objp = np.zeros((chessboard_rows * chessboard_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_cols, 0:chessboard_rows].T.reshape(-1, 2)
    objp = objp * square_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 3. Захват видео
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру.")
        return False

    print("\nКамера включена. Наведите на шахматную доску.")
    print("Доска должна быть расположена в ТОЧНО ИЗВЕСТНОМ месте относительно ОСНОВАНИЯ робота.")
    print(f"Ожидаемые координаты центра доски (x, y, z) в метрах: {board_position}")
    print("\nНажмите 'S', когда изображение будет стабильным, чтобы сохранить калибровку.")
    print("Нажмите 'Q', чтобы выйти.")

    found_once = False
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка: не удалось получить кадр.")
            time.sleep(1)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, (chessboard_cols, chessboard_rows), None)

        if ret_corners:
            if not found_once:
                print("\nДоска найдена! Убедитесь, что она неподвижна.")
                found_once = True
            
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (chessboard_cols, chessboard_rows), corners2, ret)

            # 4. Вычисление трансформации Камера -> Доска (T_cam_board)
            _, rvec, tvec = cv2.solvePnP(objp, corners2, camera_matrix, dist_coeffs)
            T_cam_board = create_transform_matrix(rvec, tvec)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                print("\nКлавиша 'S' нажата. Вычисляем и сохраняем трансформацию...")
                
                # 5. Инвертируем, чтобы получить Доска -> Камера
                T_board_cam = invert_transform_matrix(T_cam_board)

                # 6. Создаем трансформацию Робот -> Доска (T_robot_board)
                T_robot_board = np.eye(4)
                T_robot_board[:3, 3] = board_position

                # 7. Вычисляем главную трансформацию: Робот -> Камера
                # T_robot_cam = T_robot_board * T_board_cam
                T_robot_camera = T_robot_board @ T_board_cam

                # 8. Сохранение
                output_filename = "hand_eye_params.yaml"
                data = {
                    "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "calibration_type": "extrinsic_hand_eye",
                    "comment": "Transformation from robot base to camera frame (T_robot_camera)",
                    "transform_robot_to_camera": T_robot_camera.tolist()
                }
                try:
                    with open(output_filename, 'w') as f:
                        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
                    print(f"\nПараметры внешней калибровки сохранены в файл: {output_filename}")
                    print("Калибровка 'Рука-Глаз' успешно завершена!")
                    break 
                except Exception as e:
                    print(f"\nОшибка при сохранении файла: {e}")
                    break

            elif key == ord('q'):
                print("\nВыход без сохранения.")
                break
        else:
            if found_once:
                print("Потеряна доска из вида...")
            found_once = False

        cv2.imshow('Extrinsic Calibration - Press "s" to save, "q" to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nВыход без сохранения.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт калибровки камеры.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=['intrinsic', 'extrinsic'],
        default='intrinsic',
        help="Режим работы: 'intrinsic' для внутренней калибровки, 'extrinsic' для калибровки 'Рука-Глаз'."
    )
    
    # --- Аргументы для intrinsic ---
    parser.add_argument(
        "--path",
        type=str,
        default='calibration_images',
        help="[Intrinsic] Путь к папке с фотографиями шахматной доски."
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='[Intrinsic] Принудительно сохранить калибровку, даже если ошибка перепроецирования > 1.0.'
    )

    # --- Общие аргументы для доски ---
    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="Количество внутренних углов по ВЕРТИКАЛИ (по короткой стороне)."
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=9,
        help="Количество внутренних углов по ГОРИЗОНТАЛИ (по длинной стороне)."
    )
    parser.add_argument(
        "--size",
        type=float,
        default=0.025,
        help="Размер одного квадрата шахматной доски в МЕТРАХ (например, 0.025)."
    )

    # --- Аргументы для extrinsic ---
    parser.add_argument(
        "--config",
        type=str,
        default='camera_params.yaml',
        help="[Extrinsic] Путь к файлу с параметрами внутренней калибровки."
    )
    parser.add_argument(
        '--board_pos', 
        nargs=3, 
        type=float, 
        default=[0.3, 0.0, 0.0],
        help='[Extrinsic] Положение центра шахматной доски относительно ОСНОВАНИЯ робота в метрах (X Y Z).'
    )


    args = parser.parse_args()

    if args.mode == 'intrinsic':
        if not os.path.isdir(args.path):
            print(f"Ошибка: Указанный путь '{args.path}' не является папкой или не существует.")
        else:
            calibrate_intrinsic(args.path, args.rows, args.cols, args.size, args.force)
    
    elif args.mode == 'extrinsic':
        calibrate_extrinsic(args.config, args.rows, args.cols, args.size, args.board_pos) 