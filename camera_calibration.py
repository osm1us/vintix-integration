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
   python camera_calibration.py --path /путь/к/папке/с/фото --rows 6 --cols 9 --size 2.5

   --path: Путь к папке с изображениями.
   --rows: Количество внутренних углов по вертикали (по короткой стороне).
   --cols: Количество внутренних углов по горизонтали (по длинной стороне).
   --size: Размер одного квадрата доски в реальных единицах (например, в сантиметрах).

   Команда для запуска: python camera_calibration.py --path calibration_images --rows 6 --cols 9 --size 2.5

   --rows: Количество внутренних углов по короткой стороне. Для доски 7x10 это будет 6.
    --cols: Количество внутренних углов по длинной стороне. Для доски 7x10 это будет 9.
    --size: Тот самый размер квадрата, который ты измерил линейкой (например, 2.5).
"""

import cv2
import numpy as np
import os
import argparse
import yaml
from datetime import datetime

def calibrate_camera(images_path, chessboard_rows, chessboard_cols, square_size):
    """
    Выполняет калибровку камеры и сохраняет параметры в файл.

    Args:
        images_path (str): Путь к папке с изображениями шахматной доски.
        chessboard_rows (int): Количество внутренних углов по вертикали.
        chessboard_cols (int): Количество внутренних углов по горизонтали.
        square_size (float): Размер квадрата шахматной доски в реальных единицах.

    Returns:
        bool: True, если калибровка прошла успешно, иначе False.
    """
    print("Начинаем калибровку камеры...")
    print(f"Параметры шахматной доски: {chessboard_rows}x{chessboard_cols}, размер квадрата: {square_size} см")

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
        print(f"\nПредупреждение: Найдено слишком мало ({found_corners_count}) удачных изображений.")
        print("Для качественной калибровки рекомендуется не менее 10-15 изображений.")
        if found_corners_count == 0:
            print("Калибровка невозможна. Проверьте параметры доски и качество изображений.")
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
        print("Предупреждение: Ошибка перепроецирования > 1.0. Результаты калибровки могут быть неточными.")
    else:
        print("Отличный результат! Ошибка < 1.0 пикселя.")

    # Сохранение результатов в файл
    output_filename = "camera_params.yaml"
    data = {
        "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_resolution": image_shape,
        "chessboard_size": (chessboard_cols, chessboard_rows),
        "square_size_cm": square_size,
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeff": dist_coeffs.tolist(),
        "rvecs": [r.tolist() for r in rvecs],
        "tvecs": [t.tolist() for t in tvecs],
        "reprojection_error": mean_error_avg
    }

    try:
        with open(output_filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        print(f"\nПараметры калибровки сохранены в файл: {output_filename}")
    except Exception as e:
        print(f"\nОшибка при сохранении файла: {e}")
        return False
        
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Скрипт калибровки камеры по шахматной доске.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Путь к папке с фотографиями шахматной доски."
    )
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
        default=2.5,
        help="Размер одного квадрата шахматной доски в сантиметрах."
    )

    args = parser.parse_args()

    if not os.path.isdir(args.path):
        print(f"Ошибка: Указанный путь '{args.path}' не является папкой или не существует.")
    else:
        calibrate_camera(args.path, args.rows, args.cols, args.size) 