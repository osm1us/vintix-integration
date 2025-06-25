#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Вспомогательные функции для системы компьютерного зрения и манипулятора
"""

import socket
import logging
import requests
import time
import cv2
import sys
import math
import numpy as np
from config import MANIPULATOR_CONFIG, QUADRANT_CORRECTIONS, CALIBRATION_POINTS, SERVO_CONFIG, QUADRANT_BUFFER

def setup_logger(name="robot", level=logging.INFO):
    """
    Настраивает и возвращает объект логгера
    
    Returns:
        logger: настроенный объект логгера
    """
    logger = logging.getLogger(name)
    
    # Проверяем, не настроен ли уже логгер
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # Создаем обработчик для вывода в консоль
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(level)
    
    # Создаем форматтер для логов
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Добавляем обработчик к логгеру
    logger.addHandler(console_handler)
    
    # Отключаем распространение логов от дочерних логгеров к родительским
    logger.propagate = False
    
    return logger

# Создаем логгер при импорте модуля
logger = setup_logger()

def check_connection(ip, port=80, timeout=3):
    """
    Проверяет соединение с устройством по IP
    
    Args:
        ip (str): IP-адрес устройства
        port (int, optional): Порт для проверки. По умолчанию 80.
        timeout (int, optional): Таймаут в секундах. По умолчанию 3.
        
    Returns:
        bool: True если соединение установлено, иначе False
    """
    logger.debug(f"Проверка соединения с {ip}:{port} (таймаут: {timeout}с)")
    
    # Сначала попробуем базовое подключение через сокет (быстрее)
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((ip, port))
        logger.debug(f"Соединение с {ip}:{port} установлено через сокет")
        return True
    except (socket.timeout, socket.error, ConnectionRefusedError, OSError) as e:
        logger.debug(f"Не удалось подключиться через сокет: {e}")
    
    # Если базовое не сработало, пробуем через HTTP запрос
    try:
        response = requests.get(f"http://{ip}", timeout=timeout)
        logger.debug(f"Соединение с {ip} установлено через HTTP")
        return True
    except requests.RequestException as e:
        logger.debug(f"Не удалось подключиться через HTTP: {e}")
    
    # Обе попытки не удались
    return False

def safe_cv2_rectangle(frame, pt1, pt2, color, thickness=1):
    """
    Безопасно рисует прямоугольник на кадре с проверкой границ
    
    Args:
        frame: Изображение
        pt1: Верхняя левая точка (x1, y1)
        pt2: Нижняя правая точка (x2, y2)
        color: Цвет в формате BGR
        thickness: Толщина линии
        
    Returns:
        frame: Изображение с нарисованным прямоугольником
    """
    if frame is None:
        return None
        
    h, w = frame.shape[:2]
    x1, y1 = max(0, min(pt1[0], w-1)), max(0, min(pt1[1], h-1))
    x2, y2 = max(0, min(pt2[0], w-1)), max(0, min(pt2[1], h-1))
    
    return cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

def calculate_optimal_angles(pixel_x, pixel_y, z_offset=None):
    """
    Рассчитывает оптимальные углы сервоприводов на основе пиксельных координат.
    
    Использует комбинацию линейной интерполяции по расстоянию и полярной системы координат
    для обеспечения точности в пределах 1 градуса для всех сервоприводов.
    
    Args:
        pixel_x: X-координата целевой точки в пикселях
        pixel_y: Y-координата целевой точки в пикселях
        z_offset: смещение по высоте в миллиметрах (для захвата объекта)
        
    Returns:
        словарь с углами и параметрами для всех сервоприводов
    """
    # Получаем параметры из конфигурации
    config = MANIPULATOR_CONFIG
    center_x = config.get('CENTER_PIXEL', {}).get('X', 700)
    center_y = config.get('CENTER_PIXEL', {}).get('Y', 374)
    scale = config['SCALE_FACTOR']
    
    # Проверка диапазона расстояний от центра манипулятора
    min_reach = config.get('MIN_REACH', 100)
    max_reach = config.get('MAX_REACH', 400)
    
    # 1. Преобразование пиксельных координат в миллиметры
    x_mm = (pixel_x - center_x) * scale
    y_mm = (center_y - pixel_y) * scale  # Инверсия оси Y
    
    # 2. Расчет расстояния от центра манипулятора и полярного угла
    distance = math.sqrt(x_mm**2 + y_mm**2)
    polar_angle = math.degrees(math.atan2(y_mm, x_mm))
    
    # Нормализация полярного угла в диапазоне 0-360 градусов
    polar_angle = (polar_angle + 360) % 360
    
    # 3. Проверка пределов досягаемости
    if distance < min_reach:
        logger.warning(f"Расстояние {distance:.1f} мм меньше минимального {min_reach} мм")
    elif distance > max_reach:
        logger.warning(f"Расстояние {distance:.1f} мм больше максимального {max_reach} мм")
    
    # 4. Определение четверти для коррекции угла основания
    if x_mm >= 0 and y_mm >= 0:
        quadrant = 1
    elif x_mm < 0 and y_mm >= 0:
        quadrant = 2
    elif x_mm < 0 and y_mm < 0:
        quadrant = 3
    else:  # x_mm >= 0 and y_mm < 0
        quadrant = 4
    
    logger.info(f"Координаты: x={x_mm:.1f}мм, y={y_mm:.1f}мм, расстояние={distance:.1f}мм, полярный угол={polar_angle:.1f}°, четверть={quadrant}")
    
    # 5. Расчет угла поворота основания (ось 5)
    # Преобразуем полярный угол в угол сервопривода основания (0-180°)
    if 0 <= polar_angle < 180:
        servo_base_angle = polar_angle
    else:  # 180 <= polar_angle < 360
        servo_base_angle = 360 - polar_angle
    
    # Применяем плавную коррекцию угла на границах четвертей
    buffer_config = QUADRANT_BUFFER
    buffer_width = buffer_config.get('width', 15)  # ширина буферной зоны в мм
    buffer_enabled = buffer_config.get('enabled', True)
    all_borders = buffer_config.get('all_borders', True)
    
    # Применяем плавную коррекцию только если буферная зона включена
    if buffer_enabled:
        # Определяем расстояние до границы четверти
        distance_to_x_border = abs(x_mm)  # расстояние до оси Y (граница между 1-2 и 3-4 четвертями)
        distance_to_y_border = abs(y_mm)  # расстояние до оси X (граница между 1-4 и 2-3 четвертями)
        
        # Плавная коррекция для границы между 1 и 2 четвертями (ось Y)
        if distance_to_x_border < buffer_width and ((quadrant == 1 and y_mm > 0) or (quadrant == 2 and y_mm > 0)):
            # Рассчитываем вес для каждой четверти в диапазоне от 0 до 1
            weight_q2 = 1.0 - (distance_to_x_border / buffer_width)
            weight_q1 = 1.0 - weight_q2
            
            # Применяем взвешенную коррекцию
            correction = (QUADRANT_CORRECTIONS[1] * weight_q1) + (QUADRANT_CORRECTIONS[2] * weight_q2)
            logger.info(f"Плавная коррекция между 1 и 2 четвертями: {correction:.2f}° (вес_1={weight_q1:.2f}, вес_2={weight_q2:.2f})")
            servo_base_angle += correction
        
        # Плавная коррекция для других границ, если включено
        elif all_borders:
            # Граница между 3 и 4 четвертями (нижняя горизонтальная)
            if distance_to_x_border < buffer_width and ((quadrant == 3 and y_mm < 0) or (quadrant == 4 and y_mm < 0)):
                weight_q3 = 1.0 - (distance_to_x_border / buffer_width)
                weight_q4 = 1.0 - weight_q3
                correction = (QUADRANT_CORRECTIONS[3] * weight_q3) + (QUADRANT_CORRECTIONS[4] * weight_q4)
                logger.info(f"Плавная коррекция между 3 и 4 четвертями: {correction:.2f}°")
                servo_base_angle += correction
            
            # Граница между 1 и 4 четвертями (правая вертикальная)
            elif distance_to_y_border < buffer_width and ((quadrant == 1 and x_mm > 0) or (quadrant == 4 and x_mm > 0)):
                weight_q4 = 1.0 - (distance_to_y_border / buffer_width)
                weight_q1 = 1.0 - weight_q4
                correction = (QUADRANT_CORRECTIONS[1] * weight_q1) + (QUADRANT_CORRECTIONS[4] * weight_q4)
                logger.info(f"Плавная коррекция между 1 и 4 четвертями: {correction:.2f}°")
                servo_base_angle += correction
            
            # Граница между 2 и 3 четвертями (левая вертикальная)
            elif distance_to_y_border < buffer_width and ((quadrant == 2 and x_mm < 0) or (quadrant == 3 and x_mm < 0)):
                weight_q3 = 1.0 - (distance_to_y_border / buffer_width)
                weight_q2 = 1.0 - weight_q3
                correction = (QUADRANT_CORRECTIONS[2] * weight_q2) + (QUADRANT_CORRECTIONS[3] * weight_q3)
                logger.info(f"Плавная коррекция между 2 и 3 четвертями: {correction:.2f}°")
                servo_base_angle += correction
            
            # Для точек не в буферной зоне - стандартная коррекция по четверти
            else:
                servo_base_angle += QUADRANT_CORRECTIONS[quadrant]
                logger.info(f"Стандартная коррекция для четверти {quadrant}: {QUADRANT_CORRECTIONS[quadrant]}°")
        else:
            # Если включена только буферная зона между 1 и 2 четвертями, для остальных - стандартная коррекция
            servo_base_angle += QUADRANT_CORRECTIONS[quadrant]
            logger.info(f"Стандартная коррекция для четверти {quadrant}: {QUADRANT_CORRECTIONS[quadrant]}°")
    else:
        # Если буферная зона отключена - стандартная коррекция
        servo_base_angle += QUADRANT_CORRECTIONS[quadrant]
        logger.info(f"Буферная зона отключена. Стандартная коррекция: {QUADRANT_CORRECTIONS[quadrant]}°")
    
    logger.info(f"Угол основания с коррекцией: {servo_base_angle:.1f}°")
    
    # Ограничиваем угол основания диапазоном 0-180°
    servo_base_angle = max(0, min(180, servo_base_angle))
    
    # 6. Получение калибровочных данных
    cal_points = CALIBRATION_POINTS['distance_points']
    cal_distances = []
    
    # Преобразуем калибровочные точки в расстояния в мм
    for point in cal_points:
        point_x_mm = (point['pixel_x'] - center_x) * scale
        point_y_mm = (center_y - point['pixel_y']) * scale
        point_distance = math.sqrt(point_x_mm**2 + point_y_mm**2)
        cal_distances.append(point_distance)
    
    shoulder_angles = CALIBRATION_POINTS['shoulder_angles']
    elbow_angles = CALIBRATION_POINTS['elbow_angles']
    wrist_angles = CALIBRATION_POINTS['wrist_angles']
    
    # 7. Поиск ближайших калибровочных точек по расстоянию
    
    # Проверка на точное совпадение с калибровочной точкой
    exact_match = False
    exact_index = -1
    
    for i, d in enumerate(cal_distances):
        if abs(distance - d) < 0.5:  # Допуск 0.5 мм для точного совпадения
            exact_match = True
            exact_index = i
            break
    
    if exact_match:
        # Если есть точное совпадение, используем значения из калибровки напрямую
        shoulder_angle = shoulder_angles[exact_index]
        elbow_angle = elbow_angles[exact_index]
        wrist_angle = wrist_angles[exact_index]
        logger.info(f"Точное совпадение с калибровочной точкой {exact_index+1}")
    else:
        # Если расстояние выходит за пределы калибровочного диапазона
        if distance <= min(cal_distances):
            # Для слишком близких точек берем значения первой калибровочной точки
            shoulder_angle = shoulder_angles[0]
            elbow_angle = elbow_angles[0]
            wrist_angle = wrist_angles[0]
            logger.info(f"Расстояние {distance:.1f}мм меньше минимального калибровочного {min(cal_distances):.1f}мм, используем крайнюю точку")
        elif distance >= max(cal_distances):
            # Для слишком далеких точек берем значения последней калибровочной точки
            shoulder_angle = shoulder_angles[-1]
            elbow_angle = elbow_angles[-1]
            wrist_angle = wrist_angles[-1]
            logger.info(f"Расстояние {distance:.1f}мм больше максимального калибровочного {max(cal_distances):.1f}мм, используем крайнюю точку")
        else:
            # 8. Интерполяция для точек в диапазоне калибровки
            
            # Сортируем расстояния с сохранением индексов
            sorted_distances_with_idx = sorted([(i, d) for i, d in enumerate(cal_distances)], key=lambda x: x[1])
            
            # Находим две ближайшие точки для основной интерполяции
            idx_below = -1
            idx_above = -1
            
            # Ищем точки, между которыми находится целевое расстояние
            for i in range(len(sorted_distances_with_idx) - 1):
                idx1, d1 = sorted_distances_with_idx[i]
                idx2, d2 = sorted_distances_with_idx[i+1]
                
                if d1 <= distance <= d2 or d2 <= distance <= d1:
                    if d1 <= d2:
                        idx_below, idx_above = idx1, idx2
                    else:
                        idx_below, idx_above = idx2, idx1
                    break
            
            # Если не нашли подходящий интервал, используем две ближайшие точки
            if idx_below == -1:
                # Сортируем расстояния по близости к целевой точке
                distances_by_proximity = sorted([(i, abs(d - distance)) for i, d in enumerate(cal_distances)], key=lambda x: x[1])
                idx_below = distances_by_proximity[0][0]
                idx_above = distances_by_proximity[1][0]
                logger.warning(f"Не найден точный интервал, используем ближайшие точки {idx_below+1} и {idx_above+1}")
            
            # Расчет коэффициентов интерполяции
            d_below = cal_distances[idx_below]
            d_above = cal_distances[idx_above]
            
            # Защита от деления на ноль
            if abs(d_above - d_below) < 0.001:
                # Если расстояния совпадают, используем среднее значение углов
                weight_below = 0.5
                weight_above = 0.5
            else:
                # Линейная интерполяция по расстоянию - чем ближе точка, тем больше её вес
                weight_below = 1.0 - abs(distance - d_below) / abs(d_above - d_below)
                weight_above = 1.0 - abs(distance - d_above) / abs(d_above - d_below)
                
                # Нормализация весов, чтобы их сумма была равна 1
                total_weight = weight_below + weight_above
                weight_below /= total_weight
                weight_above /= total_weight
            
            # Интерполяция углов для плеча и локтя (стандартная линейная)
            shoulder_angle = weight_below * shoulder_angles[idx_below] + weight_above * shoulder_angles[idx_above]
            elbow_angle = weight_below * elbow_angles[idx_below] + weight_above * elbow_angles[idx_above]
            
            # 9. Специальная обработка для угла кисти (ось 2)
            
            # Анализ тенденции изменения угла кисти
            wrist_diffs = [wrist_angles[i+1] - wrist_angles[i] for i in range(len(wrist_angles)-1)]
            
            # Проверка на нелинейность - есть ли смена знака или нерегулярные изменения
            is_nonlinear = False
            for i in range(len(wrist_diffs)-1):
                if (wrist_diffs[i] > 0 and wrist_diffs[i+1] < 0) or (wrist_diffs[i] < 0 and wrist_diffs[i+1] > 0):
                    is_nonlinear = True
                    break
            
            if is_nonlinear:
                logger.info("Обнаружена нелинейность в углах кисти, применяем полиномиальную аппроксимацию")
                
                # Для нелинейных данных создаем полиномиальную модель 2-й степени
                # Собираем все точки для построения полинома
                x_points = cal_distances
                y_points = wrist_angles
                
                # Если достаточно точек для построения полинома 2 степени
                if len(x_points) >= 3:
                    try:
                        # Вычисляем коэффициенты полинома 2-й степени (парабола)
                        coeffs = np.polyfit(x_points, y_points, 2)
                        
                        # Функция для вычисления значения полинома
                        poly_func = np.poly1d(coeffs)
                        
                        # Рассчитываем угол кисти через полином
                        wrist_angle = float(poly_func(distance))
                        logger.info(f"Угол кисти через полином: {wrist_angle:.2f}°")
                    except Exception as e:
                        logger.error(f"Ошибка полиномиальной интерполяции: {e}")
                        # В случае ошибки используем стандартную интерполяцию
                        wrist_angle = weight_below * wrist_angles[idx_below] + weight_above * wrist_angles[idx_above]
                else:
                    # Недостаточно точек для полинома, используем обычную интерполяцию
                    wrist_angle = weight_below * wrist_angles[idx_below] + weight_above * wrist_angles[idx_above]
            else:
                # Линейная интерполяция для кисти
                wrist_angle = weight_below * wrist_angles[idx_below] + weight_above * wrist_angles[idx_above]
            
            logger.info(f"Интерполяция между точками {idx_below+1} и {idx_above+1} с весами {weight_below:.2f} и {weight_above:.2f}")
    
    # 10. Округление углов до целых градусов (точность до 1°)
    shoulder_angle = round(shoulder_angle)
    elbow_angle = round(elbow_angle)
    wrist_angle = round(wrist_angle)
    servo_base_angle = round(servo_base_angle)
    
    # 11. Проверка физических ограничений
    shoulder_angle = max(0, min(180, shoulder_angle) + 0)  # плечо
    elbow_angle = max(0, min(180, elbow_angle) + 1)  # локоть
    wrist_angle = max(0, min(180, wrist_angle) -5)  # кисть
    
    # 12. Получение углов захвата из конфигурации
    gripper_open = SERVO_CONFIG['gripper']['open_angle']
    gripper_close = SERVO_CONFIG['gripper']['close_angle']
    
    # 13. Итоговое логирование
    logger.info(f"Итоговые углы: основание={servo_base_angle}°, плечо={shoulder_angle}°, локоть={elbow_angle}°, кисть={wrist_angle}°")
    
    # 14. Формирование результата в требуемом формате
    return {
        'angles': [
            180,  # не используется (исторически)
            180,  # сервопривод 1 - не задействован здесь
            wrist_angle,  # сервопривод 2 - кисть
            elbow_angle,  # сервопривод 3 - локоть
            shoulder_angle,  # сервопривод 4 - плечо
            servo_base_angle,  # сервопривод 5 - поворот основания
        ],
        # Дополнительные параметры для совместимости
        'base_angle': servo_base_angle,
        'wrist_angle': wrist_angle,
        'elbow_angle': elbow_angle,
        'shoulder_angle': shoulder_angle,
        'distance': distance,
        'quadrant': quadrant,
        'status': 'success',
        'wrist': wrist_angle,
        'elbow': elbow_angle,
        'shoulder': shoulder_angle,
        'base': servo_base_angle,
        'gripper_open': gripper_open,
        'gripper_close': gripper_close
    }
