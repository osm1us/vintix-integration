#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Вспомогательные функции для системы компьютерного зрения и манипулятора
"""

import logging
import sys
import signal

def setup_logging(name="robot", level=logging.INFO):
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

class GracefulShutdown:
    """
    Handles SIGINT and SIGTERM to allow for a graceful shutdown.
    """
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._shutdown_handler)
        signal.signal(signal.SIGTERM, self._shutdown_handler)
        self.logger = logging.getLogger(__name__)

    def _shutdown_handler(self, signum, frame):
        self.logger.info(f"Получен сигнал завершения ({signal.Signals(signum).name}). Инициируется остановка.")
        self.shutdown_requested = True

    def is_shutting_down(self):
        """
        Returns True if a shutdown signal has been received.
        """
        return self.shutdown_requested
