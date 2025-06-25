#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for handling voice commands using the Vosk library.
"""
import vosk
import pyaudio
import threading
import json
import logging
import queue
import time

# Настройка логирования
logger = logging.getLogger(__name__)


class VoiceCommandHandler:
    """
    Обработчик голосовых команд для управления манипулятором.
    Распознает команды и кладет их в очередь для дальнейшей обработки.
    """

    def __init__(self, command_queue: queue.Queue, model_path="models/vosk/vosk-model-small-ru-0.22"):
        """
        Инициализация обработчика голосовых команд.

        Args:
            command_queue (queue.Queue): Очередь для отправки распознанных команд.
            model_path (str): Путь к модели распознавания речи Vosk.
        """
        self.command_queue = command_queue
        self.model_path = model_path
        self.sample_rate = 16000
        self.is_listening = False
        self.listening_thread = None
        self.stop_event = threading.Event()

        self.vosk_model = None
        self.recognizer = None
        self.audio = None

        self._initialize_vosk()
        self._initialize_keywords()

    def _initialize_vosk(self):
        """Инициализация Vosk и PyAudio."""
        logger.info("Инициализация голосового управления...")
        try:
            self.vosk_model = vosk.Model(self.model_path)
            logger.info(f"Голосовая модель Vosk загружена из {self.model_path}")
            self.recognizer = vosk.KaldiRecognizer(self.vosk_model, self.sample_rate)
            self.audio = pyaudio.PyAudio()
            logger.info("Vosk и PyAudio успешно инициализированы")
        except Exception as e:
            logger.error(f"Ошибка при инициализации Vosk: {e}")
            raise

    def _initialize_keywords(self):
        """Инициализация словарей с ключевыми словами для распознавания команд."""
        self.action_keywords = {
            'захвати': 'grab', 'возьми': 'grab', 'подбери': 'grab', 'подними': 'grab',
            'схвати': 'grab', 'хватай': 'grab', 'взять': 'grab', 'поднять': 'grab',
            'забери': 'grab', 'захват': 'grab', 'старт': 'grab', 'начать': 'grab',
            'запуск': 'grab', 'активируй': 'grab'
        }
        self.color_keywords = {
            'красный': 'red', 'красного': 'red', 'красную': 'red', 'красн': 'red',
            'красны': 'red', 'красненький': 'red',
            'жёлтый': 'yellow', 'желтый': 'yellow', 'жёлтого': 'yellow', 'желтого': 'yellow',
            'жёлтую': 'yellow', 'желтую': 'yellow', 'жёлт': 'yellow', 'желт': 'yellow',
            'жёлты': 'yellow', 'желты': 'yellow', 'жёлтенький': 'yellow', 'желтенький': 'yellow',
            'зелёный': 'green', 'зеленый': 'green', 'зелёного': 'green', 'зеленого': 'green',
            'зелёную': 'green', 'зеленую': 'green', 'зелён': 'green', 'зелёны': 'green',
            'зелен': 'green', 'зелены': 'green', 'зелёненький': 'green', 'зелененький': 'green'
        }
        self.stop_keywords = {
            'стоп': 'stop', 'остановись': 'stop', 'останови': 'stop', 'отмена': 'stop',
            'хватит': 'stop', 'прекрати': 'stop', 'достаточно': 'stop', 'закончить': 'stop',
            'отключи': 'stop', 'выключи': 'stop', 'отбой': 'stop', 'сброс': 'stop',
            'домой': 'home'
        }
        logger.info("Словари ключевых слов для голосового управления инициализированы.")

    def start_listening(self):
        """Запуск прослушивания голосовых команд в отдельном потоке."""
        if self.is_listening:
            logger.info("Голосовое управление уже активно.")
            return
        if not self.recognizer:
            logger.error("Vosk не инициализирован. Прослушивание невозможно.")
            return

        self.stop_event.clear()
        self.is_listening = True
        self.listening_thread = threading.Thread(target=self._listen_in_background, daemon=True)
        self.listening_thread.start()
        logger.info("Активировано прослушивание голосовых команд.")

    def stop_listening(self):
        """Остановка прослушивания голосовых команд."""
        if not self.is_listening:
            logger.info("Голосовое управление уже неактивно.")
            return

        self.stop_event.set()
        self.is_listening = False
        if self.listening_thread:
            self.listening_thread.join(timeout=1.0)
        logger.info("Прослушивание голосовых команд деактивировано.")

    def _listen_in_background(self):
        """Фоновый процесс прослушивания микрофона и распознавания речи."""
        logger.info("Запущен фоновый поток прослушивания.")
        stream = self.audio.open(format=pyaudio.paInt16,
                                 channels=1,
                                 rate=self.sample_rate,
                                 input=True,
                                 frames_per_buffer=8000)
        stream.start_stream()

        try:
            while not self.stop_event.is_set():
                data = stream.read(4000, exception_on_overflow=False)
                if self.recognizer.AcceptWaveform(data):
                    result_json = self.recognizer.Result()
                    result = json.loads(result_json)
                    if result.get("text"):
                        text = result["text"]
                        logger.info(f"Распознана фраза: '{text}'")
                        parsed_command = self._parse_command(text.lower())
                        if parsed_command:
                            logger.info(f"Сформирована команда: {parsed_command}")
                            self.command_queue.put(parsed_command)
        except Exception as e:
            logger.error(f"Ошибка в потоке прослушивания: {e}")
        finally:
            stream.stop_stream()
            stream.close()
            logger.info("Фоновый поток прослушивания завершен.")

    def _parse_command(self, command_text: str) -> dict | None:
        """
        Анализирует текст и преобразует его в структурированную команду.

        Args:
            command_text: Распознанный текст.

        Returns:
            Словарь с командой или None, если команда не распознана.
        """
        words = command_text.split()
        action = None
        color = None

        for word in words:
            if word in self.action_keywords:
                action = self.action_keywords.get(word)
            elif word in self.color_keywords:
                color = self.color_keywords.get(word)
            elif word in self.stop_keywords:
                stop_action = self.stop_keywords.get(word)
                if stop_action == 'home':
                    return {'action': 'home'}
                else:  # 'stop', 'отмена' и т.д.
                    return {'action': 'stop'}

        if action == 'grab' and color:
            return {'action': 'grab', 'target': color}

        logger.warning(f"Не удалось распознать известную команду в фразе: '{command_text}'")
        return None

    def __del__(self):
        """Корректное завершение работы при удалении объекта."""
        self.stop_listening()
        if self.audio:
            self.audio.terminate() 