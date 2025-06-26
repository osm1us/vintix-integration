import json
import logging
import pyaudio
import threading
import queue
from vosk import Model, KaldiRecognizer

logger = logging.getLogger(__name__)

# Enum для команд для большей надежности
class Command:
    PICK_UP = "PICK_UP"
    GO_HOME = "GO_HOME"
    STOP = "STOP"
    UNKNOWN = "UNKNOWN"

class VoiceControl:
    """
    Распознает голосовые команды в фоновом потоке и складывает их в очередь.
    """
    def __init__(self, model_path: str, device_index: int | None = None):
        """
        Инициализирует систему голосового управления.

        Args:
            model_path (str): Путь к директории с моделью Vosk.
            device_index (int, optional): Индекс аудиоустройства. None для устройства по умолчанию.
        """
        logger.info(f"Загрузка модели Vosk из: {model_path}")
        try:
            self.model = Model(model_path)
        except Exception as e:
            logger.critical(f"Не удалось загрузить модель Vosk. Убедитесь, что путь '{model_path}' корректен. Ошибка: {e}")
            raise

        self.p = pyaudio.PyAudio()
        self.device_index = device_index or self._find_default_input_device()
        if self.device_index is None:
            raise RuntimeError("Не найдено ни одного активного аудиовхода.")

        self.recognizer = KaldiRecognizer(self.model, 16000)
        self.command_queue = queue.Queue()
        
        self._is_running = False
        self._thread = None
        logger.info(f"Голосовое управление инициализировано на устройстве {self.device_index}.")

    def _find_default_input_device(self) -> int | None:
        """Находит первое доступное аудиоустройство для ввода."""
        logger.debug("Поиск аудиоустройств...")
        for i in range(self.p.get_device_count()):
            dev = self.p.get_device_info_by_index(i)
            if dev['maxInputChannels'] > 0:
                logger.info(f"Найдено аудиоустройство ввода: index={i}, name={dev['name']}")
                return i
        logger.error("Не найдено ни одного подходящего аудиоустройства.")
        return None

    def _listen_loop(self):
        """
        Основной цикл, который работает в фоновом потоке, слушает микрофон
        и кладет распознанные команды в очередь.
        """
        stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=4096,
            input_device_index=self.device_index
        )
        stream.start_stream()
        logger.info("Фоновое прослушивание голоса запущено.")
        
        while self._is_running:
            data = stream.read(4096, exception_on_overflow=False)
            if self.recognizer.AcceptWaveform(data):
                result_json = self.recognizer.Result()
                result_text = json.loads(result_json).get("text", "")
                
                if result_text:
                    logger.info(f"Распознан текст: '{result_text}'")
                    command = self.parse_command(result_text)
                    self.command_queue.put(command)
        
        stream.stop_stream()
        stream.close()
        logger.info("Фоновое прослушивание голоса остановлено.")

    def get_command(self) -> dict | None:
        """
        Неблокирующий метод для получения команды из очереди.
        Возвращает команду или None, если очередь пуста.
        """
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

    @staticmethod
    def parse_command(text: str) -> dict:
        """
        Парсит распознанный текст и возвращает структурированную команду,
        используя ключевые слова из файла конфигурации.
        """
        # Импортируем настройки здесь, чтобы избежать циклического импорта
        # и сохранить статический метод
        from config import settings
        
        words = set(text.lower().split())
        
        # Поиск команд (стоп, домой)
        for word in words:
            if word in settings.voice.KEYWORDS['command']:
                command_type = settings.voice.KEYWORDS['command'][word]
                if command_type == 'stop':
                    return {"type": Command.STOP, "data": None}
                elif command_type == 'home':
                    return {"type": Command.GO_HOME, "data": None}
        
        # Поиск сложной команды "захватить <цвет>"
        action_found = None
        color_found = None

        for word in words:
            if word in settings.voice.KEYWORDS['action']:
                action_found = settings.voice.KEYWORDS['action'][word]
            if word in settings.voice.KEYWORDS['target_color']:
                color_found = settings.voice.KEYWORDS['target_color'][word]

        if action_found == 'grab' and color_found:
            return {"type": Command.PICK_UP, "data": color_found}

        # Если ничего не подошло
        return {"type": Command.UNKNOWN, "data": text}

    def start(self):
        """Запускает фоновый поток прослушивания."""
        if self._is_running:
            logger.warning("Попытка запустить уже работающий VoiceControl.")
            return
            
        self._is_running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Останавливает прослушивание и освобождает ресурсы.
        """
        logger.info("Остановка голосового управления...")
        if not self._is_running:
            return
            
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=2) # Даем потоку время на завершение
            
        if hasattr(self, 'p'):
            self.p.terminate()
        logger.info("Ресурсы PyAudio освобождены.")