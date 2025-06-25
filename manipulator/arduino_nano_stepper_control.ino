#include <Arduino.h>
#include <Servo.h> // Добавляем библиотеку для сервопривода

// Управление четырьмя шаговыми двигателями Nema17 через драйверы DM542
// и одним сервоприводом MG996R (для захвата)
// Принимает команды от ESP32 через UART

// Определение пинов для всех двигателей (PUL+ и DIR+)
const int MOTOR_COUNT = 4; // Уменьшаем до 4 шаговых двигателей

// Назначение пинов для соответствия веб-интерфейсу
// Мотор 1: PUL+ на пине 2, DIR+ на пине 3
// Мотор 2: PUL+ на пине 4, DIR+ на пине 5
// Мотор 3: PUL+ на пине 6, DIR+ на пине 7
// Мотор 4: PUL+ на пине 8, DIR+ на пине 9 (бывший мотор 5)
// Сервопривод (захват): пин 10
const int PUL_PINS[MOTOR_COUNT] = {2, 4, 6, 8}; // Пины для сигналов PUL+
const int DIR_PINS[MOTOR_COUNT] = {3, 5, 7, 9}; // Пины для сигналов DIR+
const int SERVO_PIN = 10; // Пин для сервопривода

// Константы для движения
const int MAX_SPEED = 1000;  // Минимальная задержка в микросекундах (максимальная скорость)
const int MIN_SPEED = 3000;  // Максимальная задержка в микросекундах (минимальная скорость)
const int ACCEL_STEPS = 1000; // Количество шагов для разгона и торможения

// Константы для сервопривода
const int SERVO_MIN_ANGLE = 0;     // Минимальный угол сервопривода (закрытый захват)
const int SERVO_MAX_ANGLE = 180;   // Максимальный угол сервопривода (открытый захват)
const int SERVO_CENTER = 90;       // Среднее положение сервопривода

// Объект сервопривода
Servo gripper;

// Текущие позиции моторов
long motorPositions[MOTOR_COUNT] = {0, 0, 0, 0};
// Целевые позиции моторов
long targetPositions[MOTOR_COUNT] = {0, 0, 0, 0};
// Флаги движения моторов
bool motorsMoving[MOTOR_COUNT] = {false, false, false, false};
// Скорости моторов (задержка между шагами в микросекундах)
int motorSpeeds[MOTOR_COUNT] = {MAX_SPEED, MAX_SPEED, MAX_SPEED, MAX_SPEED};
// Текущая позиция сервопривода
int servoPosition = 90; // Начальная позиция

// Буфер для приема команд
String inputBuffer = "";
bool commandComplete = false;

// Установка светодиода индикации для отладки
const int LED_PIN = 13; // Встроенный светодиод на Arduino Nano

// Новые переменные для синхронизированного движения
bool syncMovementActive = false;
long maxStepsToMove = 0;

void setup() {
  // Инициализация пинов для всех двигателей
  for (int i = 0; i < MOTOR_COUNT; i++) {
    pinMode(PUL_PINS[i], OUTPUT);
    pinMode(DIR_PINS[i], OUTPUT);
    
    // Начальные значения
    digitalWrite(PUL_PINS[i], LOW);
    digitalWrite(DIR_PINS[i], LOW);
  }
  
  // Инициализация серийного порта (используем большую скорость для связи с ESP32)
  Serial.begin(115200);
  Serial.println("Инициализация системы управления...");
  
  // Индикатор для отладки
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  
  // Инициализация сервопривода
  Serial.println("Инициализация сервопривода...");
  gripper.attach(SERVO_PIN);
  delay(500); // Даем время на инициализацию сервопривода
  
  // Тестирование сервопривода
  Serial.println("Тестирование сервопривода...");
  // Сначала устанавливаем центральную позицию
  servoPosition = SERVO_CENTER;
  gripper.write(servoPosition);
  delay(1000);
  
  // Перемещаем в крайнее положение и возвращаем обратно
  Serial.println("Перемещение в минимальную позицию");
  setGripperPosition(SERVO_MIN_ANGLE);
  delay(1000);
  
  Serial.println("Перемещение в максимальную позицию");
  setGripperPosition(SERVO_MAX_ANGLE);
  delay(1000);
  
  Serial.println("Возврат в центральную позицию");
  setGripperPosition(SERVO_CENTER);
  delay(1000);
  
  Serial.println("Система управления готова: 4 шаговых двигателя и сервопривод для захвата");
  
  // Моргнем светодиодом при завершении инициализации
  digitalWrite(LED_PIN, HIGH);
  delay(500);
  digitalWrite(LED_PIN, LOW);
}

// Функция для выполнения шага в заданном направлении для конкретного мотора
void makeStep(int motorIndex, bool direction) {
  // Установка направления
  digitalWrite(DIR_PINS[motorIndex], direction ? HIGH : LOW);
  delayMicroseconds(5); // Минимальная задержка для установки направления
  
  // Формирование импульса
  digitalWrite(PUL_PINS[motorIndex], HIGH);
  delayMicroseconds(10); // Минимальная ширина импульса
  digitalWrite(PUL_PINS[motorIndex], LOW);
  delayMicroseconds(10); // Минимальная ширина импульса
  
  // Обновление позиции
  if (direction) {
    motorPositions[motorIndex]++;
  } else {
    motorPositions[motorIndex]--;
  }
}

// Функция для управления сервоприводом
void setGripperPosition(int position) {
  // Ограничиваем позицию в диапазоне допустимых значений
  position = constrain(position, SERVO_MIN_ANGLE, SERVO_MAX_ANGLE);
  
  // Добавляем отладочную информацию
  Serial.print("Попытка установить захват в позицию ");
  Serial.println(position);
  
  // Медленно перемещаем сервопривод в нужную позицию для предотвращения рывков
  // Это особенно важно для мощных сервоприводов, таких как MG996R
  int step = (position > servoPosition) ? 1 : -1;
  int currentPos = servoPosition;
  
  while (currentPos != position) {
    currentPos += step;
    gripper.write(currentPos);
    delay(15); // Небольшая задержка между шагами для плавного движения
  }
  
  // Обновляем текущую позицию
  servoPosition = position;
  
  Serial.print("Захват установлен в позицию ");
  Serial.println(position);
}

// Функция для тестирования сервопривода
void testServo() {
  Serial.println("Запуск теста сервопривода...");
  
  // Проверка напрямую через объект Servo без использования setGripperPosition
  // Центральная позиция
  Serial.println("Прямое управление - центр (90)");
  gripper.write(SERVO_CENTER);
  delay(1000);
  
  // Минимальная позиция
  Serial.println("Прямое управление - минимум (0)");
  gripper.write(SERVO_MIN_ANGLE);
  delay(1000);
  
  // Максимальная позиция
  Serial.println("Прямое управление - максимум (180)");
  gripper.write(SERVO_MAX_ANGLE);
  delay(1000);
  
  // Промежуточные позиции
  Serial.println("Прямое управление - 45 градусов");
  gripper.write(45);
  delay(1000);
  
  Serial.println("Прямое управление - 135 градусов");
  gripper.write(135);
  delay(1000);
  
  // Возврат в центральную позицию
  Serial.println("Прямое управление - возврат в центр (90)");
  gripper.write(SERVO_CENTER);
  delay(1000);
  
  // Обновляем записанную позицию
  servoPosition = SERVO_CENTER;
  
  Serial.println("Тест сервопривода завершен");
}

// Проверка входящих данных и их обработка в loop
void checkSerial() {
  while (Serial.available() > 0) {
    char inChar = (char)Serial.read();
    
    // Если получен символ новой строки, команда завершена
    if (inChar == '\n' || inChar == '\r') {
      commandComplete = true;
      break;
    }
    
    // Добавляем символ в буфер
    inputBuffer += inChar;
  }
}

// Новая функция: подтверждение выполнения команды
void confirmCommand(const String& command, bool success) {
  Serial.print("CMD:");
  Serial.print(command);
  Serial.print(",STATUS:");
  Serial.println(success ? "OK" : "FAIL");
}

// Новая функция: подготовка к синхронизированному движению
void prepareForSyncMovement() {
  // Рассчитываем максимальное количество шагов для любого мотора
  maxStepsToMove = 0;
  for (int i = 0; i < MOTOR_COUNT; i++) {
    long stepsToMove = abs(targetPositions[i] - motorPositions[i]);
    if (stepsToMove > maxStepsToMove) {
      maxStepsToMove = stepsToMove;
    }
  }
  
  syncMovementActive = maxStepsToMove > 0;
  Serial.print("Синхронное движение начато. Максимальное число шагов: ");
  Serial.println(maxStepsToMove);
}

// Новая функция: выполнение одного шага синхронизированного движения
bool performSyncMovementStep() {
  if (!syncMovementActive || maxStepsToMove <= 0) {
    return false;
  }
  
  for (int i = 0; i < MOTOR_COUNT; i++) {
    long stepsRemaining = targetPositions[i] - motorPositions[i];
    if (stepsRemaining != 0) {
      // Рассчитываем, должен ли мотор сделать шаг на этой итерации
      // для обеспечения равномерного движения
      long stepsToMove = abs(stepsRemaining);
      if (stepsToMove * maxStepsToMove / (maxStepsToMove - 1) > 
          stepsToMove * maxStepsToMove / maxStepsToMove) {
        bool direction = stepsRemaining > 0;
        makeStep(i, direction);
      }
    }
  }
  
  maxStepsToMove--;
  if (maxStepsToMove <= 0) {
    syncMovementActive = false;
    Serial.println("Синхронное движение завершено");
    return false;
  }
  
  return true;
}

// Обработка полученной команды
void processCommand() {
  // Команда должна быть не пустой
  if (inputBuffer.length() > 0) {
    // Кратковременно включаем светодиод для индикации получения команды
    digitalWrite(LED_PIN, HIGH);
    
    char cmd = inputBuffer.charAt(0);
    bool commandSuccess = true; // Флаг успешности выполнения команды
    
    // Команда перемещения мотора: M<номер_мотора>,<целевая_позиция>[,<скорость>]
    if (cmd == 'M' || cmd == 'm') {
      // Разбор команды
      int firstCommaIndex = inputBuffer.indexOf(',');
      if (firstCommaIndex > 1) {
        int motorIndex = inputBuffer.substring(1, firstCommaIndex).toInt() - 1;
        
        // Ищем вторую запятую (опциональный параметр скорости)
        int secondCommaIndex = inputBuffer.indexOf(',', firstCommaIndex + 1);
        
        long targetPosition;
        int motorSpeed = MAX_SPEED; // По умолчанию используем максимальную скорость
        
        if (secondCommaIndex > firstCommaIndex) {
          // Если есть второй параметр (скорость)
          targetPosition = inputBuffer.substring(firstCommaIndex + 1, secondCommaIndex).toInt();
          motorSpeed = inputBuffer.substring(secondCommaIndex + 1).toInt();
          
          // Ограничиваем скорость в допустимых пределах
          motorSpeed = constrain(motorSpeed, MAX_SPEED, MIN_SPEED);
        } else {
          // Только позиция, без параметра скорости
          targetPosition = inputBuffer.substring(firstCommaIndex + 1).toInt();
        }
        
        // Проверка на команду для захвата (используется номер 5, но обрабатывается как сервопривод)
        if (motorIndex == 4) {
          // Управление захватом (сервоприводом)
          Serial.print("Получена команда для сервопривода, позиция: ");
          Serial.println(targetPosition);
          
          // Проверка корректности значения
          if (targetPosition >= SERVO_MIN_ANGLE && targetPosition <= SERVO_MAX_ANGLE) {
            // Прямой контроль для проверки
            if (targetPosition == 999) {
              // Специальная команда для тестирования сервопривода
              testServo();
            } else {
              // Обычное управление позицией
              setGripperPosition(targetPosition);
            }
          } else {
            Serial.print("Ошибка: позиция вне допустимого диапазона ");
            Serial.print(SERVO_MIN_ANGLE);
            Serial.print("-");
            Serial.println(SERVO_MAX_ANGLE);
            commandSuccess = false;
          }
        } 
        // Проверка валидности номера мотора для шаговиков
        else if (motorIndex >= 0 && motorIndex < MOTOR_COUNT) {
          targetPositions[motorIndex] = targetPosition;
          motorsMoving[motorIndex] = true;
          
          // Сохраняем скорость для этого мотора (не влияет на сервопривод)
          motorSpeeds[motorIndex] = motorSpeed;
          
          Serial.print("Мотор ");
          Serial.print(motorIndex + 1);
          Serial.print(" будет перемещен в позицию ");
          Serial.print(targetPosition);
          Serial.print(" со скоростью ");
          Serial.println(motorSpeed);
        } else {
          Serial.println("Ошибка: неверный номер мотора");
          commandSuccess = false;
        }
      } else {
        Serial.println("Ошибка: неверный формат команды");
        commandSuccess = false;
      }
    }
    // Команда сброса позиции мотора: R<номер_мотора>
    else if (cmd == 'R' || cmd == 'r') {
      int motorIndex = inputBuffer.substring(1).toInt() - 1;
      
      // Проверка на команду для захвата
      if (motorIndex == 4) {
        // Сброс позиции сервопривода на среднюю
        setGripperPosition(90);
        Serial.println("Позиция захвата сброшена на 90 градусов");
      }
      // Проверка валидности номера мотора для шаговиков
      else if (motorIndex >= 0 && motorIndex < MOTOR_COUNT) {
        motorPositions[motorIndex] = 0;
        targetPositions[motorIndex] = 0;
        motorsMoving[motorIndex] = false;
        
        Serial.print("Позиция мотора ");
        Serial.print(motorIndex + 1);
        Serial.println(" сброшена в 0");
      } else {
        Serial.println("Ошибка: неверный номер мотора");
        commandSuccess = false;
      }
    }
    // Команда статуса моторов: S
    else if (cmd == 'S' || cmd == 's') {
      for (int i = 0; i < MOTOR_COUNT; i++) {
        Serial.print("Мотор ");
        Serial.print(i + 1);
        Serial.print(": позиция = ");
        Serial.print(motorPositions[i]);
        Serial.print(", целевая позиция = ");
        Serial.print(targetPositions[i]);
        Serial.print(", скорость = ");
        Serial.print(motorSpeeds[i]);
        Serial.print(", статус = ");
        Serial.println(motorsMoving[i] ? "движется" : "остановлен");
      }
      Serial.print("Захват: позиция = ");
      Serial.println(servoPosition);
    }
    // Новая команда: J<pos1>,<pos2>,<pos3>,<pos4>[,<speed>] - синхронное движение
    else if (cmd == 'J' || cmd == 'j') {
      // Парсим позиции для всех моторов
      int values[MOTOR_COUNT + 1]; // +1 для скорости
      int valueCount = 0;
      
      int startPos = 1;
      int commaPos = inputBuffer.indexOf(',', startPos);
      
      while (commaPos > 0 && valueCount < MOTOR_COUNT + 1) {
        values[valueCount++] = inputBuffer.substring(startPos, commaPos).toInt();
        startPos = commaPos + 1;
        commaPos = inputBuffer.indexOf(',', startPos);
      }
      
      if (startPos < inputBuffer.length()) {
        values[valueCount++] = inputBuffer.substring(startPos).toInt();
      }
      
      // Устанавливаем целевые позиции для всех моторов
      int speed = MAX_SPEED;
      if (valueCount > MOTOR_COUNT) {
        speed = values[MOTOR_COUNT];
        speed = constrain(speed, MAX_SPEED, MIN_SPEED);
      }
      
      for (int i = 0; i < min(valueCount, MOTOR_COUNT); i++) {
        targetPositions[i] = values[i];
        motorsMoving[i] = true;
        motorSpeeds[i] = speed;
      }
      
      // Активируем синхронное движение
      prepareForSyncMovement();
    }
    // Новая команда: T<траектория> - передача траектории (зарезервировано для будущего использования)
    else if (cmd == 'T' || cmd == 't') {
      // Пока просто подтверждаем получение
      Serial.println("Команда траектории получена, но пока не реализована");
    }
    else {
      Serial.println("Неизвестная команда");
      commandSuccess = false;
    }
    
    // Отправляем подтверждение выполнения команды
    confirmCommand(inputBuffer, commandSuccess);
    
    // Выключаем светодиод через 100 мс
    delay(100);
    digitalWrite(LED_PIN, LOW);
  }
  
  // Очистка буфера
  inputBuffer = "";
  commandComplete = false;
}

void loop() {
  // Проверка входящих данных с Serial
  checkSerial();
  
  // Обработка команды, если она получена
  if (commandComplete) {
    processCommand();
  }
  
  // Приоритет синхронизированному движению
  if (syncMovementActive) {
    if (performSyncMovementStep()) {
      // Используем общую задержку для всех моторов
      delayMicroseconds(motorSpeeds[0]);
      return; // Пропускаем обычное движение, если выполняется синхронное
    }
  }
  
  // Обработка движения для каждого мотора
  for (int i = 0; i < MOTOR_COUNT; i++) {
    if (motorsMoving[i]) {
      // Определяем направление
      long stepsRemaining = targetPositions[i] - motorPositions[i];
      
      // Если достигли целевой позиции, останавливаем мотор
      if (stepsRemaining == 0) {
        motorsMoving[i] = false;
        Serial.print("Мотор ");
        Serial.print(i + 1);
        Serial.println(" достиг целевой позиции");
      } else {
        // Иначе делаем шаг в нужном направлении
        bool direction = stepsRemaining > 0;
        makeStep(i, direction);
        
        // Небольшая задержка между шагами для регулировки скорости
        // Используем индивидуальную скорость для каждого двигателя
        delayMicroseconds(motorSpeeds[i]);
      }
    }
  }
} 