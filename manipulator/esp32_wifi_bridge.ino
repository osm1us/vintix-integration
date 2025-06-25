#include <WiFi.h>
#include <WiFiClient.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <SPIFFS.h>
#include <queue>

// Настройки WiFi
const char *ssid = "Upvel";      // Измените на имя вашей сети
const char *password = "de3cDD4632f7";  // Измените на пароль вашей сети

// Настройки сервера
WebServer server(80);
WiFiClient client;

// Настройки UART для связи с Arduino Nano
#define RX_PIN 16  // Пин RX для UART
#define TX_PIN 17  // Пин TX для UART
#define UART_SPEED 115200

// Очередь команд для Arduino
std::queue<String> commandQueue;
unsigned long lastCommandTime = 0;
const int COMMAND_INTERVAL = 50; // Минимальный интервал между командами (мс)

// Статус системы
struct SystemStatus {
  bool connected = false;
  long motorPositions[4] = {0, 0, 0, 0};
  int gripperPosition = 90;
  bool motorsMoving[4] = {false, false, false, false};
  String lastError = "";
  String lastCommand = "";
  unsigned long lastUpdateTime = 0;
} status;

// Буфер для приема данных от Arduino
String serialBuffer = "";

void setup() {
  // Инициализация последовательных портов
  Serial.begin(115200); // Для отладки
  Serial2.begin(UART_SPEED, SERIAL_8N1, RX_PIN, TX_PIN); // Для связи с Arduino
  
  Serial.println("Инициализация ESP32 WiFi Bridge...");
  
  // Инициализация SPIFFS для хранения веб-файлов
  if(!SPIFFS.begin(true)) {
    Serial.println("Ошибка монтирования SPIFFS");
    return;
  }
  
  // Подключение к WiFi
  WiFi.begin(ssid, password);
  Serial.print("Подключение к WiFi");
  
  // Ожидание подключения
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("\nНе удалось подключиться к WiFi. Перезагрузите устройство.");
    // Можно добавить режим точки доступа как запасной вариант
    return;
  }
  
  Serial.println("");
  Serial.println("WiFi подключен");
  Serial.print("IP адрес: ");
  Serial.println(WiFi.localIP());
  
  // Настройка маршрутов веб-сервера
  setupServerRoutes();
  
  // Запуск веб-сервера
  server.begin();
  Serial.println("Веб-сервер запущен");
  
  // Отправка тестовой команды на Arduino
  Serial.println("Отправка запроса статуса на Arduino...");
  sendCommandToArduino("S");
}

void setupServerRoutes() {
  // Корневой маршрут - отображение веб-интерфейса
  server.on("/", HTTP_GET, []() {
    if (SPIFFS.exists("/index.html")) {
      server.sendHeader("Cache-Control", "max-age=3600");
      server.send(200, "text/html", loadFromSPIFFS("/index.html"));
    } else {
      // Если файл не найден, отправляем встроенный HTML
      server.send(200, "text/html", generateDefaultHtml());
    }
  });
  
  // API для получения статуса системы
  server.on("/api/status", HTTP_GET, []() {
    DynamicJsonDocument doc(1024);
    doc["connected"] = status.connected;
    for (int i = 0; i < 4; i++) {
      doc["motors"][i]["position"] = status.motorPositions[i];
      doc["motors"][i]["moving"] = status.motorsMoving[i];
    }
    doc["gripper"] = status.gripperPosition;
    doc["last_error"] = status.lastError;
    doc["last_command"] = status.lastCommand;
    doc["timestamp"] = millis();
    
    String response;
    serializeJson(doc, response);
    server.send(200, "application/json", response);
  });
  
  // API для отправки команды на Arduino
  server.on("/api/command", HTTP_POST, []() {
    if (!server.hasArg("plain")) {
      server.send(400, "text/plain", "Body required");
      return;
    }
    
    String body = server.arg("plain");
    DynamicJsonDocument doc(1024);
    DeserializationError error = deserializeJson(doc, body);
    
    if (error) {
      server.send(400, "text/plain", "Invalid JSON");
      return;
    }
    
    if (!doc.containsKey("command")) {
      server.send(400, "text/plain", "Command required");
      return;
    }
    
    String command = doc["command"].as<String>();
    sendCommandToArduino(command);
    
    server.send(200, "application/json", "{\"status\":\"queued\"}");
  });
  
  // API для приема траектории от ROS 2
  server.on("/api/trajectory", HTTP_POST, []() {
    if (!server.hasArg("plain")) {
      server.send(400, "text/plain", "Body required");
      return;
    }
    
    String body = server.arg("plain");
    DynamicJsonDocument doc(4096);
    DeserializationError error = deserializeJson(doc, body);
    
    if (error) {
      server.send(400, "text/plain", "Invalid JSON");
      return;
    }
    
    if (!doc.containsKey("commands")) {
      server.send(400, "text/plain", "Commands array required");
      return;
    }
    
    JsonArray commands = doc["commands"];
    for (JsonVariant cmd : commands) {
      commandQueue.push(cmd.as<String>());
    }
    
    server.send(200, "application/json", "{\"status\":\"queued\",\"count\":" + String(commands.size()) + "}");
  });
  
  // Отдача статических файлов из SPIFFS
  server.on("/css/style.css", HTTP_GET, []() {
    server.sendHeader("Cache-Control", "max-age=3600");
    server.send(200, "text/css", loadFromSPIFFS("/css/style.css"));
  });
  
  server.on("/js/app.js", HTTP_GET, []() {
    server.sendHeader("Cache-Control", "max-age=3600");
    server.send(200, "text/javascript", loadFromSPIFFS("/js/app.js"));
  });
  
  // Обработка 404
  server.onNotFound([]() {
    server.send(404, "text/plain", "Not found");
  });
}

String loadFromSPIFFS(String path) {
  if (SPIFFS.exists(path)) {
    File file = SPIFFS.open(path, "r");
    if (file) {
      String content = file.readString();
      file.close();
      return content;
    }
  }
  return "";
}

String generateDefaultHtml() {
  // Простой встроенный HTML-интерфейс если файлы в SPIFFS отсутствуют
  return R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Robot Control</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
    .container { max-width: 800px; margin: 0 auto; }
    .status { background: #f5f5f5; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
    .controls { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
    .control-card { background: #fff; border: 1px solid #ddd; border-radius: 5px; padding: 10px; }
    button { background: #4CAF50; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
    button:hover { background: #45a049; }
    input[type="range"] { width: 100%; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Robot Control Interface</h1>
    
    <div class="status" id="status">
      <h2>Status</h2>
      <p>Connected: <span id="connected">checking...</span></p>
      <p>Motors: <span id="motors">-</span></p>
      <p>Gripper: <span id="gripper">-</span></p>
    </div>
    
    <div class="controls">
      <div class="control-card">
        <h3>Motor 1</h3>
        <input type="range" min="-1000" max="1000" value="0" id="motor1">
        <p>Position: <span id="motor1Value">0</span></p>
        <button onclick="moveMotor(1, document.getElementById('motor1').value)">Move</button>
        <button onclick="resetMotor(1)">Reset</button>
      </div>
      
      <div class="control-card">
        <h3>Motor 2</h3>
        <input type="range" min="-1000" max="1000" value="0" id="motor2">
        <p>Position: <span id="motor2Value">0</span></p>
        <button onclick="moveMotor(2, document.getElementById('motor2').value)">Move</button>
        <button onclick="resetMotor(2)">Reset</button>
      </div>
      
      <div class="control-card">
        <h3>Motor 3</h3>
        <input type="range" min="-1000" max="1000" value="0" id="motor3">
        <p>Position: <span id="motor3Value">0</span></p>
        <button onclick="moveMotor(3, document.getElementById('motor3').value)">Move</button>
        <button onclick="resetMotor(3)">Reset</button>
      </div>
      
      <div class="control-card">
        <h3>Motor 4</h3>
        <input type="range" min="-1000" max="1000" value="0" id="motor4">
        <p>Position: <span id="motor4Value">0</span></p>
        <button onclick="moveMotor(4, document.getElementById('motor4').value)">Move</button>
        <button onclick="resetMotor(4)">Reset</button>
      </div>
      
      <div class="control-card">
        <h3>Gripper</h3>
        <input type="range" min="0" max="180" value="90" id="gripper-control">
        <p>Position: <span id="gripperValue">90</span></p>
        <button onclick="moveGripper(document.getElementById('gripper-control').value)">Set</button>
      </div>
    </div>
    
    <div style="margin-top: 20px;">
      <h3>Joint Movement</h3>
      <button onclick="syncMove()">Synchronized Move</button>
      <button onclick="sendStatus()">Get Status</button>
    </div>
  </div>
  
  <script>
    // Обновление отображаемых значений при перемещении ползунков
    document.querySelectorAll('input[type="range"]').forEach(function(input) {
      var valueDisplay = document.getElementById(input.id + 'Value');
      if (valueDisplay) {
        input.addEventListener('input', function() {
          valueDisplay.textContent = input.value;
        });
      }
    });
    
    // Периодическое обновление статуса
    function updateStatus() {
      fetch('/api/status')
        .then(function(response) { return response.json(); })
        .then(function(data) {
          document.getElementById('connected').textContent = data.connected ? 'Yes' : 'No';
          
          var motorsText = '';
          for (var i = 0; i < data.motors.length; i++) {
            motorsText += 'M' + (i+1) + ': ' + data.motors[i].position + ' (' + 
                         (data.motors[i].moving ? 'moving' : 'stopped') + ') ';
          }
          document.getElementById('motors').textContent = motorsText;
          document.getElementById('gripper').textContent = data.gripper;
        })
        .catch(function(error) { console.error('Error fetching status:', error); });
    }
    
    // Функции управления
    function moveMotor(id, position) {
      sendCommand('M' + id + ',' + position);
    }
    
    function resetMotor(id) {
      sendCommand('R' + id);
    }
    
    function moveGripper(position) {
      sendCommand('M5,' + position);
    }
    
    function syncMove() {
      var pos1 = document.getElementById('motor1').value;
      var pos2 = document.getElementById('motor2').value;
      var pos3 = document.getElementById('motor3').value;
      var pos4 = document.getElementById('motor4').value;
      sendCommand('J' + pos1 + ',' + pos2 + ',' + pos3 + ',' + pos4);
    }
    
    function sendStatus() {
      sendCommand('S');
    }
    
    function sendCommand(command) {
      fetch('/api/command', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ command: command }),
      })
      .then(function(response) { return response.json(); })
      .then(function(data) { console.log('Command sent:', data); })
      .catch(function(error) { console.error('Error sending command:', error); });
    }
    
    // Обновление статуса каждые 2 секунды
    setInterval(updateStatus, 2000);
    updateStatus();
  </script>
</body>
</html>
)rawliteral";
}

void sendCommandToArduino(String command) {
  commandQueue.push(command);
  status.lastCommand = command;
}

void processSerialFromArduino() {
  while (Serial2.available()) {
    char c = Serial2.read();
    
    // Если получен конец строки, обрабатываем команду
    if (c == '\n' || c == '\r') {
      if (serialBuffer.length() > 0) {
        parseArduinoResponse(serialBuffer);
        serialBuffer = "";
      }
    } else {
      serialBuffer += c;
    }
  }
}

void parseArduinoResponse(String response) {
  Serial.print("Arduino >> ");
  Serial.println(response);
  
  // Обновляем флаг подключения
  status.connected = true;
  status.lastUpdateTime = millis();
  
  // Проверяем, если это статус мотора
  if (response.startsWith("Мотор ") && response.indexOf("позиция =") > 0) {
    int motorIdx = response.substring(6, 7).toInt() - 1;
    if (motorIdx >= 0 && motorIdx < 4) {
      // Извлекаем позицию мотора
      int posStart = response.indexOf("позиция = ") + 10;
      int posEnd = response.indexOf(",", posStart);
      if (posEnd > posStart) {
        status.motorPositions[motorIdx] = response.substring(posStart, posEnd).toInt();
      }
      
      // Извлекаем статус движения
      if (response.indexOf("статус = движется") > 0) {
        status.motorsMoving[motorIdx] = true;
      } else if (response.indexOf("статус = остановлен") > 0) {
        status.motorsMoving[motorIdx] = false;
      }
    }
  }
  // Проверяем, если это статус сервопривода
  else if (response.startsWith("Захват: позиция = ")) {
    status.gripperPosition = response.substring(18).toInt();
  }
  // Проверяем, если это подтверждение команды
  else if (response.startsWith("CMD:")) {
    // Формат: CMD:команда,STATUS:OK/FAIL
    if (response.indexOf("STATUS:FAIL") > 0) {
      status.lastError = response;
    }
  }
}

void checkConnectionStatus() {
  // Если нет ответа от Arduino более 5 секунд, считаем связь потерянной
  if (millis() - status.lastUpdateTime > 5000) {
    status.connected = false;
  }
}

void loop() {
  // Обработка запросов веб-сервера
  server.handleClient();
  
  // Чтение данных от Arduino
  processSerialFromArduino();
  
  // Отправка команд из очереди на Arduino
  if (!commandQueue.empty() && millis() - lastCommandTime > COMMAND_INTERVAL) {
    String command = commandQueue.front();
    commandQueue.pop();
    
    Serial.print("Отправка на Arduino: ");
    Serial.println(command);
    
    Serial2.println(command);
    lastCommandTime = millis();
  }
  
  // Проверка статуса подключения
  checkConnectionStatus();
  
  // Периодический запрос статуса, если нет активных команд
  static unsigned long lastStatusRequest = 0;
  if (commandQueue.empty() && millis() - lastStatusRequest > 5000) {
    sendCommandToArduino("S");
    lastStatusRequest = millis();
  }
} 