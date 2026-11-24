#include <Arduino.h>
#include <WiFi.h>
#include <PubSubClient.h> // Librería para MQTT
#include <ArduinoJson.h>
#include <WiFiManager.h>
#include <math.h>

// =================================================================
// CONFIGURACIÓN DEL BROKER MQTT (DOCKER / MOSQUITTO)
// =================================================================
// Cambia esta IP por la IP local del servidor donde corre Docker
const char* MQTT_SERVER = "192.168.1.8"; 
const int   MQTT_PORT   = 1883;
const char* MQTT_TOPIC  = "telemetria/mantenimiento";
const char* CLIENT_ID   = "ESP32_Predictivo_Client";

WiFiClient espClient;
PubSubClient mqttClient(espClient);

// =================================================================
// VARIABLES DEL SIMULADOR MPU6050 Y ESTADOS ML
// =================================================================
unsigned long lastSendTime = 0;
unsigned long lastStateChangeTime = 0;
const unsigned long SEND_INTERVAL_MS = 1000;       // Envío cada 1 segundo
const unsigned long STATE_CHANGE_INTERVAL = 12000; // Cambia de estado cada 12 segundos

int currentState = 0; // Estado simulado (0 a 4)

const char* STATE_NAMES[] = {
  "0: Apagado",
  "1: Normal Sin Carga",
  "2: Normal Con Carga",
  "3: Advertencia",
  "4: Anomala Critica"
};

// =================================================================
// GENERADOR DE DATOS SINTÉTICOS DE ACELERACIÓN
// =================================================================
void generateSyntheticData(int state, float &ax_rms, float &ay_rms, float &az_rms, 
                          float &ax_inst, float &ay_inst, float &az_inst) {
  float noise_x = ((float)random(-50, 50) / 1000.0);
  float noise_y = ((float)random(-50, 50) / 1000.0);
  float noise_z = ((float)random(-50, 50) / 1000.0);

  switch (state) {
    case 0: // Apagado
      ax_rms = 0.008 + noise_x * 0.1;
      ay_rms = 0.009 + noise_y * 0.1;
      az_rms = 1.000 + noise_z * 0.1; // 1g por gravedad
      break;

    case 1: // Normal Sin Carga
      ax_rms = 0.150 + noise_x;
      ay_rms = 0.120 + noise_y;
      az_rms = 1.050 + noise_z;
      break;

    case 2: // Normal Con Carga
      ax_rms = 0.280 + noise_x * 1.5;
      ay_rms = 0.220 + noise_y * 1.5;
      az_rms = 1.120 + noise_z * 1.5;
      break;

    case 3: // Advertencia (Desbalance/Holgura)
      ax_rms = 0.580 + noise_x * 2.5;
      ay_rms = 0.450 + noise_y * 2.5;
      az_rms = 1.350 + noise_z * 2.5;
      break;

    case 4: // Anómala Crítica (Golpeteo severo)
      ax_rms = 1.250 + noise_x * 4.0;
      ay_rms = 1.050 + noise_y * 4.0;
      az_rms = 1.850 + noise_z * 4.0;
      break;
  }

  float time_factor = millis() / 100.0;
  ax_inst = ax_rms * sin(time_factor) + noise_x;
  ay_inst = ay_rms * cos(time_factor) + noise_y;
  az_inst = az_rms + (noise_z * 0.5);
}

// =================================================================
// FUNCIÓN DE CONEXIÓN Y RECONEXIÓN MQTT
// =================================================================
void reconnectMQTT() {
  while (!mqttClient.connected()) {
    Serial.print("Intentando conexión MQTT...");
    
    if (mqttClient.connect(CLIENT_ID)) {
      Serial.println(" ¡Conectado al Broker MQTT!");
    } else {
      Serial.printf(" Falló, rc=%d. Reintentando en 5 segundos...\n", mqttClient.state());
      delay(5000);
    }
  }
}

// =================================================================
// FUNCIÓN DE ENVÍO VÍA MQTT
// =================================================================
void sendTelemetryMQTT(int estado, float ax_rms, float ay_rms, float az_rms, 
                        float ax_inst, float ay_inst, float az_inst) {
  if (!mqttClient.connected()) {
    reconnectMQTT();
  }
  mqttClient.loop();

  // Construcción del JSON
  StaticJsonDocument<256> doc;
  doc["estado"]     = estado;
  doc["ax_rms"]     = round(ax_rms * 1000) / 1000.0;
  doc["ay_rms"]     = round(ay_rms * 1000) / 1000.0;
  doc["az_rms"]     = round(az_rms * 1000) / 1000.0;
  doc["ax_instant"] = round(ax_inst * 1000) / 1000.0;
  doc["ay_instant"] = round(ay_inst * 1000) / 1000.0;
  doc["az_instant"] = round(az_inst * 1000) / 1000.0;

  char jsonBuffer[256];
  serializeJson(doc, jsonBuffer);

  // Publicar payload en el tópico MQTT
  if (mqttClient.publish(MQTT_TOPIC, jsonBuffer)) {
    Serial.printf("--> MQTT Pub [%s] | Trama: %s\n", STATE_NAMES[estado], jsonBuffer);
  } else {
    Serial.println("--> Error publicando mensaje en MQTT");
  }
}

// =================================================================
// SETUP Y CONFIGURACIÓN INICIAL
// =================================================================
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n=============================================");
  Serial.println(" ESP32-S3: SIMULADOR MQTT PARA PREDICTIVO");
  Serial.println("=============================================");

  WiFiManager wm;
  Serial.println("Iniciando portal Wi-Fi...");
  
  bool res = wm.autoConnect("ESP32-Predictive-Setup");

  if (!res) {
    Serial.println("Error de conexión o Timeout.");
    ESP.restart();
  } else {
    Serial.println("¡Conectado a Wi-Fi!");
    Serial.print("IP ESP32: ");
    Serial.println(WiFi.localIP());
  }

  // Configuración del cliente MQTT
  mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
  mqttClient.setBufferSize(512); // Ampliado para asegurar tramas JSON largas

  randomSeed(analogRead(0));
}

// =================================================================
// LOOP PRINCIPAL
// =================================================================
void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttClient.connected()) {
      reconnectMQTT();
    }
    mqttClient.loop();
  }

  unsigned long currentMillis = millis();

  // 1. Simular cambio de estado del motor (cada 12 segundos)
  if (currentMillis - lastStateChangeTime >= STATE_CHANGE_INTERVAL) {
    lastStateChangeTime = currentMillis;
    currentState = (currentState + 1) % 5;
    Serial.printf("\n[NUEVO ESTADO ML] --> %s\n\n", STATE_NAMES[currentState]);
  }

  // 2. Envío de telemetría por MQTT (cada 1 segundo)
  if (currentMillis - lastSendTime >= SEND_INTERVAL_MS) {
    lastSendTime = currentMillis;

    float ax_rms, ay_rms, az_rms;
    float ax_inst, ay_inst, az_inst;

    generateSyntheticData(currentState, ax_rms, ay_rms, az_rms, ax_inst, ay_inst, az_inst);
    sendTelemetryMQTT(currentState, ax_rms, ay_rms, az_rms, ax_inst, ay_inst, az_inst);
  }
}