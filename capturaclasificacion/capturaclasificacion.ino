// =====================================================================
//  Inferencia en el ESP32: 1D-CNN int8 + MPU6050 (FIFO a 1 kHz)
// =====================================================================
// Repite lo que hizo el entrenamiento, ventana por ventana:
//   1) toma CAPTURA muestras continuas a 1 kHz (ax, ay, az en m/s^2)
//   2) las corta en ventanas de 256 muestras con salto de 128 (7 ventanas si CAPTURA = 1024)
//   3) a cada ventana le resta el promedio de cada eje, divide por ESCALA y cuantiza a int8
//   4) corre la red y se queda con la clase mas probable
//   5) decide por mayoria entre las ventanas de la captura
//

#include <Wire.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"

// ---------------------------------------------------------------- parametros del entrenamiento
// Valores de normalizacion.json. 
const float ESCALA = 0.6668329238891602f;


const int MUESTRAS = 256;     // lo que ve la red
const int PASO = 128;         // salto entre ventanas 
const int CAPTURA = 1024;     // 1024 = 7 ventanas + voto | 256 = una sola ventana
const int N_CLASES = 3;
const char *NOMBRES[N_CLASES] = {"APAGADO", "NORMAL", "ANOMALO"};

// ---------------------------------------------------------------- sensor (igual que captura_256.ino)
const uint8_t MPU_ADDR = 0x68;
const int BYTES_PER_SAMPLE = 6;
const uint16_t FIFO_NEAR_FULL = 1000;
const int CHUNK_SAMPLES = 10;
const uint32_t CAPTURE_TIMEOUT_MS = (uint32_t)CAPTURA * 3 / 2 + 200;
const float ACCEL_SCALE = (4.0 * 9.80665) / 32768.0;   // +-4 g -> m/s^2

float buf[CAPTURA][3];        // ax, ay, az de la captura

void writeRegister(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission(true);
}

bool readRegisters(uint8_t reg, uint8_t *dst, uint8_t len) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, len, (uint8_t)true) != len) return false;
  for (uint8_t i = 0; i < len; i++) dst[i] = Wire.read();
  return true;
}

void setupMPU6050() {
  Wire.begin(21, 22);
  Wire.setClock(400000);
  Wire.setTimeOut(1000);

  writeRegister(0x6B, 0x01);   // despierta el chip
  delay(50);
  writeRegister(0x1A, 0x01);   // DLPF = 184 Hz -> base interna 1 kHz
  writeRegister(0x19, 0x00);   // SMPLRT_DIV = 0 -> fs = 1000 Hz
  writeRegister(0x1C, 0x08);   // acelerometro +-4 g
  writeRegister(0x23, 0x00);   // FIFO sin escribir
  writeRegister(0x6A, 0x04);   // reset FIFO
  delay(10);
}

// Llena buf con CAPTURA muestras continuas. Devuelve false si algo fallo.
bool capturar() {
  writeRegister(0x23, 0x00);
  writeRegister(0x6A, 0x04);   // reset FIFO
  delay(2);
  writeRegister(0x6A, 0x40);   // habilita FIFO
  writeRegister(0x23, 0x08);   // solo acelerometro XYZ

  int leidas = 0;
  bool overflow = false, i2cError = false;
  uint32_t t0 = millis();

  while (leidas < CAPTURA && millis() - t0 < CAPTURE_TIMEOUT_MS) {
    uint8_t c[2];
    if (!readRegisters(0x72, c, 2)) { i2cError = true; break; }
    uint16_t fifoCount = ((uint16_t)(c[0] & 0x1F) << 8) | c[1];
    if (fifoCount >= FIFO_NEAR_FULL) { overflow = true; break; }

    int disponibles = fifoCount / BYTES_PER_SAMPLE;
    if (disponibles > CHUNK_SAMPLES) disponibles = CHUNK_SAMPLES;
    if (disponibles > CAPTURA - leidas) disponibles = CAPTURA - leidas;

    if (disponibles > 0) {
      uint8_t raw[CHUNK_SAMPLES * BYTES_PER_SAMPLE];
      if (!readRegisters(0x74, raw, disponibles * BYTES_PER_SAMPLE)) { i2cError = true; break; }
      for (int i = 0; i < disponibles; i++) {
        int16_t rx = (int16_t)((raw[6 * i + 0] << 8) | raw[6 * i + 1]);
        int16_t ry = (int16_t)((raw[6 * i + 2] << 8) | raw[6 * i + 3]);
        int16_t rz = (int16_t)((raw[6 * i + 4] << 8) | raw[6 * i + 5]);
        buf[leidas][0] = rx * ACCEL_SCALE;
        buf[leidas][1] = ry * ACCEL_SCALE;
        buf[leidas][2] = rz * ACCEL_SCALE;
        leidas++;
      }
    }
    delay(1);
  }
  uint32_t ms = millis() - t0;

  writeRegister(0x23, 0x00);
  writeRegister(0x6A, 0x04);
  writeRegister(0x6A, 0x00);

  if (overflow) { Serial.println("[ERROR] Desborde de la FIFO: se descarta esta captura."); return false; }
  if (i2cError) { Serial.println("[ERROR] Fallo de I2C: se descarta esta captura."); return false; }
  if (leidas < CAPTURA) { Serial.println("[ERROR] Tiempo agotado: se descarta esta captura."); return false; }
  Serial.printf("[captura] %d muestras en %lu ms (esperado ~%d ms)\n", leidas, (unsigned long)ms, CAPTURA);
  return true;
}

// ---------------------------------------------------------------- red neuronal
constexpr int ARENA_BYTES = 60 * 1024;           // si AllocateTensors falla, subir este valor
alignas(16) static uint8_t tensor_arena[ARENA_BYTES];

const tflite::Model *modelo = nullptr;
tflite::MicroInterpreter *interprete = nullptr;
TfLiteTensor *entrada = nullptr;
TfLiteTensor *salida = nullptr;

bool iniciarModelo() {
  modelo = tflite::GetModel(model_data);
  if (modelo->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("[ERROR] Version de esquema %d, se esperaba %d.\n",
                  (int)modelo->version(), (int)TFLITE_SCHEMA_VERSION);
    return false;
  }
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter interp(modelo, resolver, tensor_arena, ARENA_BYTES);
  interprete = &interp;

  if (interprete->AllocateTensors() != kTfLiteOk) {
    Serial.println("[ERROR] AllocateTensors fallo: aumentar ARENA_BYTES.");
    return false;
  }
  entrada = interprete->input(0);
  salida = interprete->output(0);

  if (entrada->type != kTfLiteInt8 || entrada->dims->size != 3 ||
      entrada->dims->data[1] != MUESTRAS || entrada->dims->data[2] != 3) {
    Serial.println("[ERROR] La entrada del modelo no es int8 de forma (1, 256, 3).");
    return false;
  }
  if (salida->type != kTfLiteInt8 || salida->dims->data[salida->dims->size - 1] != N_CLASES) {
    Serial.println("[ERROR] La salida del modelo no es int8 con 3 clases.");
    return false;
  }
  Serial.printf("[modelo] arena usada: %u de %d bytes\n", (unsigned)interprete->arena_used_bytes(), ARENA_BYTES);
  Serial.printf("[modelo] entrada: scale=%.6f zero_point=%d (debe coincidir con normalizacion.json)\n",
                entrada->params.scale, (int)entrada->params.zero_point);
  return true;
}

// Clasifica la ventana de MUESTRAS muestras que empieza en buf[ini]. Devuelve la clase.
// *conf = probabilidad de esa clase (0..1).
int clasificarVentana(int ini, float *conf) {
  // 1) promedio de cada eje en la ventana (quita la gravedad), igual que en el entrenamiento
  float media[3] = {0, 0, 0};
  for (int i = 0; i < MUESTRAS; i++)
    for (int a = 0; a < 3; a++) media[a] += buf[ini + i][a];
  for (int a = 0; a < 3; a++) media[a] /= MUESTRAS;

  // 2) (v - media) / ESCALA, y cuantizacion a int8: q = round(x / scale + zero_point)
  const float inv = 1.0f / (ESCALA * entrada->params.scale);
  const int zp = entrada->params.zero_point;
  int8_t *dst = entrada->data.int8;
  for (int i = 0; i < MUESTRAS; i++) {
    for (int a = 0; a < 3; a++) {
      long q = lroundf((buf[ini + i][a] - media[a]) * inv + zp);
      if (q > 127) q = 127;
      if (q < -128) q = -128;
      dst[i * 3 + a] = (int8_t)q;               // orden: muestra 0 (ax, ay, az), muestra 1, ...
    }
  }

  // 3) inferencia
  if (interprete->Invoke() != kTfLiteOk) { *conf = 0; return -1; }

  // 4) clase con mayor salida
  int mejor = 0;
  for (int c = 1; c < N_CLASES; c++)
    if (salida->data.int8[c] > salida->data.int8[mejor]) mejor = c;
  *conf = (salida->data.int8[mejor] - salida->params.zero_point) * salida->params.scale;
  return mejor;
}

void clasificarCaptura() {
  int votos[N_CLASES] = {0, 0, 0};
  int total = 0;
  float confSuma = 0;
  uint32_t t0 = micros();
  Serial.print("  ventanas:");
  for (int ini = 0; ini + MUESTRAS <= CAPTURA; ini += PASO) {
    float conf;
    int c = clasificarVentana(ini, &conf);
    if (c < 0) { Serial.print(" ?"); continue; }
    votos[c]++;
    total++;
    confSuma += conf;
    Serial.printf(" %d", c);
  }
  uint32_t us = micros() - t0;
  if (total == 0) { Serial.println("\n  [ERROR] Invoke fallo en todas las ventanas."); return; }

  int ganador = 0;
  for (int c = 1; c < N_CLASES; c++)
    if (votos[c] > votos[ganador]) ganador = c;
  Serial.printf("\n  >>> %s  (%d de %d ventanas, confianza media %.2f, inferencia %lu ms)\n",
                NOMBRES[ganador], votos[ganador], total, confSuma / total, (unsigned long)(us / 1000));
}

void setup() {
  Serial.begin(115200);
  while (!Serial) delay(10);
  setupMPU6050();
  Serial.println("\n=== Inferencia 1D-CNN int8 en el ESP32 ===");
  if (!iniciarModelo()) {
    Serial.println("No se pudo iniciar el modelo. Se detiene.");
    while (true) delay(1000);
  }
}

void loop() {
  if (capturar()) clasificarCaptura();
  else delay(200);
}
