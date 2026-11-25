// Captura de segmentos CONTINUOS de SAMPLES_PER_BATCH muestras (ax, ay, az) a 1 kHz con la FIFO del MPU6050.
// Escribir una etiqueta (0, 1 o 2) a 460800 baud (el fin de linea no hace falta) para grabar un segmento con esa etiqueta.
// IMPORTANTE: SAMPLES_PER_BATCH debe ser igual a MUESTRAS_POR_VENTANA en crear_csv.py.
#include <Wire.h>

const uint8_t MPU_ADDR = 0x68;
const int SAMPLES_PER_BATCH = 1024;      // 1024 muestras = 1,024 s continuos a 1 kHz (antes 256)
const int BYTES_PER_SAMPLE = 6;          // solo acelerometro: 3 ejes x 2 bytes
const uint16_t FIFO_NEAR_FULL = 1000;    // la FIFO del MPU6050 tiene 1024 bytes
const int CHUNK_SAMPLES = 10;            // muestras por lectura I2C (60 bytes)
// Tiempo maximo para llenar el segmento: 1,5 ms por muestra + 200 ms de margen (antes fijo en 1000 ms).
const uint32_t CAPTURE_TIMEOUT_MS = (uint32_t)SAMPLES_PER_BATCH * 3 / 2 + 200;

struct Sample { float ax, ay, az; };
Sample ramBuffer[SAMPLES_PER_BATCH];

const float ACCEL_SCALE = (4.0 * 9.80665) / 32768.0;   // +-4 g -> m/s^2

void writeRegister(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission(true);
}

// Lee len bytes consecutivos a partir de reg. Devuelve false si falla.
bool readRegisters(uint8_t reg, uint8_t *dst, uint8_t len) {
  Wire.beginTransmission(MPU_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(MPU_ADDR, len, (uint8_t)true) != len) return false;
  for (uint8_t i = 0; i < len; i++) dst[i] = Wire.read();   // un Wire.read() por sentencia
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

void promptUser() {
  Serial.println("\n-----------------------------------------------------------");
  Serial.println("ESPERANDO ENTRADA: Ingrese una etiqueta (0, 1, 2, 3 o 4)...");
  Serial.println("-----------------------------------------------------------");
}

void captureAndSend(int label) {
  Serial.printf("\n[CAPTURANDO] %d muestras a 1 kHz, etiqueta [%d]...\n", SAMPLES_PER_BATCH, label);

  writeRegister(0x23, 0x00);
  writeRegister(0x6A, 0x04);   // reset FIFO
  delay(2);
  writeRegister(0x6A, 0x40);   // habilita FIFO
  writeRegister(0x23, 0x08);   // solo acelerometro XYZ

  int samplesRead = 0;
  bool overflow = false, i2cError = false;
  uint32_t t0 = millis();

  // La FIFO guarda como mucho 1024 bytes = 170 muestras de 6 bytes, y se
  // necesitan SAMPLES_PER_BATCH: hay que ir vaciandola MIENTRAS el sensor muestrea.
  while (samplesRead < SAMPLES_PER_BATCH && millis() - t0 < CAPTURE_TIMEOUT_MS) {
    uint8_t c[2];
    if (!readRegisters(0x72, c, 2)) { i2cError = true; break; }
    uint16_t fifoCount = ((uint16_t)(c[0] & 0x1F) << 8) | c[1];

    if (fifoCount >= FIFO_NEAR_FULL) { overflow = true; break; }   // se perdieron datos

    int available = fifoCount / BYTES_PER_SAMPLE;
    if (available > CHUNK_SAMPLES) available = CHUNK_SAMPLES;
    if (available > SAMPLES_PER_BATCH - samplesRead) available = SAMPLES_PER_BATCH - samplesRead;

    if (available > 0) {
      uint8_t raw[CHUNK_SAMPLES * BYTES_PER_SAMPLE];
      if (!readRegisters(0x74, raw, available * BYTES_PER_SAMPLE)) { i2cError = true; break; }
      for (int i = 0; i < available; i++) {
        int16_t rx = (int16_t)((raw[6 * i + 0] << 8) | raw[6 * i + 1]);
        int16_t ry = (int16_t)((raw[6 * i + 2] << 8) | raw[6 * i + 3]);
        int16_t rz = (int16_t)((raw[6 * i + 4] << 8) | raw[6 * i + 5]);
        ramBuffer[samplesRead].ax = rx * ACCEL_SCALE;
        ramBuffer[samplesRead].ay = ry * ACCEL_SCALE;
        ramBuffer[samplesRead].az = rz * ACCEL_SCALE;
        samplesRead++;
      }
    }
    delay(1);
  }
  uint32_t elapsed_ms = millis() - t0;

  writeRegister(0x23, 0x00);   // deja de escribir en la FIFO
  writeRegister(0x6A, 0x04);   // reset FIFO
  writeRegister(0x6A, 0x00);   // deshabilita FIFO

  if (overflow)  Serial.println("[ERROR] Desborde de la FIFO: se perdieron muestras. Repetir.");
  if (i2cError)  Serial.println("[ERROR] Fallo de I2C. Repetir.");
  if (!overflow && !i2cError && samplesRead < SAMPLES_PER_BATCH) Serial.println("[ERROR] Tiempo agotado. Repetir.");

  if (samplesRead == SAMPLES_PER_BATCH) {
    Serial.printf("[OK] %d muestras en %lu ms (esperado ~%d ms)\n", samplesRead, (unsigned long)elapsed_ms, SAMPLES_PER_BATCH);
    for (int i = 0; i < samplesRead; i++) {
      Serial.printf("%.4f,%.4f,%.4f,%d\n", ramBuffer[i].ax, ramBuffer[i].ay, ramBuffer[i].az, label);
    }
  }
  promptUser();
}

void setup() {
  Serial.begin(460800);
  while (!Serial) delay(10);
  setupMPU6050();
  Serial.println("\n===========================================================");
  Serial.println("   ESP32 DATA LOGGER (FIFO 1 kHz, 3 ejes) - SISTEMA LISTO ");
  Serial.println("===========================================================");
  promptUser();
}

void loop() {
  if (Serial.available() > 0) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') return;
    if (ch >= '0' && ch <= '4') captureAndSend(ch - '0');
  }
}
