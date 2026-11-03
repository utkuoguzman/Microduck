/*
 * ESP32 DevKit V1 (4MB Flash) Real-Time Servo & IMU Bridge for Microduck / XGO-Duck
 * =================================================================================
 * Replaces BOTH the Arduino Uno Q STM32 and Robotis OpenRB-150!
 *
 * Features:
 *   1. Drives 15x Feetech HD-1910M (1910) TTL Serial Bus Servos at 1,000,000 baud
 *      on UART2 (RX = GPIO16, TX = GPIO17, optional Half-Duplex TX_EN = GPIO4).
 *   2. Polls 6-axis IMU over I2C (SDA = GPIO21, SCL = GPIO22 @ 400 kHz) at 100 Hz.
 *      Auto-detects QMI8658A/C (0x6A/0x6B) or MPU6050 (0x68/0x69).
 *   3. Automatic Hardware-in-the-Loop (HIL) Desk Emulation Fallback:
 *      If you flash this onto your ESP32 DevKit V1 BEFORE buying the servos or IMU,
 *      any missing servo or IMU is automatically simulated on the ESP32 with first-order
 *      actuator dynamics + upright gravity so you can test the full 100 Hz Pi Zero W <-> ESP32
 *      control loop right on your desk!
 *   4. Communicates with Raspberry Pi Zero W (v1.1) (or PC) over USB/UART0 at 921600 baud
 *      using the exact XGO-Duck binary packet ABI (DQS1 240-byte StatePacket @ 100 Hz,
 *      DQC1 80-byte CommandPacket @ 50 Hz) framed with [0xA5, 0x5A, type, len, payload, xor_crc].
 */

#include <Arduino.h>
#include <Wire.h>
#include <math.h>
#include <string.h>

// ===================== PIN & BUS CONFIGURATION =====================
// Automatically adapts to ESP32 DevKit V1 / Lite, ESP32-S2 Mini, ESP32-C3 Pro Mini, and ESP32-H2 Mini!
static const uint32_t HOST_BAUD  = 921600;   // USB / UART0 to Raspberry Pi Zero W (v1.1) or PC
static const uint32_t SERVO_BAUD = 1000000;  // 1 Mbps for Feetech 1910 (HD-1910M) bus

#if defined(CONFIG_IDF_TARGET_ESP32S2)
  // Wemos / LOLIN S2 Mini (ESP32-S2FN4R2): Native USB CDC (Serial) + UART0 (Serial0 for Pi) + UART1 (Serial1 for Servos)
  #define ServoSerial Serial1
  #define HAS_DEDICATED_PI_UART 1
  #define PiSerial Serial0
  static const int PI_RX_PIN     = 44;       // UART0 RX (GPIO44) to Pi TXD (GPIO14)
  static const int PI_TX_PIN     = 43;       // UART0 TX (GPIO43) to Pi RXD (GPIO15)
  static const int SERVO_RX_PIN  = 18;       // UART1 RX (GPIO18)
  static const int SERVO_TX_PIN  = 17;       // UART1 TX (GPIO17)
  static const int SERVO_DIR_PIN = 4;        // Optional TX-enable pin (-1 if using 2-resistor circuit)
  static const int I2C_SDA_PIN   = 8;        // S2 Mini default SDA (GPIO8 or GPIO33)
  static const int I2C_SCL_PIN   = 9;        // S2 Mini default SCL (GPIO9 or GPIO35)
#elif defined(CONFIG_IDF_TARGET_ESP32C3) || defined(CONFIG_IDF_TARGET_ESP32H2)
  // ESP32-C3 Pro Mini / SuperMini or ESP32-H2 Mini: Native USB Serial/JTAG (Serial) + UART0 (Serial0 for Pi) + UART1 (Serial1 for Servos)
  #define ServoSerial Serial1
  #define HAS_DEDICATED_PI_UART 1
  #define PiSerial Serial0
  static const int PI_RX_PIN     = 20;       // UART0 RX (GPIO20) to Pi TXD (GPIO14)
  static const int PI_TX_PIN     = 21;       // UART0 TX (GPIO21) to Pi RXD (GPIO15)
  static const int SERVO_RX_PIN  = 5;        // UART1 RX (GPIO5)
  static const int SERVO_TX_PIN  = 4;        // UART1 TX (GPIO4)
  static const int SERVO_DIR_PIN = -1;       // Use 2-resistor (1k + 10k) half-duplex network
  static const int I2C_SDA_PIN   = 8;        // SDA (GPIO8)
  static const int I2C_SCL_PIN   = 9;        // SCL (GPIO9)
#else
  // Classic ESP32 DevKit V1 (30-Pin) & Wemos LOLIN32 Lite V1.0.0 (Rev1, 4MB)
  // Has 3 Hardware UARTs:
  //   - UART0 (Serial):  USB to PC (for MuJoCo HIL / Web UI / Flashing)
  //   - UART1 (Serial1): Dedicated 921600-baud link to Raspberry Pi Zero W (RX1=GPIO25, TX1=GPIO26)
  //   - UART2 (Serial2): 1 Mbps Feetech HD-1910-C001 Servo Bus (RX2=GPIO16, TX2=GPIO17)
  #define ServoSerial Serial2
  #define HAS_DEDICATED_PI_UART 1
  #define PiSerial Serial1
  static const int PI_RX_PIN     = 25;       // Connect to Pi Zero W Pin 8  (GPIO14 / UART TXD)
  static const int PI_TX_PIN     = 26;       // Connect to Pi Zero W Pin 10 (GPIO15 / UART RXD)
  static const int SERVO_RX_PIN  = 16;       // UART2 RX (GPIO16)
  static const int SERVO_TX_PIN  = 17;       // UART2 TX (GPIO17)
  static const int SERVO_DIR_PIN = 4;        // Optional TX-enable pin for 74LVC1G126 buffer
  static const int I2C_SDA_PIN   = 21;       // SDA (GPIO21)
  static const int I2C_SCL_PIN   = 22;       // SCL (GPIO22)
#endif

// If true, missing servos/IMU are emulated in real time so you can test with a bare ESP32!
#define AUTO_EMULATE_MISSING_HARDWARE 1

// ===================== SERVO & PROTOCOL CONSTANTS ==================
static const uint8_t NUM_MOTORS = 15;
static const uint32_t TICK_US = 10000;       // 100 Hz loop
static const uint32_t SERVO_REPLY_US = 3500;
static const float M_N = 4095.0f;
static const float M_A = 360.0f;
static const float SERVO_SPEED_UNIT = 50.0f;
static const uint16_t POS_MIN = 0;
static const uint16_t POS_MAX = 4095;
static const uint16_t SERVO_CENTER_POS = 2047;

static const uint8_t ADDR_GOAL_POS = 0x2A;
static const uint8_t ADDR_PRESENT  = 0x38;
static const uint8_t PRESENT_LEN   = 6;
static const uint8_t ADDR_TORQUE   = 40;
static const uint8_t ADDR_TEMP_KP  = 50;
static const uint8_t ADDR_TEMP_KD  = 51;
static const uint8_t ADDR_PERM_KP_KD = 0x15;
static const uint8_t ADDR_UNLOCK   = 0x37;
static const uint8_t ADDR_ID       = 0x05;
static const uint8_t ADDR_UNLOAD_CONDITION   = 0x13;
static const uint8_t ADDR_PROTECTION_CURRENT = 0x1C;
static const uint8_t PROTECTION_DISABLED = 0x04;
static const uint8_t PROTECTION_ENABLED  = 0x0D;
static const uint16_t CAL_HOLD_POS = 2048;

static const uint8_t MOTOR_IDS[NUM_MOTORS] = {
    10, 11, 12, 13, 14,   // Left leg
    20, 21, 22, 23, 24,   // Right leg
    30, 31, 32, 33, 34    // Neck, head (30..33), and mouth (34)
};

static const float MOTOR_SIGN[NUM_MOTORS] = {
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1
};

enum CommandMode : uint8_t { DISARM = 0, ARM = 1, POLICY = 2, HOLD = 3 };

enum ServoOp : uint8_t {
  SERVO_ENTER = 0,
  SERVO_EXIT = 1,
  SERVO_UNLOCK = 2,
  SERVO_SET_ID = 3,
  SERVO_SET_PERM_KP_KD = 5,
  SERVO_READ = 6,
  SERVO_PING = 7,
  SERVO_RESET_MID = 8,
  SERVO_SET_PROTECTION = 9,
  SERVO_SET_PROTECTION_CURRENT = 10
};

// Binary ABI shared with wire.py (240 bytes StatePacket, 80 bytes CommandPacket, 12B ServoPacket, 14B ServoReplyPacket)
struct __attribute__((packed)) CommandPacket {
  char magic[4];          // "DQC1"
  uint32_t seq, sourceUs, ttlUs;
  uint8_t mode, reserved;
  uint16_t reserved2;
  float target[15];
};

struct __attribute__((packed)) StatePacket {
  char magic[4];          // "DQS1"
  uint8_t version, flags;
  uint16_t servoMask;
  uint32_t seq, sampleUs, imuUs, commandSeq, commandAgeUs;
  uint32_t periodUs, workUs, overruns, reportDrops, invalidCommands;
  int16_t acc[3], gyro[3];
  float angle[15], velocity[15], target[15];
};

struct __attribute__((packed)) ServoPacket {
  char magic[4];          // "DQV2"
  uint8_t op, targetId, newId, kp, kd;
  union { uint8_t reserved; uint8_t protectionFlags; };
  union { uint16_t reserved2; uint16_t protectionCurrent; };
};

struct __attribute__((packed)) ServoReplyPacket {
  char magic[4];          // "DQS3"
  uint8_t op, targetId, ok;
  union { uint8_t reserved; uint8_t protectionFlags; };
  union { uint16_t rawPos; uint16_t protectionCurrent; };
  uint16_t centerPos;
  uint8_t kp, kd;
};

static_assert(sizeof(CommandPacket) == 80, "CommandPacket must be 80 bytes");
static_assert(sizeof(StatePacket) == 240, "StatePacket must be 240 bytes");
static_assert(sizeof(ServoPacket) == 12, "ServoPacket must be 12 bytes");
static_assert(sizeof(ServoReplyPacket) == 14, "ServoReplyPacket must be 14 bytes");

struct MotorState {
  uint8_t id;
  float sign;
  uint16_t fbPos;
  float desAngleDeg;
  uint16_t desPos;
  float fbAngleDeg;
  float fbVelDeg;
  bool physicalPresent;
};

static MotorState motors[NUM_MOTORS];
static uint16_t servoRxMask = 0;
static bool torqueEnabled = false;
static uint32_t stateSeq = 0;
static uint32_t acceptedCmdSeq = 0;
static uint32_t lastCmdUs = 0;
static uint32_t invalidCmds = 0;
static uint32_t loopOverruns = 0;

// ===================== IMU DRIVER (QMI8658 / MPU6050 / EMULATED) ===
enum ImuType { IMU_NONE = 0, IMU_QMI8658 = 1, IMU_MPU6050 = 2 };
static ImuType activeImu = IMU_NONE;
static uint8_t imuI2cAddr = 0;
static int16_t imuAccRaw[3] = {0, 0, -16384}; // 1g = 16384 LSB
static int16_t imuGyroRaw[3] = {0, 0, 0};     // 1 dps = 32 LSB
static uint32_t imuSampleUs = 0;
static bool imuOk = false;

static bool i2cWriteReg(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool i2cReadRegs(uint8_t addr, uint8_t reg, uint8_t *buf, uint8_t len) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)addr, (int)len) != len) return false;
  for (uint8_t i = 0; i < len; i++) buf[i] = (uint8_t)Wire.read();
  return true;
}

static void initImuSensor() {
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, 400000);
  delay(20);

  // 1. Probe QMI8658 (0x6A, 0x6B)
  const uint8_t qmiAddrs[2] = {0x6A, 0x6B};
  for (int i = 0; i < 2; i++) {
    uint8_t who = 0;
    if (i2cReadRegs(qmiAddrs[i], 0x00, &who, 1) && who == 0x05) {
      imuI2cAddr = qmiAddrs[i];
      activeImu = IMU_QMI8658;
      i2cWriteReg(imuI2cAddr, 0x60, 0xB0);
      delay(20);
      i2cWriteReg(imuI2cAddr, 0x08, 0x00);
      i2cWriteReg(imuI2cAddr, 0x02, 0x40);
      i2cWriteReg(imuI2cAddr, 0x03, 0x04); // +/- 2g (16384 LSB/g)
      i2cWriteReg(imuI2cAddr, 0x04, 0x64); // +/- 1024 dps (32 LSB/dps)
      i2cWriteReg(imuI2cAddr, 0x06, 0x00);
      i2cWriteReg(imuI2cAddr, 0x08, 0x03);
      imuOk = true;
      return;
    }
  }

  // 2. Probe MPU6050 / MPU6500 (0x68, 0x69)
  const uint8_t mpuAddrs[2] = {0x68, 0x69};
  for (int i = 0; i < 2; i++) {
    uint8_t who = 0;
    if (i2cReadRegs(mpuAddrs[i], 0x75, &who, 1) && (who == 0x68 || who == 0x70 || who == 0x71)) {
      imuI2cAddr = mpuAddrs[i];
      activeImu = IMU_MPU6050;
      i2cWriteReg(imuI2cAddr, 0x6B, 0x01); // Wake up, PLL with X axis gyro
      i2cWriteReg(imuI2cAddr, 0x1C, 0x00); // +/- 2g (16384 LSB/g)
      i2cWriteReg(imuI2cAddr, 0x1B, 0x10); // +/- 1000 dps (32.8 LSB/dps)
      imuOk = true;
      return;
    }
  }

  activeImu = IMU_NONE;
}

static void pollImuSensor() {
  imuSampleUs = micros();
  if (activeImu == IMU_QMI8658) {
    uint8_t d[12];
    if (i2cReadRegs(imuI2cAddr, 0x35, d, 12)) {
      int16_t ax = (int16_t)((d[1] << 8) | d[0]);
      int16_t ay = (int16_t)((d[3] << 8) | d[2]);
      int16_t az = (int16_t)((d[5] << 8) | d[4]);
      int16_t gx = (int16_t)((d[7] << 8) | d[6]);
      int16_t gy = (int16_t)((d[9] << 8) | d[8]);
      int16_t gz = (int16_t)((d[11] << 8) | d[10]);
      imuAccRaw[0] = az;  imuAccRaw[1] = ax;  imuAccRaw[2] = -ay;
      imuGyroRaw[0] = -gz; imuGyroRaw[1] = -gx; imuGyroRaw[2] = gy;
      imuOk = true;
      return;
    }
  } else if (activeImu == IMU_MPU6050) {
    uint8_t d[14];
    if (i2cReadRegs(imuI2cAddr, 0x3B, d, 14)) {
      int16_t ax = (int16_t)((d[0] << 8) | d[1]);
      int16_t ay = (int16_t)((d[2] << 8) | d[3]);
      int16_t az = (int16_t)((d[4] << 8) | d[5]);
      int16_t gx = (int16_t)((d[8] << 8) | d[9]);
      int16_t gy = (int16_t)((d[10] << 8) | d[11]);
      int16_t gz = (int16_t)((d[12] << 8) | d[13]);
      imuAccRaw[0] = ax;  imuAccRaw[1] = ay;  imuAccRaw[2] = az;
      imuGyroRaw[0] = gx; imuGyroRaw[1] = gy; imuGyroRaw[2] = gz;
      imuOk = true;
      return;
    }
  }

#if AUTO_EMULATE_MISSING_HARDWARE
  // Emulate level upright duck IMU in policy body axes (acc_n = [0, 0, -1g] -> DQS1 raw [0, 0, -16384])
  imuAccRaw[0] = 0;
  imuAccRaw[1] = 0;
  imuAccRaw[2] = -16384;
  imuGyroRaw[0] = 0;
  imuGyroRaw[1] = 0;
  imuGyroRaw[2] = 0;
  imuOk = true;
#else
  imuOk = false;
#endif
}

// ===================== FEETECH 1910 (SCS/HLS) BUS ==================
static inline void busTxEnable(bool tx) {
  if (SERVO_DIR_PIN >= 0) {
    digitalWrite(SERVO_DIR_PIN, tx ? HIGH : LOW);
  }
}

static void scsWriteBuf(uint8_t id, uint8_t fun, const uint8_t *nDat, uint8_t nLen) {
  uint8_t buf[8 + NUM_MOTORS * 7];
  uint8_t check = 0;
  buf[0] = 0xFF;
  buf[1] = 0xFF;
  buf[2] = id;
  buf[3] = nLen + 2;
  buf[4] = fun;
  for (uint8_t i = 0; i < nLen; i++) {
    buf[5 + i] = nDat[i];
    check = (uint8_t)(check + nDat[i]);
  }
  check = (uint8_t)(check + buf[2] + buf[3] + buf[4]);
  buf[nLen + 5] = (uint8_t)~check;

  busTxEnable(true);
  ServoSerial.write(buf, nLen + 6);
  ServoSerial.flush();
  busTxEnable(false);
  while (ServoSerial.available()) (void)ServoSerial.read();
}

static void scsWriteByte(uint8_t id, uint8_t addr, uint8_t value) {
  const uint8_t data[2] = {addr, value};
  scsWriteBuf(id, 0x03, data, 2);
}

static void scsWriteWord(uint8_t id, uint8_t addr, uint16_t value) {
  const uint8_t data[3] = {addr, (uint8_t)(value & 0xFF), (uint8_t)(value >> 8)};
  scsWriteBuf(id, 0x03, data, 3);
}

// ===================== BUILT-IN FEETECH 1910 BUS PROGRAMMER ========
// Replaces the external FE-URT-1 / FE-URT-2 USB programmer completely!
// Supports: Ping, Read Pos/Kp/Kd, Unlock EEPROM, Assign Servo ID (1 -> 10..34),
//           Calibrate Center Position (2047 via 0x0B), Set Permanent Kp/Kd, and Overload Protection.
static void servoUnlock(uint8_t id) {
  for (uint8_t i = 0; i < 3; i++) {
    scsWriteByte(id, ADDR_UNLOCK, 0x00);
    delay(10);
  }
}

static void servoSetId(uint8_t id, uint8_t newId) {
  servoUnlock(id);
  for (uint8_t i = 0; i < 3; i++) {
    scsWriteByte(id, ADDR_ID, newId);
    if (i < 2) delay(10);
  }
}

static void servoMid(uint8_t id) {
  scsWriteByte(id, ADDR_TORQUE, 0);
  delay(5);
  for (uint8_t i = 0; i < 3; i++) {
    scsWriteBuf(id, 0x0B, nullptr, 0);
    delay(10);
  }
}

static void servoSetPermKpKd(uint8_t id, uint8_t kp, uint8_t kd) {
  const uint8_t data[3] = {ADDR_PERM_KP_KD, kp, kd};
  servoUnlock(id);
  for (uint8_t i = 0; i < 3; i++) {
    scsWriteBuf(id, 0x03, data, 3);
    if (i < 2) delay(10);
  }
}

static void servoSetProtection(uint8_t flags) {
  servoUnlock(0xFE);
  for (uint8_t i = 0; i < 3; i++) {
    scsWriteByte(0xFE, ADDR_UNLOAD_CONDITION, flags);
    if (i < 2) delay(10);
  }
}

static void servoSetProtectionCurrent(uint16_t current) {
  servoUnlock(0xFE);
  for (uint8_t i = 0; i < 3; i++) {
    scsWriteWord(0xFE, ADDR_PROTECTION_CURRENT, current);
    if (i < 2) delay(10);
  }
}

static bool pingServo(uint8_t id) {
  while (ServoSerial.available()) (void)ServoSerial.read();
  scsWriteBuf(id, 0x01, nullptr, 0);
  uint8_t rx[8];
  uint8_t state = 0, len = 0, idx = 0;
  const uint32_t t0 = micros();
  while ((uint32_t)(micros() - t0) < 6000) {
    while (ServoSerial.available()) {
      uint8_t b = (uint8_t)ServoSerial.read();
      if (state == 0) { if (b == 0xFF) state = 1; }
      else if (state == 1) { state = (b == 0xFF) ? 2 : 0; }
      else if (state == 2) { rx[2] = b; state = 3; }
      else if (state == 3) {
        rx[3] = b;
        if (b == 2 && b <= 4) { len = b; idx = 0; state = 4; }
        else state = 0;
      } else if (state == 4) {
        rx[4 + idx++] = b;
        if (idx == len) {
          uint8_t sum = rx[2] + rx[3] + rx[4];
          if ((uint8_t)~sum == rx[5] && rx[2] == id) return true;
          state = 0;
        }
      }
    }
  }
  return false;
}

static bool readServoRegBytes(uint8_t id, uint8_t regAddr, uint8_t count, uint8_t *out) {
  while (ServoSerial.available()) (void)ServoSerial.read();
  const uint8_t req[2] = {regAddr, count};
  scsWriteBuf(id, 0x02, req, 2);
  uint8_t rx[24];
  uint8_t state = 0, len = 0, idx = 0;
  const uint32_t t0 = micros();
  while ((uint32_t)(micros() - t0) < 6000) {
    while (ServoSerial.available()) {
      uint8_t b = (uint8_t)ServoSerial.read();
      if (state == 0) { if (b == 0xFF) state = 1; }
      else if (state == 1) { state = (b == 0xFF) ? 2 : 0; }
      else if (state == 2) { rx[2] = b; state = 3; }
      else if (state == 3) {
        rx[3] = b;
        if (b == (uint8_t)(count + 2) && b <= 16) { len = b; idx = 0; state = 4; }
        else state = 0;
      } else if (state == 4) {
        rx[4 + idx++] = b;
        if (idx == len) {
          uint8_t sum = rx[2] + rx[3];
          for (uint8_t k = 0; k < len - 1; k++) sum += rx[4 + k];
          if ((uint8_t)~sum == rx[3 + len] && rx[2] == id && rx[4] == 0) {
            memcpy(out, &rx[5], count);
            return true;
          }
          state = 0;
        }
      }
    }
  }
  return false;
}

static uint16_t satPos(int32_t v) {
  if (v < (int32_t)POS_MIN) return POS_MIN;
  if (v > (int32_t)POS_MAX) return POS_MAX;
  return (uint16_t)v;
}

static void syncWriteGoals() {
  uint8_t sync[2 + NUM_MOTORS * 7];
  sync[0] = ADDR_GOAL_POS;
  sync[1] = 6;
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    int32_t pos = (int32_t)SERVO_CENTER_POS +
                  (int32_t)(motors[i].desAngleDeg / M_A * M_N * motors[i].sign);
    motors[i].desPos = satPos(pos);
    uint8_t *p = sync + 2 + i * 7;
    p[0] = motors[i].id;
    p[1] = (uint8_t)(motors[i].desPos & 0xFF);
    p[2] = (uint8_t)(motors[i].desPos >> 8);
    p[3] = 0; p[4] = 0; p[5] = 0; p[6] = 0;
  }
  scsWriteBuf(0xFE, 0x83, sync, (uint8_t)(2 + NUM_MOTORS * 7));
}

static void requestAllServoState() {
  uint8_t buf[8 + NUM_MOTORS];
  uint8_t check = 0;
  buf[0] = 0xFF;
  buf[1] = 0xFF;
  buf[2] = 0xFE;
  buf[3] = NUM_MOTORS + 4;
  buf[4] = 0x82;
  buf[5] = ADDR_PRESENT;
  buf[6] = PRESENT_LEN;
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    buf[7 + i] = motors[i].id;
    check = (uint8_t)(check + motors[i].id);
  }
  check = (uint8_t)(check + buf[2] + buf[3] + buf[4] + buf[5] + buf[6]);
  buf[7 + NUM_MOTORS] = (uint8_t)~check;

  while (ServoSerial.available()) (void)ServoSerial.read();
  busTxEnable(true);
  ServoSerial.write(buf, (uint8_t)(8 + NUM_MOTORS));
  ServoSerial.flush();
  busTxEnable(false);
}

static int motorIndexById(uint8_t id) {
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    if (motors[i].id == id) return i;
  }
  return -1;
}

static void pollAndEmulateServos(float dt) {
  servoRxMask = 0;
  requestAllServoState();

  uint8_t rxBuf[16];
  uint8_t state = 0, len = 0, idx = 0;
  const uint32_t t0 = micros();
  while ((uint32_t)(micros() - t0) < SERVO_REPLY_US) {
    while (ServoSerial.available()) {
      uint8_t b = (uint8_t)ServoSerial.read();
      if (state == 0) { if (b == 0xFF) state = 1; }
      else if (state == 1) { state = (b == 0xFF) ? 2 : 0; }
      else if (state == 2) { rxBuf[2] = b; state = 3; }
      else if (state == 3) {
        rxBuf[3] = b;
        if (b == 8) { len = b; idx = 0; state = 4; }
        else state = 0;
      } else if (state == 4) {
        rxBuf[4 + idx++] = b;
        if (idx == len) {
          state = 0;
          uint8_t sum = rxBuf[2] + rxBuf[3];
          for (uint8_t k = 0; k < len - 1; k++) sum += rxBuf[4 + k];
          if ((uint8_t)~sum == rxBuf[3 + len] && rxBuf[4] == 0) {
            int m = motorIndexById(rxBuf[2]);
            if (m >= 0) {
              uint16_t rawPos = (uint16_t)rxBuf[5] | ((uint16_t)rxBuf[6] << 8);
              uint16_t rawSpd = (uint16_t)rxBuf[7] | ((uint16_t)rxBuf[8] << 8);
              int32_t spd = (int32_t)(rawSpd & 0x7FFFu);
              if (rawSpd & 0x8000u) spd = -spd;
              motors[m].fbPos = rawPos;
              motors[m].fbAngleDeg = (float)((int32_t)rawPos - SERVO_CENTER_POS) * M_A / M_N * motors[m].sign;
              motors[m].fbVelDeg = (float)spd * SERVO_SPEED_UNIT * (360.0f / 4096.0f) * motors[m].sign;
              motors[m].physicalPresent = true;
              servoRxMask |= (1u << m);
            }
          }
        }
      }
    }
  }

#if AUTO_EMULATE_MISSING_HARDWARE
  // Smoothly simulate any servo not physically connected on the UART2 bus
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    if (!(servoRxMask & (1u << i))) {
      if (torqueEnabled) {
        float err = motors[i].desAngleDeg - motors[i].fbAngleDeg;
        float stepVel = err * 18.0f; // First-order servo response (~55 ms time constant)
        motors[i].fbVelDeg = stepVel;
        motors[i].fbAngleDeg += stepVel * dt;
      } else {
        motors[i].fbVelDeg = 0.0f;
      }
      int32_t pos = (int32_t)SERVO_CENTER_POS +
                    (int32_t)(motors[i].fbAngleDeg / M_A * M_N * motors[i].sign);
      motors[i].fbPos = satPos(pos);
      servoRxMask |= (1u << i);
    }
  }
#endif
}

// ===================== DUAL-PORT HIL FRAMING (PC USB + PI ZERO W UART) ========
// Frame format: [0xA5, 0x5A, pktType, payloadLen, ...payload..., xorChecksum]
// pktType: 0x01 = StatePacket (240B, DQS1), 0x02 = CommandPacket (80B, DQC1),
//          0x05 = ServoPacket (12B, DQV2),  0x07 = ServoReplyPacket (14B, DQS3)
//
// Supports 3-Node Closed-Loop HIL:
//   [PC MuJoCo Physics] <--USB (Serial)--> [ESP32 Bridge] <--UART1 (GPIO25/26)--> [Raspberry Pi Zero W]
struct HostParser {
  uint8_t buf[256];
  uint8_t state = 0;
  uint8_t type = 0;
  uint8_t len = 0;
  uint8_t idx = 0;
};

static HostParser pcParser;
static HostParser piParser;
static uint32_t lastPcSimStateUs = 0;

static void sendFramedToStream(Stream &port, uint8_t pktType, const void *payload, uint8_t len) {
  uint8_t header[4] = {0xA5, 0x5A, pktType, len};
  uint8_t crc = pktType ^ len;
  const uint8_t *p = (const uint8_t *)payload;
  for (uint8_t i = 0; i < len; i++) crc ^= p[i];
  port.write(header, 4);
  port.write(p, len);
  port.write(crc);
}

static void sendFramedPacket(uint8_t pktType, const void *payload, uint8_t len) {
  sendFramedToStream(Serial, pktType, payload, len);
#if HAS_DEDICATED_PI_UART
  sendFramedToStream(PiSerial, pktType, payload, len);
#endif
}

static void handleServoPacket(const ServoPacket &pkt) {
  ServoReplyPacket reply = {};
  memcpy(reply.magic, "DQS3", 4);
  reply.op = pkt.op;
  reply.targetId = pkt.targetId;
  reply.centerPos = SERVO_CENTER_POS;
  reply.ok = 1;

  switch (pkt.op) {
    case SERVO_ENTER:
    case SERVO_EXIT:
      torqueEnabled = false;
      scsWriteByte(0xFE, ADDR_TORQUE, 0);
      break;
    case SERVO_UNLOCK:
      servoUnlock(pkt.targetId);
      break;
    case SERVO_SET_ID:
      torqueEnabled = false;
      scsWriteByte(0xFE, ADDR_TORQUE, 0);
      servoSetId(pkt.targetId, pkt.newId);
      break;
    case SERVO_RESET_MID:
      servoMid(pkt.targetId);
      break;
    case SERVO_SET_PERM_KP_KD:
      servoSetPermKpKd(pkt.targetId, pkt.kp, pkt.kd);
      break;
    case SERVO_SET_PROTECTION:
      servoSetProtection(pkt.protectionFlags);
      reply.protectionFlags = pkt.protectionFlags;
      break;
    case SERVO_SET_PROTECTION_CURRENT:
      servoSetProtectionCurrent(pkt.protectionCurrent);
      reply.protectionCurrent = pkt.protectionCurrent;
      break;
    case SERVO_PING:
      reply.ok = pingServo(pkt.targetId) ? 1 : 0;
      break;
    case SERVO_READ: {
      uint8_t posBytes[2] = {0, 0};
      uint8_t gainBytes[2] = {0, 0};
      bool okPos  = readServoRegBytes(pkt.targetId, ADDR_PRESENT, 2, posBytes);
      bool okGain = readServoRegBytes(pkt.targetId, ADDR_PERM_KP_KD, 2, gainBytes);
      reply.ok = (okPos || okGain) ? 1 : 0;
      reply.rawPos = (uint16_t)posBytes[0] | ((uint16_t)posBytes[1] << 8);
      reply.kp = gainBytes[0];
      reply.kd = gainBytes[1];
      break;
    }
    default:
      reply.ok = 0;
      break;
  }
  sendFramedPacket(0x07, &reply, sizeof(reply));
}

static void applyCommandPacket(const CommandPacket &cmd, bool fromPi) {
  if (memcmp(cmd.magic, "DQC1", 4) == 0 && cmd.mode <= HOLD) {
    acceptedCmdSeq = cmd.seq;
    lastCmdUs = micros();
    if (cmd.mode == DISARM) torqueEnabled = false;
    else if (cmd.mode == ARM || cmd.mode == POLICY || cmd.mode == HOLD) torqueEnabled = true;
    for (uint8_t i = 0; i < NUM_MOTORS; i++) {
      if (isfinite(cmd.target[i])) motors[i].desAngleDeg = cmd.target[i];
    }
    // In 3-node HIL mode: when Pi Zero W sends a DQC1 command over UART1,
    // forward it over USB (Serial) so PC MuJoCo steps the 3D duck in real time!
    if (fromPi) {
      sendFramedToStream(Serial, 0x02, &cmd, sizeof(cmd));
    }
  } else {
    invalidCmds++;
  }
}

static void pollStream(Stream &port, HostParser &p, bool isPiPort) {
  while (port.available()) {
    uint8_t b = (uint8_t)port.read();
    switch (p.state) {
      case 0: if (b == 0xA5) p.state = 1; break;
      case 1: p.state = (b == 0x5A) ? 2 : 0; break;
      case 2: p.type = b; p.state = 3; break;
      case 3:
        p.len = b;
        if ((p.len == sizeof(CommandPacket) && (p.type == 0x02 || p.type == 0x10)) ||
            (p.len == sizeof(ServoPacket) && p.type == 0x05) ||
            (!isPiPort && p.len == sizeof(StatePacket) && p.type == 0x01)) {
          p.idx = 0;
          p.state = 4;
        } else {
          p.state = 0;
        }
        break;
      case 4:
        p.buf[p.idx++] = b;
        if (p.idx == p.len) p.state = 5;
        break;
      case 5: {
        p.state = 0;
        uint8_t crc = p.type ^ p.len;
        for (uint8_t i = 0; i < p.len; i++) crc ^= p.buf[i];
        if (crc == b) {
          if ((p.type == 0x02 || p.type == 0x10) && p.len == sizeof(CommandPacket)) {
            CommandPacket cmd;
            memcpy(&cmd, p.buf, sizeof(cmd));
            applyCommandPacket(cmd, isPiPort);
          } else if (p.type == 0x05 && p.len == sizeof(ServoPacket)) {
            ServoPacket spkt;
            memcpy(&spkt, p.buf, sizeof(spkt));
            if (memcmp(spkt.magic, "DQV2", 4) == 0) {
              handleServoPacket(spkt);
            } else {
              invalidCmds++;
            }
          } else if (!isPiPort && p.type == 0x01 && p.len == sizeof(StatePacket)) {
            // PC MuJoCo injected a 240-byte DQS1 physics state over USB:
            // Forward it immediately over UART1 (PiSerial) to the Raspberry Pi Zero W!
            StatePacket simSt;
            memcpy(&simSt, p.buf, sizeof(simSt));
            if (memcmp(simSt.magic, "DQS1", 4) == 0) {
              lastPcSimStateUs = micros();
              memcpy(imuAccRaw, simSt.acc, sizeof(imuAccRaw));
              memcpy(imuGyroRaw, simSt.gyro, sizeof(imuGyroRaw));
              for (uint8_t i = 0; i < NUM_MOTORS; i++) {
                motors[i].fbAngleDeg = simSt.angle[i];
                motors[i].fbVelDeg = simSt.velocity[i];
              }
#if HAS_DEDICATED_PI_UART
              sendFramedToStream(PiSerial, 0x01, &simSt, sizeof(simSt));
#endif
            }
          }
        } else {
          invalidCmds++;
        }
        break;
      }
    }
  }
}

static void pollHostSerial() {
  pollStream(Serial, pcParser, false);
#if HAS_DEDICATED_PI_UART
  pollStream(PiSerial, piParser, true);
#endif
}

void setup() {
  if (SERVO_DIR_PIN >= 0) {
    pinMode(SERVO_DIR_PIN, OUTPUT);
    digitalWrite(SERVO_DIR_PIN, LOW);
  }
  Serial.begin(HOST_BAUD);
#if HAS_DEDICATED_PI_UART
  PiSerial.begin(HOST_BAUD, SERIAL_8N1, PI_RX_PIN, PI_TX_PIN);
#endif
  ServoSerial.begin(SERVO_BAUD, SERIAL_8N1, SERVO_RX_PIN, SERVO_TX_PIN);

  // Default standing angles (degrees)
  const float defaultDeg[NUM_MOTORS] = {
      0.0f, -5.0f, -22.0f, 0.0f, 22.0f,
      0.0f,  5.0f,  22.0f, 0.0f, -22.0f,
      20.0f, 20.0f,  0.0f, 0.0f,  0.0f
  };
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    motors[i].id = MOTOR_IDS[i];
    motors[i].sign = MOTOR_SIGN[i];
    motors[i].fbPos = SERVO_CENTER_POS;
    motors[i].desPos = SERVO_CENTER_POS;
    motors[i].desAngleDeg = defaultDeg[i];
    motors[i].fbAngleDeg = defaultDeg[i];
    motors[i].fbVelDeg = 0.0f;
    motors[i].physicalPresent = false;
  }

  initImuSensor();
}

void loop() {
  static uint32_t lastTickUs = micros();
  pollHostSerial();

  uint32_t nowUs = micros();
  if ((uint32_t)(nowUs - lastTickUs) < TICK_US) {
    return;
  }
  uint32_t periodUs = nowUs - lastTickUs;
  if (periodUs > TICK_US * 2) loopOverruns++;
  lastTickUs = nowUs;

  const bool pcMuJoCoActive = (lastPcSimStateUs != 0) &&
                              ((uint32_t)(nowUs - lastPcSimStateUs) < 250000u);

  if (pcMuJoCoActive) {
    // 3-Node HIL Mode: PC MuJoCo provides physics DQS1 state over USB (already forwarded to PiSerial).
    // Still drive any physical servos connected to ServoSerial!
    if (torqueEnabled) {
      syncWriteGoals();
    } else {
      scsWriteByte(0xFE, ADDR_TORQUE, 0);
    }
    return;
  }

  // Standalone / Real Robot Mode:
  // 1. Update IMU
  pollImuSensor();

  // 2. Write servo goals & read feedback (or emulate missing servos)
  if (torqueEnabled) {
    syncWriteGoals();
  } else {
    scsWriteByte(0xFE, ADDR_TORQUE, 0);
  }
  pollAndEmulateServos(periodUs * 1e-6f);

  // 3. Build and transmit 240-byte StatePacket (DQS1) to Pi Zero W / PC
  StatePacket s = {};
  memcpy(s.magic, "DQS1", 4);
  s.version = 1;
  s.flags = (imuOk ? 1 : 0) | (torqueEnabled ? 2 : 0);
  s.servoMask = servoRxMask;
  s.seq = ++stateSeq;
  s.sampleUs = micros();
  s.imuUs = imuSampleUs;
  s.commandSeq = acceptedCmdSeq;
  s.commandAgeUs = lastCmdUs ? (s.sampleUs - lastCmdUs) : UINT32_MAX;
  s.periodUs = periodUs;
  s.workUs = micros() - nowUs;
  s.overruns = loopOverruns;
  s.reportDrops = 0;
  s.invalidCommands = invalidCmds;
  memcpy(s.acc, imuAccRaw, sizeof(s.acc));
  memcpy(s.gyro, imuGyroRaw, sizeof(s.gyro));
  for (uint8_t i = 0; i < NUM_MOTORS; i++) {
    s.angle[i] = motors[i].fbAngleDeg;
    s.velocity[i] = motors[i].fbVelDeg;
    s.target[i] = motors[i].desAngleDeg;
  }

  sendFramedPacket(0x01, &s, sizeof(s));
}
