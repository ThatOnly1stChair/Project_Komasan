/*
  RobStride RS01 angle reader, ESP32-WROOM-32 + SN65HVD230.
  GPIO21 -> D/TXD, GPIO22 <- R/RXD, 3V3 -> VCC, common GND.
  CAN_H -> CAN_H, CAN_L -> CAN_L; proper termination at both bus ends.
  Motor power is separate. Private protocol, extended CAN, 1 Mbps.
  All listed motors may be connected together, each with a unique ID.
  Type angle followed by newline to query all MOTOR_IDS once.
  Reports motor mechPos in radians and degrees without wrapping or zeroing.
  move ID DEGREES commands an absolute output angle; stop ID disables torque.
  Start with an unloaded, secured motor. Motor holds position after move.
  Motion settings are written to RAM; no zero, ID, or flash writes.
  Based on the supplied working ID manager and official protocol examples:
  https://github.com/RobStride/SampleProgram/tree/5f598686b05fcc527ee0fc0ea954f0afd652b234/RS
  https://github.com/RobStride/Python_Sample/blob/main/robstride_dynamics/protocol.py
*/
#include <Arduino.h>
#include <driver/twai.h>
#include <esp_err.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>

// Edit this list to match your assigned motor IDs.
static constexpr uint8_t MOTOR_IDS[] = {1, 2, 3};
static constexpr uint16_t POSITION_INDEX = 0x7019;
// Conservative bench-test defaults, not limits sized for a loaded robot.
static constexpr float MOVE_SPEED_RAD_S = 0.5f;
static constexpr float MOVE_ACCEL_RAD_S2 = 1.0f;
static constexpr float MOVE_CURRENT_A = 1.0f;
static constexpr float MAX_ABS_TARGET_DEG = 720.0f; // +/-4 pi rad
static constexpr float MAX_MOVE_DELTA_DEG = 180.0f; // maximum step per command

static constexpr gpio_num_t CAN_TX = GPIO_NUM_21;
static constexpr gpio_num_t CAN_RX = GPIO_NUM_22;
static constexpr uint8_t HOST_ID = 0xFD;
static bool canReady = false;

// All helpers are defined before use, including explicit prototypes for the
// two functions with default arguments, to work with Arduino preprocessing.
static bool sendFrame(uint32_t id, const uint8_t *data = nullptr);


static uint32_t commandID(uint8_t type, uint8_t motor) {
  return (uint32_t(type) << 24) | (uint32_t(HOST_ID) << 8) | motor;
}

static void status() {
  if (!canReady) { Serial.println("CAN initialization failed. Reset ESP32."); return; }
  twai_status_info_t s = {};
  esp_err_t e = twai_get_status_info(&s);
  if (e != ESP_OK) { Serial.println(esp_err_to_name(e)); return; }
  const char *state = s.state == TWAI_STATE_RUNNING ? "RUNNING" :
                      s.state == TWAI_STATE_BUS_OFF ? "BUS-OFF" :
                      s.state == TWAI_STATE_RECOVERING ? "RECOVERING" : "STOPPED";
  Serial.printf("CAN %s; TX errors=%lu RX errors=%lu failed TX=%lu bus errors=%lu\n",
    state, (unsigned long)s.tx_error_counter, (unsigned long)s.rx_error_counter,
    (unsigned long)s.tx_failed_count, (unsigned long)s.bus_error_count);
}

static bool sendFrame(uint32_t id, const uint8_t *data) {
  if (!canReady) { Serial.println("CAN unavailable."); return false; }
  twai_status_info_t s = {};
  if (twai_get_status_info(&s) != ESP_OK || s.state != TWAI_STATE_RUNNING) {
    Serial.println("CAN is not running. Check wiring/power, then reset ESP32.");
    return false;
  }
  uint32_t alerts = 0;
  twai_read_alerts(&alerts, 0);  // Discard alerts from a previous transaction.
  twai_message_t tx = {};
  tx.identifier = id;
  tx.extd = 1;
  tx.ss = 1;                   // One attempt: no endless retries without a motor.
  tx.data_length_code = 8;
  if (data) memcpy(tx.data, data, 8);
  esp_err_t e = twai_transmit(&tx, pdMS_TO_TICKS(30));
  if (e != ESP_OK) {
    Serial.printf("TX queue error: %s\n", esp_err_to_name(e));
    return false;
  }
  // twai_transmit() only queues a frame. Wait for actual CAN transmission.
  const uint32_t start = millis();
  while (millis() - start < 100) {
    alerts = 0;
    twai_read_alerts(&alerts, pdMS_TO_TICKS(10));
    if (alerts & (TWAI_ALERT_TX_FAILED | TWAI_ALERT_BUS_OFF)) {
      Serial.println("CAN transmit failed. Check motor power, bitrate, H/L and termination.");
      return false;
    }
    if (alerts & TWAI_ALERT_TX_SUCCESS) return true;
    delay(1);
  }
  // Stop so a delayed configuration frame cannot surprise the next command.
  twai_stop();
  Serial.println("TX completion timed out; CAN stopped. Check bus, then reset ESP32.");
  return false;
}


// Read a parameter; require matching type, motor, host and index.
static int readParameter(uint8_t motor, uint16_t index, uint8_t *value) {
  if (!canReady) return -1;
  twai_clear_receive_queue();
  uint8_t data[8] = {};
  data[0] = uint8_t(index);
  data[1] = uint8_t(index >> 8);
  if (!sendFrame(commandID(17, motor), data)) return -1;
  const uint32_t start = millis();
  while (millis() - start < 250) {
    twai_message_t rx = {};
    if (twai_receive(&rx, pdMS_TO_TICKS(5)) != ESP_OK) continue;
    if (!rx.extd || rx.rtr || rx.data_length_code != 8) continue;
    if (((rx.identifier >> 24) & 0x1F) != 17) continue;
    if ((rx.identifier & 0xFF) != HOST_ID) continue;
    if (((rx.identifier >> 8) & 0xFF) != motor) continue;
    const uint16_t replyIndex = uint16_t(rx.data[0]) | (uint16_t(rx.data[1]) << 8);
    if (replyIndex != index) continue;
    memcpy(value, rx.data + 4, 4);
    return 1;
  }
  return 0;
}

static int readAngle(uint8_t motor, float &radians) {
  uint8_t value[4] = {};
  const int result = readParameter(motor, POSITION_INDEX, value);
  if (result != 1) return result;
  static_assert(sizeof(float) == 4, "Requires 32-bit float");
  memcpy(&radians, value, 4);
  return isfinite(radians) ? 1 : -2;
}

// Type 18 writes four value bytes; floats and mode bytes are encoded differently.
static bool writeParameter(uint8_t motor, uint16_t index, const uint8_t *value) {
  uint8_t data[8] = {};
  data[0] = uint8_t(index);
  data[1] = uint8_t(index >> 8);
  memcpy(data + 4, value, 4);
  if (!sendFrame(commandID(18, motor), data)) return false;
  delay(10);
  return true;
}

static bool writeFloatVerified(uint8_t motor, uint16_t index, float value) {
  uint8_t bytes[4], reply[4];
  memcpy(bytes, &value, 4);
  if (!writeParameter(motor, index, bytes)) return false;
  if (readParameter(motor, index, reply) != 1) {
    Serial.printf("Parameter 0x%04X readback failed.\n", index);
    return false;
  }
  float actual;
  memcpy(&actual, reply, 4);
  if (!isfinite(actual) || fabsf(actual - value) > 0.0001f * fmaxf(1.0f, fabsf(value))) {
    Serial.printf("Parameter 0x%04X mismatch: wanted %.6f, read %.6f.\n",
                  index, double(value), double(actual));
    return false;
  }
  return true;
}

static bool setPositionMode(uint8_t motor) {
  // run_mode is an 8-bit integer; 1 = profiled position (PP), not a float.
  const uint8_t mode[4] = {1, 0, 0, 0};
  uint8_t reply[4] = {};
  if (!writeParameter(motor, 0x7005, mode)) return false;
  if (readParameter(motor, 0x7005, reply) != 1 || reply[0] != 1) {
    Serial.println("Position mode readback failed.");
    return false;
  }
  return true;
}

static void stopMotor(uint8_t motor) {
  // Zero payload: disable without requesting fault clearing.
  if (sendFrame(commandID(4, motor))) {
    Serial.printf("Stop frame transmitted to motor %u; torque disabled if accepted.\n", motor);
  } else Serial.println("Stop transmission FAILED. Cut motor power if necessary.");
}

static void moveMotor(uint8_t motor, float goalDegrees) {
  if (!isfinite(goalDegrees) || fabsf(goalDegrees) > MAX_ABS_TARGET_DEG) {
    Serial.printf("Target must be finite and within +/-%.1f degrees.\n", double(MAX_ABS_TARGET_DEG));
    return;
  }
  float current;
  if (readAngle(motor, current) != 1) {
    Serial.println("No valid starting angle; move cancelled.");
    return;
  }
  const float target = goalDegrees * (PI / 180.0f);
  if (fabsf(target - current) * (180.0f / PI) > MAX_MOVE_DELTA_DEG) {
    Serial.printf("Move exceeds %.1f-degree step limit; use smaller steps.\n", double(MAX_MOVE_DELTA_DEG));
    return;
  }
  Serial.printf("Motor %u: %.3f -> %.3f deg\n", motor,
                double(current) * 180.0 / PI, double(goalDegrees));
  // Stop before changing modes. No fault clearing or automatic calibration.
  if (!sendFrame(commandID(4, motor))) return;
  delay(20);
  if (readAngle(motor, current) != 1) {
    Serial.println("Starting angle read failed after stop; motor remains disabled.");
    return;
  }
  if (fabsf(current) * (180.0f / PI) > MAX_ABS_TARGET_DEG ||
      fabsf(target - current) * (180.0f / PI) > MAX_MOVE_DELTA_DEG) {
    Serial.println("Starting position changed or is outside target range; move cancelled.");
    return;
  }
  // Set and verify limits and a hold target BEFORE enabling, avoiding a stale target.
  if (!setPositionMode(motor) ||
      !writeFloatVerified(motor, 0x7018, MOVE_CURRENT_A) ||
      !writeFloatVerified(motor, 0x7017, MOVE_SPEED_RAD_S) ||
      !writeFloatVerified(motor, 0x7024, MOVE_SPEED_RAD_S) ||
      !writeFloatVerified(motor, 0x7025, MOVE_ACCEL_RAD_S2) ||
      !writeFloatVerified(motor, 0x7016, current)) {
    Serial.println("Setup verification failed; move cancelled.");
    stopMotor(motor);
    return;
  }
  if (!sendFrame(commandID(3, motor))) {
    stopMotor(motor);
    return;
  }
  delay(20);
  if (!writeFloatVerified(motor, 0x7016, target)) {
    Serial.println("Target verification failed; requesting stop.");
    stopMotor(motor);
    return;
  }
  Serial.printf("Motor %u target accepted by parameter readback: %.3f deg.\n",
                motor, double(goalDegrees));
  Serial.println("Use angle to check actual position; stop ID releases torque.");
  Serial.println("Target readback does not confirm motion completion or absence of faults.");
}

static void printAngles() {
  Serial.println("\nMotor angle readings (queried sequentially):");
  for (size_t i = 0; i < sizeof(MOTOR_IDS) / sizeof(MOTOR_IDS[0]); ++i) {
    const uint8_t motor = MOTOR_IDS[i];
    float radians = 0.0f;
    const int result = readAngle(motor, radians);
    if (result == 1) {
      Serial.printf("Motor %u: %.3f deg  (%.6f rad)\n",
                    motor, double(radians) * 180.0 / PI, double(radians));
    } else if (result == 0) {
      Serial.printf("Motor %u: no matching position reply (timeout).\n", motor);
    } else if (result == -2) {
      Serial.printf("Motor %u: invalid position value.\n", motor);
    } else {
      Serial.printf("Motor %u: CAN transport error; remaining reads cancelled.\n", motor);
      status();
      return;
    }
  }
}

static void help() {
  Serial.println("\nRobStride RS01 angle reader -- ESP32 + SN65HVD230");
  Serial.println("115200 baud; select Newline or Both NL & CR.");
  Serial.println("Commands: angle | move ID DEGREES | stop ID | status | help");
  Serial.println("move uses absolute angles and holds torque; stop disables torque.");
  Serial.println("No zero, ID or flash writes. Only listed motor IDs may move.");
  Serial.print("Configured motor IDs:");
  for (uint8_t id : MOTOR_IDS) Serial.printf(" %u", id);
  Serial.println();
}

static bool parseMotorID(const char *text, uint8_t &motor) {
  if (!text || !*text) return false;
  for (const char *p = text; *p; ++p) if (*p < '0' || *p > '9') return false;
  errno = 0;
  char *end;
  long id = strtol(text, &end, 10);
  if (errno || *end || id < 1 || id > 127) return false;
  for (uint8_t configured : MOTOR_IDS) {
    if (configured == id) { motor = uint8_t(id); return true; }
  }
  return false;
}

static void command(char *line) {
  char *context = nullptr;
  char *tokens[4] = {};
  int n = 0;
  for (char *t = strtok_r(line, " \t", &context); t && n < 4;
       t = strtok_r(nullptr, " \t", &context)) tokens[n++] = t;
  if (!n) return;
  if (n == 1 && !strcmp(tokens[0], "angle")) { printAngles(); return; }
  if (n == 1 && !strcmp(tokens[0], "status")) { status(); return; }
  if (n == 1 && !strcmp(tokens[0], "help")) { help(); return; }
  uint8_t motor;
  if (n == 2 && !strcmp(tokens[0], "stop") && parseMotorID(tokens[1], motor)) {
    stopMotor(motor); return;
  }
  if (n == 3 && !strcmp(tokens[0], "move") && parseMotorID(tokens[1], motor)) {
    errno = 0;
    char *end;
    float goal = strtof(tokens[2], &end);
    if (!errno && end != tokens[2] && !*end && isfinite(goal)) {
      moveMotor(motor, goal); return;
    }
  }
  Serial.println("Invalid command or ID. Examples: angle | move 1 90 | stop 1");
  Serial.println("Motor ID must be listed in MOTOR_IDS; angle must be a finite number.");
}
void setup() {
  Serial.begin(115200);
  delay(700);
  twai_general_config_t g = TWAI_GENERAL_CONFIG_DEFAULT(CAN_TX, CAN_RX, TWAI_MODE_NORMAL);
  twai_timing_config_t t = TWAI_TIMING_CONFIG_1MBITS();
  twai_filter_config_t f = TWAI_FILTER_CONFIG_ACCEPT_ALL();
  g.tx_queue_len = 4;
  g.rx_queue_len = 32;
  g.alerts_enabled = TWAI_ALERT_TX_SUCCESS | TWAI_ALERT_TX_FAILED | TWAI_ALERT_BUS_OFF;
  esp_err_t e = twai_driver_install(&g, &t, &f);
  if (e == ESP_OK) e = twai_start();
  canReady = (e == ESP_OK);
  if (!canReady) Serial.printf("CAN initialization failed: %s\n", esp_err_to_name(e));
  help();
}

void loop() {
  static char line[80];
  static size_t length = 0;
  static bool overflow = false;
  while (Serial.available()) {
    char c = char(Serial.read());
    if (c == '\r' || c == '\n') {
      if (overflow) Serial.println("Command too long; discarded.");
      else if (length) { line[length] = '\0'; command(line); }
      length = 0;
      overflow = false;
    } else if (!overflow) {
      if (length + 1 < sizeof(line)) line[length++] = c;
      else overflow = true;
    }
  }
  // Discard unsolicited reports so auto-reporting cannot fill the RX queue.
  if (canReady) twai_clear_receive_queue();
  delay(2);
}

