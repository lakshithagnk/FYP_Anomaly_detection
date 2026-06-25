#include <Arduino.h>
#include <string.h>
#include "battery_model.h"
#include "wifi_server.h" 
#include <WiFi.h>

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_timer.h"

// ================= MODEL / WINDOW CONFIG =================
#define WINDOW_SIZE 32
#define N_CLASSES 4
#define N_FEATURES 31

#define SAMPLE_TIMEOUT_MS 30000

// ================= OLED CONFIG =================
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C
#define OLED_SDA 10
#define OLED_SCL 11

// ================= BUZZER =================
#define BUZZER_PIN 39
#define BUZZ_ON_MS  120
#define BUZZ_OFF_MS 400

// ================= GLOBALS =================
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

static const char* CLASS_NAMES[N_CLASSES] = {
    "Normal", "High Resist", "Low Cap", "Self Disch"
};
#define CLASS_NORMAL 0

float window_buffer[WINDOW_SIZE][N_FEATURES];
int   window_head  = 0;     // index of the OLDEST row once the buffer is full
int   window_count = 0;
bool  window_ready = false;

unsigned long last_sample_time = 0;   // millis() of last sample RECEIVED over MQTT

bool fault_flag = false;
bool buzzer_on = false;
unsigned long buzzer_last_toggle = 0;

// Display stats
unsigned long fault_count = 0;     // number of distinct fault episodes since boot
bool prev_fault_state = false;     // used to detect normal->fault edges

// ================= TFLITE =================
constexpr int TENSOR_ARENA_SIZE = 120 * 1024;
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
tflite::AllOpsResolver resolver;

void print_ram_usage(const char* stage) {
    const unsigned long heap_total = ESP.getHeapSize();
    const unsigned long heap_free = ESP.getFreeHeap();
    const unsigned long heap_used = heap_total - heap_free;

    Serial.printf("[RAM] %s\n", stage);
    Serial.printf("  Tensor Arena: %lu bytes\n", (unsigned long)TENSOR_ARENA_SIZE);
    Serial.printf("  Heap Used   : %lu bytes\n", heap_used);
    Serial.printf("  Heap Free   : %lu bytes\n", heap_free);
    Serial.printf("  Heap MinFree: %lu bytes\n", (unsigned long)ESP.getMinFreeHeap());
    Serial.printf("  Total RAM Footprint (Arena + Heap Used): %lu bytes\n", (unsigned long)TENSOR_ARENA_SIZE + heap_used);
}

// ================= NORMALIZATION =================

float feature_mean[N_FEATURES] = {
    13.9295,25.3831,25.1976,3.3095,3.3084,3.3085,3.3087,3.3078,3.3083,3.3081,
    3.3088,3.3085,3.3071,3.3083,3.3089,3.3083,3.3093,3.3098,3.3094,3.3092,
    3.3089,3.3086,3.3088,3.3083,3.3092,3.3075,3.3091,3.3081,3.3092,3.3091,3.3097
};

float feature_std[N_FEATURES] = {
    36.8721,0.7408,0.4649,0.0451,0.0481,0.0465,0.0461,0.0485,0.0468,0.0477,
    0.0461,0.0469,0.0501,0.0473,0.0458,0.0470,0.0458,0.0459,0.0463,0.0469,
    0.0468,0.0474,0.0473,0.0500,0.0462,0.0512,0.0471,0.0494,0.0464,0.0469,0.0454
};

inline void normalize(float* data) {
    for (int i = 0; i < N_FEATURES; i++) {
        data[i] = (data[i] - feature_mean[i]) / feature_std[i];
    }
}

inline void print_features(const float* features) {
    Serial.print("Features: ");
    for (int i = 0; i < N_FEATURES; i++) {
        Serial.print(features[i], 4);
        if (i < N_FEATURES - 1) Serial.print(", ");
    }
    Serial.println();
}

void oled_status(const char* label, float conf, float latency_ms, bool fault);
void oled_calibrating(int count, int total);

// ================= SLIDING WINDOW (circular buffer) =================
void update_window(const float* features) {
    memcpy(window_buffer[window_head], features, sizeof(float) * N_FEATURES);
    window_head = (window_head + 1) % WINDOW_SIZE;

    if (window_count < WINDOW_SIZE) {
        window_count++;
        if (window_count == WINDOW_SIZE) window_ready = true;
    }
}

// ================= INFERENCE =================
int run_inference(float& conf, float& latency_ms) {
    int idx = 0;
    for (int t = 0; t < WINDOW_SIZE; t++) {
        int row = (window_head + t) % WINDOW_SIZE;
        for (int f = 0; f < N_FEATURES; f++) {
            input->data.f[idx++] = window_buffer[row][f];
        }
    }

    int64_t start = esp_timer_get_time();
    interpreter->Invoke();
    int64_t end = esp_timer_get_time();
    latency_ms = (end - start) / 1000.0f;

    float max_val = -1e9f;
    int best = -1;
    for (int i = 0; i < N_CLASSES; i++) {
        float v = output->data.f[i];
        if (v > max_val) {
            max_val = v;
            best = i;
        }
    }

    conf = max_val;
    return best;
}

// ================= WIFI INFERENCE CALLBACK =================

void on_features_received(const float* raw_features, InferenceResult* out) {
    float features[N_FEATURES];
    memcpy(features, raw_features, sizeof(features));

    print_features(features);
    normalize(features);
    update_window(features);

    last_sample_time = millis();

    out->ok = true;
    out->window_ready = window_ready;
    out->fault_count = fault_count;

    if (!window_ready) {
        out->class_index = -1;
        out->class_name = nullptr;
        out->confidence = 0.0f;
        out->latency_ms = 0.0f;
        out->fault = false;

        oled_calibrating(window_count, WINDOW_SIZE);
        return;
    }

    float conf, latency;
    int pred = run_inference(conf, latency);

    fault_flag = (pred != CLASS_NORMAL);

    if (fault_flag && !prev_fault_state) fault_count++;
    prev_fault_state = fault_flag;

    out->class_index = pred;
    out->class_name = CLASS_NAMES[pred];
    out->confidence = conf; // Pass the decimal value (0..1), client parses to %
    out->latency_ms = latency;
    out->fault = fault_flag;
    out->fault_count = fault_count;

    Serial.print("Pred: ");
    Serial.print(CLASS_NAMES[pred]);
    Serial.print(" Conf: ");
    Serial.print(conf);
    Serial.print(" Latency(ms): ");
    Serial.println(latency);

    oled_status(CLASS_NAMES[pred], conf, latency, fault_flag);
}

// ================= BUZZER (non-blocking) =================
void updateBuzzer(bool fault) {
    if (!fault) {
        if (buzzer_on) {
            digitalWrite(BUZZER_PIN, LOW);
            buzzer_on = false;
        }
        return;
    }

    unsigned long now = millis();
    unsigned long interval = buzzer_on ? BUZZ_ON_MS : BUZZ_OFF_MS;

    if (now - buzzer_last_toggle >= interval) {
        buzzer_on = !buzzer_on;
        digitalWrite(BUZZER_PIN, buzzer_on ? HIGH : LOW);
        buzzer_last_toggle = now;
    }
}

// ================= TIME / STATS HELPERS =================
void format_uptime(unsigned long ms, char* buf, size_t buf_len) {
    unsigned long total_sec = ms / 1000;
    unsigned int hh = total_sec / 3600;
    unsigned int mm = (total_sec / 60) % 60;
    unsigned int ss = total_sec % 60;
    snprintf(buf, buf_len, "%02u:%02u:%02u", hh, mm, ss);
}

void oled_draw_header() {
    char ts[12];
    format_uptime(millis(), ts, sizeof(ts));

    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.print("Up ");
    display.print(ts);

    display.setCursor(92, 0);
    display.print("F:");
    display.print(fault_count);

    display.drawFastHLine(0, 9, SCREEN_WIDTH, SSD1306_WHITE);
}

// ================= OLED =================
void oled_status(const char* label, float conf, float latency_ms, bool fault) {
    display.clearDisplay();
    oled_draw_header();

    size_t len = strlen(label);
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(len <= 8 ? 2 : 1);
    display.setCursor(0, 12);
    display.println(label);

    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0, 32);
    display.print("Conf: ");
    display.print(conf * 100.0f, 1);
    display.print(" %");

    display.setCursor(0, 42);
    display.print("Lat:  ");
    display.print(latency_ms, 1);
    display.print(" ms");

    if (fault) {
        display.fillRect(0, 50, SCREEN_WIDTH, 13, SSD1306_WHITE);
        display.setTextColor(SSD1306_BLACK);
        display.setCursor(2, 52);
        display.print("STATUS: FAULT");
    } else {
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(0, 52);
        display.print("Status: Normal");
    }

    display.display();
}

void oled_calibrating(int count, int total) {
    display.clearDisplay();
    oled_draw_header();

    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(0, 16);
    display.print("Filling window...");

    display.setCursor(0, 30);
    display.print(count);
    display.print(" / ");
    display.print(total);

    int barWidth = SCREEN_WIDTH - 4;
    int filled = (barWidth * count) / total;
    display.drawRect(0, 44, barWidth + 4, 10, SSD1306_WHITE);
    display.fillRect(2, 46, filled, 6, SSD1306_WHITE);

    display.display();
}

// Shown when client has stopped sending MQTT data.
void oled_waiting_for_data(unsigned long since_last_ms) {
    display.clearDisplay();
    oled_draw_header();

    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(0, 20);
    display.print("Waiting for MQTT...");

    display.setCursor(0, 34);
    display.print("Last sample: ");
    display.print(since_last_ms / 1000);
    display.print("s ago");

    display.setCursor(0, 48);
    display.print(mqtt_client_is_connected() ? "Broker: Connected" : "Broker: Offline");

    display.display();
}

// ================= SETUP =================
void setup() {
    Serial.begin(115200);
    delay(200);

    print_ram_usage("Boot");

    pinMode(BUZZER_PIN, OUTPUT);
    digitalWrite(BUZZER_PIN, LOW);

    // OLED
    Wire.begin(OLED_SDA, OLED_SCL);
    if (!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
        Serial.println("SSD1306 init failed");
        while (true) delay(1000);
    }

    // Model
    const tflite::Model* model = tflite::GetModel(battery_model_fp32_tflite);

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("AllocateTensors() failed");
        display.clearDisplay();
        display.setCursor(0, 20);
        display.println("Model init FAIL");
        display.display();
        while (true) delay(1000);
    }

    print_ram_usage("After AllocateTensors");

    input = interpreter->input(0);
    output = interpreter->output(0);

    // Setup WiFi Station and MQTT secure client loop
    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 16);
    display.println("Connecting WiFi...");
    display.display();

    mqtt_client_begin(on_features_received);

    display.clearDisplay();
    display.setTextSize(1);
    display.setCursor(0, 12);
    display.println("System Ready ✅");
    display.setCursor(0, 26);
    display.print("SSID: ");
    display.println(WiFi.SSID());
    display.setCursor(0, 40);
    display.print("IP: ");
    display.println(WiFi.localIP());
    display.display();

    last_sample_time = millis();
}

// ================= LOOP =================
void loop() {
    // Keep connection alive, listen for messages
    mqtt_client_handle();

    // Check watchdog timer for incoming messages
    unsigned long since_last = millis() - last_sample_time;
    if (since_last > SAMPLE_TIMEOUT_MS) {
        oled_waiting_for_data(since_last);
    }

    updateBuzzer(fault_flag);
}