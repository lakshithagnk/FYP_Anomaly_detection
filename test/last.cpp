#include <Arduino.h>
#include "battery_model.h"

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_timer.h"

// ================= CONFIG =================
#define N_FEATURES 31
#define WINDOW_SIZE 32
#define N_CLASSES 4

#define SAMPLE_INTERVAL_MS 200   // Faster sampling

// OLED
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define SCREEN_ADDRESS 0x3C

// ================= MUX CONFIG =================
#define MUX1_SIG 1
#define MUX2_SIG 2

// Shared select pins of mux 01
#define S0 10
#define S1 11
#define S2 12
#define S3 13

// ================= GLOBALS =================
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

float window_buffer[WINDOW_SIZE][N_FEATURES];
int window_index = 0;
bool window_ready = false;

unsigned long last_sample_time = 0;

// ================= TFLITE =================
constexpr int TENSOR_ARENA_SIZE = 120 * 1024;
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
tflite::AllOpsResolver resolver;

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

void normalize(float* data) {
    for (int i = 0; i < N_FEATURES; i++) {
        data[i] = (data[i] - feature_mean[i]) / feature_std[i];
    }
}

// ================= OLED =================
void oled_prediction(const char* label, float prob) {
    display.clearDisplay();
    display.setTextSize(2);
    display.setCursor(0,20);
    display.println(label);

    display.setTextSize(1);
    display.setCursor(0,50);
    display.print("Conf: ");
    display.print(prob*100,1);
    display.print("%");

    display.display();
}

// ================= MUX CONTROL =================
inline void setMuxChannel(uint8_t ch) {
    digitalWrite(S0, ch & 0x01);
    digitalWrite(S1, (ch >> 1) & 0x01);
    digitalWrite(S2, (ch >> 2) & 0x01);
    digitalWrite(S3, (ch >> 3) & 0x01);
}

// ================= FAST ADC READ =================
inline float readADC(int pin) {
    return analogRead(pin) / 4095.0f;  // normalize 0–1
}

// ================= READ FEATURES =================
void read_features(float* features) {

    int idx = 0;

    for (int ch = 0; ch < 16; ch++) {

        setMuxChannel(ch);
        delayMicroseconds(5); // settle time

        float v1 = readADC(MUX1_SIG);
        float v2 = readADC(MUX2_SIG);

        if (idx < N_FEATURES) features[idx++] = v1;
        if (idx < N_FEATURES) features[idx++] = v2;
    }
}

// ================= WINDOW =================
void update_window(float* features) {

    if (!window_ready) {
        memcpy(window_buffer[window_index++], features, sizeof(float)*N_FEATURES);
        if (window_index >= WINDOW_SIZE) window_ready = true;
    } else {
        memmove(window_buffer[0], window_buffer[1],
                sizeof(float)*(WINDOW_SIZE-1)*N_FEATURES);

        memcpy(window_buffer[WINDOW_SIZE-1], features,
               sizeof(float)*N_FEATURES);
    }
}

// ================= INFERENCE =================
int run_inference(float& conf, float& latency_ms) {

    int idx = 0;
    for (int t = 0; t < WINDOW_SIZE; t++) {
        for (int f = 0; f < N_FEATURES; f++) {
            input->data.f[idx++] = window_buffer[t][f];
        }
    }

    int64_t start = esp_timer_get_time();

    interpreter->Invoke();

    int64_t end = esp_timer_get_time();
    latency_ms = (end - start) / 1000.0;

    float max_val = -1e9;
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

// ================= SETUP =================
void setup() {

    Serial.begin(115200);

    // MUX pins
    pinMode(S0, OUTPUT);
    pinMode(S1, OUTPUT);
    pinMode(S2, OUTPUT);
    pinMode(S3, OUTPUT);

    analogReadResolution(12);

    // OLED
    Wire.begin();
    display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS);

    // Model
    const tflite::Model* model = tflite::GetModel(battery_model_fp32_tflite);

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

    interpreter = &static_interpreter;

    interpreter->AllocateTensors();
    input = interpreter->input(0);
    output = interpreter->output(0);

    display.clearDisplay();
    display.setCursor(0,20);
    display.println("System Ready");
    display.display();
}

// ================= LOOP =================
void loop() {

    if (millis() - last_sample_time >= SAMPLE_INTERVAL_MS) {

        last_sample_time = millis();

        float features[N_FEATURES];

        read_features(features);
        normalize(features);
        update_window(features);

        if (window_ready) {

            float conf, latency;
            int pred = run_inference(conf, latency);

            Serial.print("Pred: ");
            Serial.print(pred);
            Serial.print(" Conf: ");
            Serial.print(conf);
            Serial.print(" Latency: ");
            Serial.println(latency);

            const char* CLASS_NAMES[4] = {
                "Normal","HighRes","LowCap","SelfDisc"
            };

            oled_prediction(CLASS_NAMES[pred], conf);
        }
    }
}