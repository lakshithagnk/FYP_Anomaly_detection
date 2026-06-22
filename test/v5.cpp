#include <Arduino.h>
#include "battery_model.h"
#include "test_data.h"

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// 🔹 Added for metrics
#include "esp_heap_caps.h"
#include "esp_timer.h"

// ================= CONFIG =================
#define N_FEATURES 31
#define WINDOW_SIZE 32
#define N_CLASSES 4
#define SAMPLE_INTERVAL_MS 1000

// OLED
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define I2C_SDA 17
#define I2C_SCL 18
#define SCREEN_ADDRESS 0x3C

// ================= GLOBALS =================
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

const char* CLASS_NAMES[N_CLASSES] = {
    "Normal",
    "High Resist",
    "Low Cap",
    "Self Disch"
};

unsigned long last_sample_time = 0;
int current_step = 0;

float window_buffer[WINDOW_SIZE][N_FEATURES];
int window_index = 0;
bool window_ready = false;

// ================= TFLITE =================
constexpr int TENSOR_ARENA_SIZE = 120 * 1024;
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

const tflite::Model* model;
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
void oled_print(const char* l1, const char* l2="") {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(0,10);
    display.println(l1);
    display.setCursor(0,30);
    display.println(l2);
    display.display();
}

void oled_prediction(const char* label, float prob, int step, bool match) {
    display.clearDisplay();

    display.setTextSize(1);
    display.setCursor(0,0);
    display.print("Step: "); display.print(step);

    display.setCursor(80,0);
    display.print(match ? "[OK]" : "[ERR]");

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

// ================= DATA =================
void read_features(float* features, int& actual_label) {

    if (current_step >= TOTAL_CYCLE_STEPS) {
        oled_print("Done","End");
        while(1);
    }

    for (int i=0;i<N_FEATURES;i++) {
        features[i] = high_res_cycle_data[current_step][i];
    }

    actual_label = 1;
    current_step++;
}

void update_window(float* features) {
    if (!window_ready) {
        memcpy(window_buffer[window_index], features, sizeof(float)*N_FEATURES);
        window_index++;
        if (window_index>=WINDOW_SIZE) window_ready=true;
    } else {
        memmove(window_buffer[0], window_buffer[1],
                sizeof(float)*(WINDOW_SIZE-1)*N_FEATURES);

        memcpy(window_buffer[WINDOW_SIZE-1], features,
               sizeof(float)*N_FEATURES);
    }
}

// ================= INFERENCE =================
int run_inference(float& conf, float& latency_ms) {

    int idx=0;
    for(int t=0;t<WINDOW_SIZE;t++) {
        for(int f=0;f<N_FEATURES;f++) {
            input->data.f[idx++] = window_buffer[t][f];
        }
    }

    int64_t start = esp_timer_get_time();

    if (interpreter->Invoke()!=kTfLiteOk) {
        Serial.println("Invoke failed");
        return -1;
    }

    int64_t end = esp_timer_get_time();
    latency_ms = (end - start)/1000.0;

    float max_val=-1e9;
    int best=-1;

    for(int i=0;i<N_CLASSES;i++) {
        float v = output->data.f[i];
        if(v>max_val){
            max_val=v;
            best=i;
        }
    }

    conf=max_val;
    return best;
}

// ================= SETUP =================
void setup() {

    Serial.begin(115200);
    delay(1000);

    size_t heap_before = heap_caps_get_free_size(MALLOC_CAP_8BIT);
    Serial.print("Heap before: "); Serial.println(heap_before);

    Wire.begin(I2C_SDA, I2C_SCL);

    display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS);
    oled_print("Init","Loading Model");

    model = tflite::GetModel(battery_model_fp32_tflite);

    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

    interpreter = &static_interpreter;

    interpreter->AllocateTensors();
    input = interpreter->input(0);
    output = interpreter->output(0);

    size_t heap_after = heap_caps_get_free_size(MALLOC_CAP_8BIT);

    Serial.print("Heap after: "); Serial.println(heap_after);
    Serial.print("Model Memory: ");
    Serial.println(heap_before - heap_after);

    Serial.print("Tensor Arena: ");
    Serial.print(TENSOR_ARENA_SIZE/1024);
    Serial.println(" KB");

    oled_print("Ready","Running...");
}

// ================= LOOP =================
void loop() {

    unsigned long now = millis();

    if (now - last_sample_time >= SAMPLE_INTERVAL_MS) {

        last_sample_time = now;

        float features[N_FEATURES];
        int actual;

        read_features(features, actual);
        normalize(features);
        update_window(features);

        if (window_ready) {

            float conf, latency;
            int pred = run_inference(conf, latency);

            if (pred >= 0) {

                bool match = (pred==actual);

                Serial.print("Step "); Serial.print(current_step);
                Serial.print(" | Pred: "); Serial.print(CLASS_NAMES[pred]);
                Serial.print(" | Conf: "); Serial.print(conf);
                Serial.print(" | Latency: "); Serial.print(latency);
                Serial.print(" ms | ");
                Serial.println(match ? "MATCH" : "MISMATCH");

                oled_prediction(CLASS_NAMES[pred], conf, current_step, match);
            }

        } else {

            char buf[20];
            sprintf(buf,"%d/%d",window_index,WINDOW_SIZE);
            oled_print("Buffering",buf);
        }
    }
}