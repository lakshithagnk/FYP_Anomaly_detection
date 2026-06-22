// corrected working code for testing the model with different datasets (normal, low cap, self discharge)

#include <Arduino.h>
#include "battery_model.h"
#include "test_data.h"

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ================= CONFIG =================
#define N_FEATURES 31
#define WINDOW_SIZE 32
#define N_CLASSES 4
#define SAMPLE_INTERVAL_MS 100

// OLED Configuration
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

// Simulation state
unsigned long last_sample_time = 0;
int current_step = 0;

// Window buffer
float window_buffer[WINDOW_SIZE][N_FEATURES];
int window_index = 0;
bool window_ready = false;

// TensorFlow
constexpr int TENSOR_ARENA_SIZE = 120 * 1024;
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;
tflite::AllOpsResolver resolver;

// ================= NORMALIZATION =================
float feature_mean[N_FEATURES] = {
    13.9295, 25.3831, 25.1976, 3.3095, 3.3084, 3.3085, 3.3087, 3.3078, 3.3083, 3.3081, 
    3.3088, 3.3085, 3.3071, 3.3083, 3.3089, 3.3083, 3.3093, 3.3098, 3.3094, 3.3092, 
    3.3089, 3.3086, 3.3088, 3.3083, 3.3092, 3.3075, 3.3091, 3.3081, 3.3092, 3.3091, 3.3097
};
float feature_std[N_FEATURES] = { 
    36.8721, 0.7408, 0.4649, 0.0451, 0.0481, 0.0465, 0.0461, 0.0485, 0.0468, 0.0477, 
    0.0461, 0.0469, 0.0501, 0.0473, 0.0458, 0.0470, 0.0458, 0.0459, 0.0463, 0.0469, 
    0.0468, 0.0474, 0.0473, 0.0500, 0.0462, 0.0512, 0.0471, 0.0494, 0.0464, 0.0469, 0.0454
};

void normalize(float* data) {
    for (int i = 0; i < N_FEATURES; i++) {
        data[i] = (data[i] - feature_mean[i]) / feature_std[i];
    }
}

// ================= OLED HELPERS =================
void oled_print(const char* line1, const char* line2 = "") {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE); 
    display.setTextSize(1);
    display.setCursor(0, 10);
    display.println(line1);
    display.setCursor(0, 30);
    display.println(line2);
    display.display();
}

void oled_prediction(const char* label, float prob, int step, bool is_match) {
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    
    // Header Row
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.print("Step: "); display.print(step);
    display.setCursor(80, 0);
    display.print(is_match ? "[OK]" : "[ERR]");

    // Result Row
    display.setTextSize(2);
    display.setCursor(0, 20);
    display.println(label);

    // Footer Row
    display.setTextSize(1);
    display.setCursor(0, 50);
    display.print("Conf: "); 
    display.print(prob * 100, 1);
    display.print("%");

    display.display();
}

// ================= DATA LOGIC =================
void read_features(float* features, int& actual_label) {
    if (current_step >= TOTAL_CYCLE_STEPS) {
        oled_print("Simulation Done", "End of Data");
        while (1) delay(100);
    }

    for (int i = 0; i < N_FEATURES; i++) {
        features[i] = self_dis_cycle_data[current_step][i];
    }

    // We assume the test data provided is 'Self Disch' (Class 3)
    actual_label = 3; 
    current_step++;
}

void update_window(float* features) {
    if (!window_ready) {
        memcpy(window_buffer[window_index], features, sizeof(float) * N_FEATURES);
        window_index++;
        if (window_index >= WINDOW_SIZE) window_ready = true;
    } else {
        memmove(window_buffer[0], window_buffer[1], sizeof(float) * (WINDOW_SIZE - 1) * N_FEATURES);
        memcpy(window_buffer[WINDOW_SIZE - 1], features, sizeof(float) * N_FEATURES);
    }
}

// ================= INFERENCE =================
int run_inference(float& confidence) {
    int idx = 0;
    for (int t = 0; t < WINDOW_SIZE; t++) {
        for (int f = 0; f < N_FEATURES; f++) {
            input->data.f[idx++] = window_buffer[t][f];
        }
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        Serial.println("Error: Invoke failed");
        return -1;
    }

    float max_val = -1e9;
    int best_class = -1;
    for (int i = 0; i < N_CLASSES; i++) {
        float val = output->data.f[i];
        if (val > max_val) {
            max_val = val;
            best_class = i;
        }
    }

    confidence = max_val;
    return best_class;
}

// ================= SETUP =================
void setup() {
    Serial.begin(115200);
    delay(1000); // Wait for serial monitor

    // 1. Start I2C
    Wire.begin(I2C_SDA, I2C_SCL);

    // 2. Start OLED
    if(!display.begin(SSD1306_SWITCHCAPVCC, SCREEN_ADDRESS)) {
        Serial.println(F("SSD1306 allocation failed"));
        for(;;); 
    }
    
    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    oled_print("System Init...", "Loading AI Model");

    // 3. Start TFLite
    model = tflite::GetModel(battery_model_fp32_tflite);
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

    interpreter = &static_interpreter;
    interpreter->AllocateTensors();
    input = interpreter->input(0);
    output = interpreter->output(0);

    Serial.println("Initialization Complete.");
    oled_print("Ready", "Waiting for data...");
    delay(1000);
}

// ================= LOOP =================
void loop() {
    unsigned long now = millis();

    if (now - last_sample_time >= SAMPLE_INTERVAL_MS) {
        last_sample_time = now;

        float features[N_FEATURES];
        int actual_label;

        read_features(features, actual_label);
        normalize(features);
        update_window(features);

        if (window_ready) {
            float conf;
            int pred = run_inference(conf);

            if (pred >= 0) {
                bool match = (pred == actual_label);

                // --- Detailed Serial Output ---
                Serial.print("Step ");
                Serial.print(current_step);
                Serial.print(" | Pred: ");
                Serial.print(CLASS_NAMES[pred]);
                Serial.print(" | Actual: ");
                Serial.print(CLASS_NAMES[actual_label]);
                Serial.print(" | ");

                if (match) {
                    Serial.println("MATCH ✅");
                } else {
                    Serial.println("MISMATCH ❌");
                }

                // --- OLED Output ---
                oled_prediction(CLASS_NAMES[pred], conf, current_step, match);
            }
        } else {
            // Buffer progress feedback
            char buf[20];
            sprintf(buf, "%d/%d samples", window_index, WINDOW_SIZE);
            oled_print("Filling Window", buf);
        }
    }
}