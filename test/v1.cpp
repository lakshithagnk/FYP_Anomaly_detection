#include <Arduino.h>
#include "model.h"

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include "microtflite.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model Parameters
#define N_FEATURES 31
#define WINDOW_SIZE 32
#define N_CLASSES 4

// OLED Display Parameters
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1
#define I2C_SDA 17
#define I2C_SCL 18

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

// Fault labels 
const char* CLASS_NAMES[N_CLASSES] = {
    "Normal",
    "High Resistance",
    "Low Capacity",
    "Self Discharge"
};

//    Normalization Parameters
float feature_mean[N_FEATURES] = {
    13.929540534137237, 25.38316258588433, 25.197613620581194, 3.309591901342288, 3.3084532078462145, 3.30859882314976, 3.308792570443536, 3.307834327998496, 3.3083765491858856, 3.3081705350359063, 3.3088580549496336, 3.3085457382580503, 3.3071277562641437, 3.3083508083952187, 3.3089596272150197, 3.3083434699312932, 3.309393227698668, 3.309844356347474, 3.309429362433927, 3.3092228785242193, 3.308979941259609, 3.3086337814432647, 3.3088253045268496, 3.3083523340931187, 3.309269895345702, 3.307545934331676, 3.309151366796565, 3.3081394328570384, 3.3092217225068024, 3.309137084058398, 3.309778230517225
};

float feature_std[N_FEATURES] = {
    36.872187296768416, 0.7408134689191022, 0.4649730677659029, 0.045108408782186984, 0.04816595802869389, 0.046527816750793545, 0.04616101219587674, 0.04859359804081387, 0.0468844020526497, 0.047782887131317106, 0.04610337101271268, 0.04695275516224864, 0.05014159867215784, 0.04734723581606758, 0.04587827562963655, 0.04702557358231738, 0.04587638845711413, 0.045928884460106914, 0.046367584378058574, 0.046950320675638764, 0.046863670211612564, 0.04747757203448686, 0.047304455173237274, 0.05000085048055305, 0.04623302623164611, 0.051293774612255455, 0.04713753317349633, 0.04945160970819231, 0.046406416731439235, 0.04692258666117734, 0.04541334716647866
};

//    Sliding Window Buffer
float window_buffer[WINDOW_SIZE][N_FEATURES];
uint8_t window_index = 0;
bool window_filled = false;

//    TensorFlow Lite Micro Setup
constexpr int TENSOR_ARENA_SIZE = 120 * 1024;
uint8_t tensor_arena[TENSOR_ARENA_SIZE];

const tflite::Model* model;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

tflite::AllOpsResolver resolver;

//    Utility Functions
void normalize_features(float* data){
    for (int i = 0; i < N_FEATURES; i++){
        data[i] = (data[i] - feature_mean[i]) / feature_std[i];
    }
}

//    Update sliding window 
void update_window(float* features){
    for (int i = 0; i < N_FEATURES; i++){
        window_buffer[window_index][i] = features[i];
    }

    window_index++;

    if (window_index >= WINDOW_SIZE)    {
        window_index = 0;
        window_filled = true;
    }
}

/* Copy window to model input tensor */
void load_model_input(){
    int idx = 0;

    for (int t = 0; t < WINDOW_SIZE; t++){
        for (int f = 0; f < N_FEATURES; f++){
            input->data.f[idx++] = window_buffer[t][f];
        }
    }
}

void oled_show_message(const char* msg){
    display.clearDisplay();
    display.setCursor(0, 10);
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.println(msg);
    display.display();
}

void oled_show_status(const char* step, const char* value){
    display.clearDisplay();

    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Battery Monitor");

    display.setCursor(0, 20);
    display.print("Step: ");
    display.println(step);

    display.setCursor(0, 40);
    display.print("Status: ");
    display.println(value);

    display.display();
}

void oled_show_prediction(const char* label, float prob){
    display.clearDisplay();

    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("FAULT RESULT");

    display.setTextSize(2);
    display.setCursor(0, 20);
    display.println(label);

    display.setTextSize(1);
    display.setCursor(0, 50);
    display.print("Conf: ");
    display.print(prob, 2);

    display.display();
}

// Run inference 
void run_inference(){
    oled_show_status("Inference", "Running");

    load_model_input();

    if (interpreter->Invoke() != kTfLiteOk){
        Serial.println("Inference failed");
        oled_show_message("Inference Failed");
        return;
    }

    float max_prob = -1;
    int best_class = -1;

    for (int i = 0; i < N_CLASSES; i++){
        float prob = output->data.f[i];

        if (prob > max_prob){
            max_prob = prob;
            best_class = i;
        }
    }

    Serial.print("Prediction: ");
    Serial.print(CLASS_NAMES[best_class]);
    Serial.print(" (");
    Serial.print(max_prob, 4);
    Serial.println(")");

    oled_show_prediction(CLASS_NAMES[best_class], max_prob);

    delay(2000);
}

/* =============================
   Simulated Feature Input
   Replace this with real sensors
   ============================= */

void read_features(float* features){
    // for (int i = 0; i < N_FEATURES; i++){
    //     features[i] = analogRead(i % 10); // placeholder
    // }
}



void setup(){
    Serial.begin(115200);

    Wire.begin(I2C_SDA, I2C_SCL);

    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
        Serial.println("OLED init failed");
        while (1);
    }

    oled_show_message("Initializing...");

    delay(1000);

    Serial.println("Initializing model...");

    model = tflite::GetModel(battery_model_full_int8_tflite);

    static tflite::MicroInterpreter static_interpreter(
        model,
        resolver,
        tensor_arena,
        TENSOR_ARENA_SIZE
    );

    interpreter = &static_interpreter;

    interpreter->AllocateTensors();

    input = interpreter->input(0);
    output = interpreter->output(0);

    oled_show_message("Model Ready");

    Serial.println("Model initialized.");
    delay(1000);
}


void loop(){
    float features[N_FEATURES];

    oled_show_status("Reading", "Sensors");
    read_features(features);

    oled_show_status("Processing", "Normalize");
    normalize_features(features);

    oled_show_status("Buffering", "Window");
    update_window(features);

    if (window_filled){
        run_inference();
    }

    delay(200);
}
