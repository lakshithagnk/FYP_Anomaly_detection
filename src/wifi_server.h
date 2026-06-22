#ifndef WIFI_SERVER_H
#define WIFI_SERVER_H

#include <Arduino.h>
#define WIFI_N_FEATURES 31

// ================= INFERENCE RESULT (filled by main.cpp's callback) =================
struct InferenceResult {
    bool  ok;            // false if inference could not run for some reason
    bool  window_ready;  // false while the sliding window is still filling
    int   class_index;   // -1 if not available
    const char* class_name;   // nullptr if not available
    float confidence;    // 0..100 (percentage), 0 if not available
    float latency_ms;    // 0 if not available
    bool  fault;          // true if class_index != CLASS_NORMAL
    unsigned long fault_count; // cumulative distinct fault episodes since boot
};

typedef void (*InferenceCallback)(const float* raw_features, InferenceResult* out);

void wifi_server_set_inference_callback(InferenceCallback cb);

// ================= AP CONFIG =================
#define WIFI_AP_SSID     "BatteryMonitor_AP"
#define WIFI_AP_PASSWORD "12345678"     // 8+ chars required by WiFi lib
#define WIFI_AP_CHANNEL  1
#define WIFI_AP_HIDDEN   0
#define WIFI_AP_MAX_CONN 4

#define WIFI_HTTP_PORT   80
#define WIFI_DATA_PATH   "/data"          // POST endpoint

void wifi_server_begin();

void wifi_server_handle();

unsigned long wifi_server_total_samples();
unsigned long wifi_server_total_errors();

bool wifi_server_result_available();

void wifi_server_get_latest_result(InferenceResult* out);

#endif 