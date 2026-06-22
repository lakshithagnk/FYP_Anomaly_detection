#ifndef WIFI_SERVER_H
#define WIFI_SERVER_H

#include <Arduino.h>

#define N_FEATURES 31

// ================= INFERENCE RESULT STRUCT =================
struct InferenceResult {
    bool  ok;            
    bool  window_ready;  
    int   class_index;  
    const char* class_name; 
    float confidence;   
    float latency_ms;   
    bool  fault;          
    unsigned long fault_count; 
};

typedef void (*InferenceCallback)(const float* raw_features, InferenceResult* out);

// ================= WIFI & MQTT CONFIGURATION =================
#define WIFI_SSID     "nuwan"
#define WIFI_PASSWORD "1968nuwan"

// Cloud MQTT Broker (e.g. HiveMQ Cloud, EMQX Cloud, or public testing brokers)
#define MQTT_BROKER   "85d119b1fc5546828fe0484af72962c3.s1.eu.hivemq.cloud" 
#define MQTT_PORT     8883                    
#define MQTT_USER     "fypG21"       
#define MQTT_PASS     "2026FYPg21"      

// Topics - Must match the topics configured in your Web App
#define TOPIC_FEATURES   "battery/features"
#define TOPIC_PREDICTION "battery/prediction"

void mqtt_client_begin(InferenceCallback cb);
void mqtt_client_handle();
bool mqtt_client_is_connected();

#endif