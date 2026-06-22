#include "wifi_server.h"
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

static WiFiClientSecure secureClient;
static WiFiClient regularClient;
static PubSubClient mqttClient;

static InferenceCallback inference_cb = nullptr;
static unsigned long last_reconnect_attempt = 0;

// ================= INCOMING MESSAGE HANDLER =================
static void mqtt_callback(char* topic, byte* payload, unsigned int length) {
    Serial.print("Message arrived on [");
    Serial.print(topic);
    Serial.println("]");

    // Parse incoming JSON payload: {"features": [f0, f1, ..., f30]}
    StaticJsonDocument<1024> doc;
    DeserializationError error = deserializeJson(doc, payload, length);
    if (error) {
        Serial.print("JSON deserialization failed: ");
        Serial.println(error.c_str());
        return;
    }

    JsonArray arr = doc["features"].as<JsonArray>();
    if (arr.isNull() || arr.size() != N_FEATURES) {
        Serial.println("Invalid features array in payload");
        return;
    }

    float parsed_features[N_FEATURES];
    for (size_t i = 0; i < N_FEATURES; i++) {
        parsed_features[i] = arr[i].as<float>();
    }

    // Call TF Lite inference engine callback in main.cpp
    InferenceResult result = {};
    result.class_index = -1;
    result.class_name = nullptr;
    
    if (inference_cb != nullptr) {
        inference_cb(parsed_features, &result);
    }

    // Publish the classification results: {"status":"ok", "window_ready": true, ...}
    StaticJsonDocument<512> responseDoc;
    responseDoc["status"] = result.ok ? "ok" : "error";
    responseDoc["window_ready"] = result.window_ready;
    responseDoc["fault_count"] = result.fault_count;

    if (result.window_ready && result.ok) {
        responseDoc["prediction"] = result.class_name;
        responseDoc["confidence"] = serialized(String(result.confidence, 1));
        responseDoc["latency_ms"] = serialized(String(result.latency_ms, 2));
        responseDoc["fault"] = result.fault;
    } else {
        responseDoc["prediction"] = nullptr;
        responseDoc["confidence"] = nullptr;
        responseDoc["latency_ms"] = nullptr;
        responseDoc["fault"] = false;
    }

    String responseString;
    serializeJson(responseDoc, responseString);
    
    mqttClient.publish(TOPIC_PREDICTION, responseString.c_str());
    Serial.print("Published prediction: ");
    Serial.println(responseString);
}

// ================= RECONNECTION MANAGEMENT =================
static bool connect_mqtt() {
    String clientId = "ESP32-BMS-" + String(random(0xffff), HEX);
    Serial.print("Attempting MQTT connection to ");
    Serial.print(MQTT_BROKER);
    Serial.print("...");
    
    bool connected = false;
    if (String(MQTT_USER).length() > 0) {
        connected = mqttClient.connect(clientId.c_str(), MQTT_USER, MQTT_PASS);
    } else {
        connected = mqttClient.connect(clientId.c_str());
    }

    if (connected) {
        Serial.println("connected ✅");
        mqttClient.subscribe(TOPIC_FEATURES);
        Serial.print("Subscribed to: ");
        Serial.println(TOPIC_FEATURES);
        return true;
    } else {
        Serial.print("failed, rc=");
        Serial.print(mqttClient.state());
        Serial.println(" ❌ (Retrying in 5 seconds)");
        return false;
    }
}

// ================= PUBLIC API =================

void mqtt_client_begin(InferenceCallback cb) {
    inference_cb = cb;

    // Connect to WiFi Station
    Serial.println();
    Serial.print("Connecting to WiFi network: ");
    Serial.println(WIFI_SSID);
    
    WiFi.mode(WIFI_STA);
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("connected! ✅");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());

    // Configure SSL/TLS Security or Standard socket depending on the port
    if (MQTT_PORT == 8883) {
        // Skips CA certificate validation to make HiveMQ/EMQX cloud setup easy
        secureClient.setInsecure(); 
        mqttClient.setClient(secureClient);
    } else {
        mqttClient.setClient(regularClient);
    }

    mqttClient.setServer(MQTT_BROKER, MQTT_PORT);
    mqttClient.setCallback(mqtt_callback);
    
    // Increase buffer size to handle the large 31-feature input string
    mqttClient.setBufferSize(2048); 
}

void mqtt_client_handle() {
    // If WiFi drops, block loop actions until reconnect
    if (WiFi.status() != WL_CONNECTED) {
        return;
    }

    // Non-blocking MQTT reconnect handler
    if (!mqttClient.connected()) {
        unsigned long now = millis();
        if (now - last_reconnect_attempt > 5000) {
            last_reconnect_attempt = now;
            if (connect_mqtt()) {
                last_reconnect_attempt = 0;
            }
        }
    } else {
        mqttClient.loop();
    }
}

bool mqtt_client_is_connected() {
    return mqttClient.connected();
}