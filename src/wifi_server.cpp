#include "wifi_server.h"

#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>

// ================= INTERNAL STATE =================
static WebServer server(WIFI_HTTP_PORT);

static InferenceCallback inference_cb = nullptr;

static InferenceResult latest_result;
static volatile bool result_available_flag = false;

static unsigned long total_samples = 0;
static unsigned long total_errors  = 0;


static StaticJsonDocument<1024> json_doc;

// ================= HELPERS =================

static String build_result_json(const InferenceResult& r, const char* error) {
    StaticJsonDocument<512> doc;

    if (error != nullptr) {
        doc["status"] = "error";
        doc["error"] = error;
        doc["prediction"] = nullptr;
        doc["confidence"] = nullptr;
        doc["latency_ms"] = nullptr;
        doc["fault"] = false;
        doc["fault_count"] = 0;
        doc["window_ready"] = false;
    } else {
        doc["status"] = "ok";
        doc["window_ready"] = r.window_ready;
        doc["fault_count"] = r.fault_count;

        if (r.window_ready && r.ok) {
            doc["prediction"] = r.class_name;
            doc["confidence"] = serialized(String(r.confidence, 1));
            doc["latency_ms"] = serialized(String(r.latency_ms, 2));
            doc["fault"] = r.fault;
        } else {
           
            doc["prediction"] = nullptr;
            doc["confidence"] = nullptr;
            doc["latency_ms"] = nullptr;
            doc["fault"] = false;
        }
    }

    String out;
    serializeJson(doc, out);
    return out;
}

// ================= HANDLERS =================

// Handler for OPTIONS preflight request
static void handle_options_data() {
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.sendHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
    server.send(204); // 204 No Content
}

// POST /data
// Body: {"features":[f0, f1, ..., f30]}
static void handle_post_data() {

    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.sendHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    server.sendHeader("Access-Control-Allow-Headers", "Content-Type");
    if (!server.hasArg("plain")) {
        total_errors++;
        server.send(400, "application/json",
                     build_result_json(InferenceResult{}, "missing body"));
        return;
    }

    if (!server.hasArg("plain")) {
        total_errors++;
        server.send(400, "application/json",
                     build_result_json(InferenceResult{}, "missing body"));
        return;
    }

    const String& body = server.arg("plain");

    json_doc.clear();
    DeserializationError err = deserializeJson(json_doc, body);
    if (err) {
        total_errors++;
        server.send(400, "application/json",
                     build_result_json(InferenceResult{}, "invalid json"));
        return;
    }

    JsonArray arr = json_doc["features"].as<JsonArray>();
    if (arr.isNull() || arr.size() != WIFI_N_FEATURES) {
        total_errors++;
        server.send(400, "application/json",
                     build_result_json(InferenceResult{},
                         "features must be an array of 31 numbers"));
        return;
    }

    float parsed[WIFI_N_FEATURES];
    size_t i = 0;
    for (JsonVariant v : arr) {
        if (!v.is<float>() && !v.is<int>()) {
            total_errors++;
            server.send(400, "application/json",
                         build_result_json(InferenceResult{},
                             "non-numeric feature value"));
            return;
        }
        parsed[i++] = v.as<float>();
    }

    total_samples++;

    if (inference_cb == nullptr) {
        // No callback registered — should not happen in normal operation,
        // but fail loudly rather than silently dropping the sample.
        total_errors++;
        server.send(500, "application/json",
                     build_result_json(InferenceResult{},
                         "inference callback not registered"));
        return;
    }

    InferenceResult result = {};
    result.class_index = -1;
    result.class_name = nullptr;
    inference_cb(parsed, &result);

    // Stash a copy so main.cpp's loop() can refresh the OLED/buzzer state
    latest_result = result;
    result_available_flag = true;

    int http_status = result.ok ? 200 : 500;
    server.send(http_status, "application/json",
                 build_result_json(result, result.ok ? nullptr : "inference failed"));
}

static void handle_not_found() {
    total_errors++;
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.send(404, "application/json",
                 build_result_json(InferenceResult{}, "not found"));
}

// ================= PUBLIC API =================

void wifi_server_set_inference_callback(InferenceCallback cb) {
    inference_cb = cb;
}

void wifi_server_begin() {
    WiFi.mode(WIFI_AP);
    WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASSWORD, WIFI_AP_CHANNEL,
                WIFI_AP_HIDDEN, WIFI_AP_MAX_CONN);

    Serial.print("AP started. SSID: ");
    Serial.print(WIFI_AP_SSID);
    Serial.print("  IP: ");
    Serial.println(WiFi.softAPIP());

    server.on(WIFI_DATA_PATH, HTTP_POST, handle_post_data);
    server.on(WIFI_DATA_PATH, HTTP_OPTIONS, handle_options_data); // <-- ADD THIS LINE
    server.onNotFound(handle_not_found);
    server.begin();

    Serial.print("HTTP server listening on port ");
    Serial.println(WIFI_HTTP_PORT);
}

void wifi_server_handle() {
    server.handleClient();
}

unsigned long wifi_server_total_samples() {
    return total_samples;
}

unsigned long wifi_server_total_errors() {
    return total_errors;
}

bool wifi_server_result_available() {
    if (result_available_flag) {
        result_available_flag = false;
        return true;
    }
    return false;
}

void wifi_server_get_latest_result(InferenceResult* out) {
    *out = latest_result;
}