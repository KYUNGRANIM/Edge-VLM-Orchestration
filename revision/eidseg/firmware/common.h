#pragma once
#include <Arduino.h>
#include <SD.h>
#include <vector>

String basename_of(const String& path);
bool ensure_dir(const char* dir_path);

bool scan_recursive(const String& root, std::vector<String>& out_files);

bool read_image_to_buffer(const String& filepath,
                          uint8_t* buf, size_t buf_cap,
                          size_t& out_size,
                          unsigned long& out_read_ms,
                          String& out_err);

int load_ckpt(const String& ckpt_path);
bool save_ckpt_overwrite(const String& ckpt_path, int next_idx);

// NVS-based checkpoint (ESP32 internal flash, SD card 무관)
int load_ckpt_nvs(const char* device_id);
bool save_ckpt_nvs(const char* device_id, int next_idx);

void normalize_inline(String& s, int maxLen);

bool ntp_sync_kst();
String now_kst_iso_ms();
