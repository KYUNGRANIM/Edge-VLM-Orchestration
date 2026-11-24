#include "common.h"
#include <WiFi.h>
#include <time.h>
#include <Preferences.h>

String basename_of(const String& path) {
  int p = path.lastIndexOf('/');
  if (p < 0) return path;
  return path.substring(p + 1);
}

bool ensure_dir(const char* dir_path) {
  if (!dir_path || dir_path[0] == '\0') return false;
  if (strcmp(dir_path, "/") == 0) return true;

  if (SD.exists(dir_path)) {
    File d = SD.open(dir_path);
    bool ok = d && d.isDirectory();
    if (d) d.close();
    return ok;
  }
  return SD.mkdir(dir_path);
}

static void scan_dir(const String& dir, std::vector<String>& out_files) {
  File root = SD.open(dir);
  if (!root || !root.isDirectory()) { if(root) root.close(); return; }

  while (true) {
    // getNextFileName only reads the directory entry. openNextFile also stat()s and fopen()s
    // every file, each walking the path from the root again: minutes per boot at 2,931 images.
    bool isDir = false;
    String full = root.getNextFileName(&isDir);
    if (full.length() == 0) break;

    // skip hidden / mac junk
    String base = basename_of(full);
    if (base.startsWith("._") || base == ".DS_Store" || base.startsWith(".")) continue;

    if (isDir) {
      scan_dir(full, out_files);
    } else {
      String low = full; low.toLowerCase();
      if (low.endsWith(".jpg") || low.endsWith(".jpeg")) out_files.push_back(full);
    }
  }
  root.close();
}

bool scan_recursive(const String& root, std::vector<String>& out_files) {
  out_files.clear();
  scan_dir(root, out_files);
  return out_files.size() > 0;
}

bool read_image_to_buffer(const String& filepath,
                          uint8_t* buf, size_t buf_cap,
                          size_t& out_size,
                          unsigned long& out_read_ms,
                          String& out_err) {
  out_err = "";
  out_size = 0;

  unsigned long t0 = millis();
  File f = SD.open(filepath, FILE_READ);
  if (!f) { out_err = "open_fail"; return false; }

  size_t sz = f.size();
  if (sz == 0 || sz > buf_cap) {
    f.close();
    out_err = "size_bad";
    return false;
  }

  size_t n = f.read(buf, sz);
  f.close();

  out_read_ms = millis() - t0;

  if (n != sz) { out_err = "read_short"; return false; }
  if (sz < 4)  { out_err = "too_small"; return false; }
  if (!(buf[0] == 0xFF && buf[1] == 0xD8)) { out_err = "jpg_magic_bad"; return false; }

  out_size = sz;
  return true;
}

int load_ckpt(const String& ckpt_path) {
  const char* p = ckpt_path.c_str();
  if (!SD.exists(p)) {
    Serial.printf("[CKPT] file not found: %s\n", p);
    return 0;
  }
  File f = SD.open(p, FILE_READ);
  if (!f) {
    Serial.printf("[CKPT] open fail: %s\n", p);
    return 0;
  }
  String s = f.readString();
  f.close();

  s.replace("\r", "");
  s.trim();
  if (s.length() == 0) return 0;

  // FILE_WRITE can append on some SD stacks. Read the last non-empty line.
  int end = s.length() - 1;
  while (end >= 0 && (s[end] == '\n' || s[end] == ' ' || s[end] == '\t')) end--;
  if (end < 0) return 0;

  int start = end;
  while (start >= 0 && s[start] != '\n') start--;
  String last = s.substring(start + 1, end + 1);
  last.trim();
  if (last.length() == 0) return 0;
  int val = last.toInt();
  Serial.printf("[CKPT] loaded %s -> %d\n", p, val);
  return val;
}

bool save_ckpt_overwrite(const String& ckpt_path, int next_idx) {
  const char* p = ckpt_path.c_str();
  if (SD.exists(p)) SD.remove(p);
  delay(20);  // FAT sync 대기
  File f = SD.open(p, FILE_WRITE);
  if (!f) {
    Serial.printf("[CKPT] save open fail: %s\n", p);
    return false;
  }
  size_t written = f.print(String(next_idx));
  f.print("\n");
  f.flush();
  f.close();
  delay(30);  // FAT sync 대기 후 verify

  // verify: 2회 시도
  int verify = -1;
  for (int attempt = 0; attempt < 2; attempt++) {
    verify = load_ckpt(ckpt_path);
    if (verify == next_idx) break;
    delay(50);
  }
  if (verify != next_idx) {
    Serial.printf("[CKPT] save verify FAIL: wrote %d, read %d (written=%u)\n",
                  next_idx, verify, (unsigned)written);
    // 쓰기 자체가 성공했으면 (written > 0) true 반환 — FAT 캐시 지연일 가능성 높음
    return (written > 0);
  }
  return true;
}

void normalize_inline(String& s, int maxLen) {
  s.replace("\r", " ");
  s.replace("\n", " ");
  s.trim();
  while (s.indexOf("  ") >= 0) s.replace("  ", " ");
  if (maxLen > 0 && (int)s.length() > maxLen) {
    s = s.substring(0, maxLen);
    s.trim();
  }
}

bool ntp_sync_kst() {
  configTime(9 * 3600, 0, "pool.ntp.org", "time.nist.gov");
  time_t now = time(nullptr);
  unsigned long t0 = millis();
  while (now < 1700000000) {
    delay(200);
    now = time(nullptr);
    if (millis() - t0 > 8000) return false;
  }
  return true;
}

String now_kst_iso_ms() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);

  time_t t = tv.tv_sec;
  struct tm tm_kst;
  localtime_r(&t, &tm_kst);

  char buf[40];
  int ms = (int)(tv.tv_usec / 1000);

  snprintf(buf, sizeof(buf),
           "%04d-%02d-%02dT%02d:%02d:%02d.%03d+09:00",
           tm_kst.tm_year + 1900, tm_kst.tm_mon + 1, tm_kst.tm_mday,
           tm_kst.tm_hour, tm_kst.tm_min, tm_kst.tm_sec, ms);

  return String(buf);
}

// ============================================================
//  NVS-based checkpoint (ESP32 internal flash)
// ============================================================
static Preferences _nvs_prefs;

int load_ckpt_nvs(const char* device_id) {
  _nvs_prefs.begin(device_id, true);  // read-only
  int val = _nvs_prefs.getInt("ckpt", 0);
  _nvs_prefs.end();
  Serial.printf("[CKPT-NVS] load %s -> %d\n", device_id, val);
  return val;
}

bool save_ckpt_nvs(const char* device_id, int next_idx) {
  _nvs_prefs.begin(device_id, false);  // read-write
  _nvs_prefs.putInt("ckpt", next_idx);
  _nvs_prefs.end();
  // verify
  int check = load_ckpt_nvs(device_id);
  if (check != next_idx) {
    Serial.printf("[CKPT-NVS] verify FAIL: wrote %d, read %d\n", next_idx, check);
    return false;
  }
  return true;
}
