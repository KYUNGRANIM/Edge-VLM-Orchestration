/*
 * tier1.cpp  --  M5->Pi Oracle Dataset Generation (v2-multipart)
 *
 * Architecture:
 * - No MQTT. Each M5 (A/B/C) independently processes images and
 * POSTs result + image to Pi via HTTP multipart.
 * - Pi merges A+B+C by image_id into merged_oracle.jsonl.
 * - After POST, M5 sends lightweight PATCH with http_e2e latency.
 *
 * Per-device flow:
 * SD scan -> for each image:
 * read jpg -> VLM infer -> multipart POST (meta+image) -> PATCH e2e -> checkpoint
 */

 #include "tier1.h"
 #include "config.h"
 #include "prompts.h"
 #include "common.h"
 
 #include <M5Unified.h>
 #include <M5ModuleLLM.h>
 #include <ArduinoJson.h>
 
 #include <WiFi.h>
 #include <SPI.h>
 #include <SD.h>
 #include <JPEGDEC.h>
 #include <JPEGENC.h>

 #include <vector>
 #include <algorithm>
 #include <esp_task_wdt.h>
 #include "soc/rtc_cntl_reg.h"
 #include <nvs_flash.h>
 
 // ============================================================
 //  SD pins (CoreS3)
 // ============================================================
 static const int SD_CS   = 4;
 static const int SD_SCK  = 36;
 static const int SD_MISO = 35;
 static const int SD_MOSI = 37;
 
 // ============================================================
 //  Globals
 // ============================================================
 static const char* DEVICE_ID = DEVICE_NAME;
 
 static uint8_t* jpg_buffer = nullptr;
 static uint8_t* vlm_jpg_buffer = nullptr;   // resized JPEG for VLM inference

 static M5ModuleLLM module_llm;
 static String vlm_work_id;
 
 static std::vector<String> g_files;
 static std::vector<String> g_retry_targets;
 static bool g_done = false;
 
 static int g_consecutive_fails = 0;
 static int g_infer_fail_streak = 0;
 static int g_net_fail_streak = 0;
static int g_recover_fail_streak = 0;
static int g_recover_failed_images = 0;
static unsigned long g_next_recover_allowed_ms = 0;
static int g_precheck_fail_streak = 0;
static int g_timeout_streak_global = 0;
static unsigned long g_last_push_ms = 0;
static unsigned long g_last_push_kbps = 0;
static unsigned long g_last_progress_ms = 0;
static String g_last_vlm_fail_reason = "";
static String g_last_a_s1_key = "";
static int g_last_a_s1_repeat = 0;
static String g_last_b_key = "";
static int g_last_b_repeat = 0;
static String g_last_c_key = "";
static int g_last_c_repeat = 0;
static bool g_wdt_armed = false;
static bool g_restart_requested = false;
static String g_restart_reason = "";
static bool g_sd_write_enabled = true;
static String g_out_dir_path = String(OUT_DIR);
static String g_failed_ids_path = String(FAILED_IDS_PATH);
static String g_retry_only_flag_path = String(RETRY_ONLY_FLAG_PATH);
static const int WIFI_RECYCLE_EVERY_N = 180;
static const int DUP_HISTORY_SIZE = 32;
static const char* FW_TAG = "tier1-latency-split-v25";

static inline void wdt_kick() {
  if (g_wdt_armed) {
    esp_task_wdt_reset();
  }
}

static inline void mark_recover_fail_cooldown() {
  g_recover_fail_streak++;
  unsigned long backoff = (unsigned long)RECOVER_FAIL_COOLDOWN_MS * (unsigned long)g_recover_fail_streak;
  if (backoff > 180000UL) backoff = 180000UL;
  g_next_recover_allowed_ms = millis() + backoff;
  Serial.printf("[RECOVER] cooldown %lus (streak=%d)\n", backoff / 1000UL, g_recover_fail_streak);
}

static inline void request_board_restart(const String& reason) {
  g_restart_requested = true;
  g_restart_reason = reason;
  Serial.printf("[SYSTEM] restart requested: %s\n", reason.c_str());
}
 static const int DUP_A_S1_HISTORY = 24;
 static String g_hist_a[DUP_HISTORY_SIZE];
 static int    g_hist_a_idx = 0;
 static String g_hist_a_s1[DUP_A_S1_HISTORY];
 static int    g_hist_a_s1_idx = 0;
 static String g_hist_b[DUP_HISTORY_SIZE];
 static int    g_hist_b_idx = 0;
 static String g_hist_c[DUP_HISTORY_SIZE];
 static int    g_hist_c_idx = 0;
 
 // forward declarations for helpers used before definition
 static int split_sentences(const String& s, String out[], int max_out);
 static void postprocess_text_output(String& s, const char* role);
static void sanitize_b_text(String& s);
static void sanitize_c_text(String& s);
static void refine_c_text_priority(String& text, const String& raw, const String& cat_hint);
static void ensure_c_human_impact(String& text, const String& cat_hint, int sev_hint);
static bool enforce_two_sentences_strict(String& text);
static bool vlm_query(const uint8_t* img, size_t img_size,
                      const char* prompt,
                      const char* role,
                      String& out_text,
                      unsigned long& out_ms,
                      unsigned long& out_uart_ms,
                      unsigned long& out_infer_ms);
 static bool recover_module(const uint8_t* probe_img = nullptr, size_t probe_size = 0);
 static bool llm_health_check();
 static bool llm_quick_infer_check(const uint8_t* probe_img, size_t probe_size);
 static void append_failed_id(const String& image_id, const char* role, const char* reason);
 static bool load_retry_targets(std::vector<String>& out_ids);
 static void filter_files_by_retry_targets();
 static int compute_vlm_wait_ms(size_t img_size, const char* role);
 static bool ensure_uart_921600();
 static bool salvage_vlm_text(const String& raw, const char* role, String& out);
 static bool power_cycle_llm_bus();
 static bool force_core_s3_bus_power(bool enable);
static bool set_ext_output_with_fallback(bool enable);
static String join_sd_path(const String& dir_path, const String& filename);
static bool probe_sd_write_capability(const String& test_path);
 
 // ============================================================
 //  Text helpers
 // ============================================================
static void append_chunk(String& buf, const String& chunk) {
  if (chunk.length() == 0) return;
  buf += chunk;
}

static String join_sd_path(const String& dir_path, const String& filename) {
  String d = dir_path;
  if (d.length() == 0 || d == "/") {
    return String("/") + filename;
  }
  if (!d.startsWith("/")) d = "/" + d;
  if (d.endsWith("/")) return d + filename;
  return d + "/" + filename;
}

static bool probe_sd_write_capability(const String& test_path) {
  if (test_path.length() == 0) return false;
  if (SD.exists(test_path.c_str())) SD.remove(test_path.c_str());
  File f = SD.open(test_path.c_str(), FILE_WRITE);
  if (!f) return false;
  f.print("probe\n");
  f.flush();
  f.close();
  bool ok = SD.exists(test_path.c_str());
  SD.remove(test_path.c_str());
  return ok;
}
 
 static int count_words(const String& s) {
   int cnt = 0;
   bool in_word = false;
   for (int i = 0; i < (int)s.length(); i++) {
     char ch = s[i];
     bool is_word = ((ch >= 'a' && ch <= 'z') ||
                     (ch >= 'A' && ch <= 'Z') ||
                     (ch >= '0' && ch <= '9'));
     if (is_word && !in_word) {
       cnt++;
       in_word = true;
     } else if (!is_word) {
       in_word = false;
     }
   }
   return cnt;
 }
 
 static int count_substr(const String& s, const char* needle) {
   int cnt = 0;
   int pos = 0;
   int step = String(needle).length();
   if (step <= 0) return 0;
   while (true) {
     int p = s.indexOf(needle, pos);
     if (p < 0) break;
     cnt++;
     pos = p + step;
   }
   return cnt;
 }
 
 static int count_char(const String& s, char ch) {
   int n = 0;
   for (int i = 0; i < (int)s.length(); i++) {
     if (s[i] == ch) n++;
   }
   return n;
 }
 
 static bool has_non_ascii(const String& s) {
   for (int i = 0; i < (int)s.length(); i++) {
     if ((uint8_t)s[i] > 127) return true;
   }
   return false;
 }
 
 static bool has_scene_anchor(const String& s) {
   String low = s;
   low.toLowerCase();
   const char* anchors[] = {
     "landslide", "debris", "rock", "soil", "slope", "cliff", "road", "highway",
     "track", "rail", "mud", "collapse", "erosion", "shore", "coast", "bridge",
     "route", "path", "hill", "lane", "terrain"
   };
   for (size_t i = 0; i < sizeof(anchors) / sizeof(anchors[0]); i++) {
     if (low.indexOf(anchors[i]) >= 0) return true;
   }
   return false;
 }
 
 static bool has_any_token_ci(const String& s_low, const char* const toks[], size_t n) {
   for (size_t i = 0; i < n; i++) {
     if (s_low.indexOf(toks[i]) >= 0) return true;
   }
   return false;
 }
 
 static bool b_has_anchor_token(const String& s) {
   String low = s;
   low.toLowerCase();
   const char* const toks[] = {
     "road", "highway", "lane", "track", "rail", "slope", "cliff", "hillside",
     "bridge", "shoreline", "coast", "beach", "debris", "mud", "rock", "boulder",
     "barrier", "vehicle", "building", "house"
   };
   return has_any_token_ci(low, toks, sizeof(toks) / sizeof(toks[0]));
 }
 
 static bool b_has_damage_token(const String& s) {
   String low = s;
   low.toLowerCase();
   const char* const toks[] = {
     "landslide", "rockfall", "debris", "damag", "collapse", "collapsed", "failure",
     "failed", "erosion", "eroded", "crack", "cracked", "blocked", "obstruct",
     "buried", "fallen", "mud", "disruption", "displaced"
   };
   return has_any_token_ci(low, toks, sizeof(toks) / sizeof(toks[0]));
 }
 
 static bool b_has_speculation_token(const String& s) {
   String low = s;
   low.toLowerCase();
   const char* const toks[] = {
     " may ", " could ", " might ", " likely ", " appears ", " seem", " suggest",
     "possibly", "probably", "maybe"
   };
   return has_any_token_ci(low, toks, sizeof(toks) / sizeof(toks[0]));
 }
 
 static bool b_has_scenic_token(const String& s) {
   String low = s;
   low.toLowerCase();
   const char* const toks[] = {
     "clear sky", "cloudy sky", "beautiful", "scenic", "picturesque", "lush",
     "volcanic landscape", "landscape with", "weather-beaten", "green hills"
   };
   return has_any_token_ci(low, toks, sizeof(toks) / sizeof(toks[0]));
 }
 
 static bool c_has_anchor_token(const String& s) {
   String low = s;
   low.toLowerCase();
   const char* const toks[] = {
     "road", "highway", "lane", "track", "rail", "slope", "cliff", "hillside",
     "debris", "mud", "rock", "boulder", "barrier", "vehicle", "shoreline",
     "coast", "beach", "pavement", "bridge", "tunnel"
   };
   return has_any_token_ci(low, toks, sizeof(toks) / sizeof(toks[0]));
 }
 
 static bool c_has_people_token(const String& s) {
   String low = s;
   low.toLowerCase();
   const char* const toks[] = {
     "people", "person", "pedestrian", "pedestrians", "public", "driver",
     "drivers", "passenger", "passengers", "road users", "travelers",
     "commuters", "nearby users", "nearby people", "traffic users"
   };
   return has_any_token_ci(low, toks, sizeof(toks) / sizeof(toks[0]));
 }
 
static bool c_has_risk_token(const String& s) {
  String low = s;
  low.toLowerCase();
  const char* const toks[] = {
    "risk", "hazard", "danger", "unsafe", "threat", "endanger", "threaten",
    "injury", "injured", "evacuat", "rescue", "trapped",
    "limit", "restrict", "block", "obstruct", "disrupt", "narrow"
  };
  return has_any_token_ci(low, toks, sizeof(toks) / sizeof(toks[0]));
}

static bool has_bad_generic_phrase(const String& s) {
  String low = s;
  low.toLowerCase();
  const char* const toks[] = {
    "affected terrain", "route section", "visible damaged zone", "nearby routes",
    "route continuity", "route corridor", "damaged terrain", "affected route",
    "physical disruption of the visible route",
    "a large section of the cliff has collapsed, exposing the rocky surface",
    "large section of the cliff has collapsed, exposing the rocky surface",
    "section of the cliff has collapsed, exposing the rocky surface"
  };
  return has_any_token_ci(low, toks, sizeof(toks) / sizeof(toks[0]));
}
 
 static bool starts_with_alpha(const String& s) {
   for (int i = 0; i < (int)s.length(); i++) {
     char ch = s[i];
     if (ch == ' ') continue;
     return ((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z'));
   }
   return false;
 }
 
 static bool ends_with_bad_tail(String s) {
   s.trim();
   if (s.length() == 0) return true;
   String low = s;
   low.toLowerCase();
   const char* bad_tails[] = {
     " or", " and", " with", " to", " of", " by", " in", " for",
     " into", " onto", " from", " at", " on", " near", " under", " over", " around", " through",
     " the", " a", " an", " is", " are", " was", " were",
     " indicating", " suggesting", " this", " that", " potential"
   };
   for (size_t i = 0; i < sizeof(bad_tails) / sizeof(bad_tails[0]); i++) {
     if (low.endsWith(bad_tails[i]) ||
         low.endsWith(String(bad_tails[i]) + ".") ||
         low.endsWith(String(bad_tails[i]) + ",") ||
         low.endsWith(String(bad_tails[i]) + ":")) {
       return true;
     }
   }
   return false;
 }
 
 static void trim_sentence_words(String& sent, int max_words) {
   if (max_words <= 0) return;
   String t = sent;
   t.replace("\r", " ");
   t.replace("\n", " ");
   t.trim();
 
   String out = "";
   int words = 0;
   int i = 0;
   int n = (int)t.length();
   while (i < n) {
     while (i < n && t[i] == ' ') i++;
     if (i >= n) break;
     int j = i;
     while (j < n && t[j] != ' ') j++;
     if (words >= max_words) break;
     String tok = t.substring(i, j);
     if (out.length() > 0) out += " ";
     out += tok;
     words++;
     i = j + 1;
   }
 
  out.trim();
  if (out.length() == 0) return;
  while (ends_with_bad_tail(out)) {
    int cut = out.lastIndexOf(' ');
    if (cut <= 0) {
      out = "";
      break;
    }
     out = out.substring(0, cut);
     out.trim();
     if (out.length() == 0) break;
  }
  if (out.length() == 0) return;
  while (out.length() > 0) {
    char c = out[out.length() - 1];
    if (c == ',' || c == ';' || c == ':' || c == '-') {
      out.remove(out.length() - 1);
      out.trim();
      continue;
    }
    break;
  }
  if (out.length() == 0) return;
  char last = out[out.length() - 1];
  if (last != '.' && last != '!' && last != '?') out += ".";
  sent = out;
}
 
 static bool scan_root_contains(const char* key) {
   String s = String(SCAN_ROOT);
   s.toLowerCase();
   String k = String(key);
   k.toLowerCase();
   return s.indexOf(k) >= 0;
 }
 
static bool text_quality_ok(const String& text, const char* role) {
   String s = text;
   s.trim();
   if (s.length() < 12 || s.length() > 500) return false;
   if (has_non_ascii(s)) return false;
   if (count_words(s) < 5) return false;
 
   String low = s;
   low.toLowerCase();
 
  int wc = count_words(s);
  bool isA = (String(role) == "a");
  if (isA && wc > 50) return false;
  if (!isA && wc > 35) return false;
  if (isA) {
    if (low.indexOf("cat:") >= 0 || low.indexOf("category:") >= 0 ||
        low.indexOf("sev:") >= 0 || low.indexOf("severity:") >= 0 ||
        low.indexOf("conf:") >= 0 || low.indexOf("confidence:") >= 0) {
      return false;
    }
  }
 
   // garbage/hallucination detection (1B model artifacts)
   const char* bad_keys[] = {
     "ch11", "chapter", "introduction", "machine learning",
     "data set", "dataset", "###", "http://", "https://",
     "2020-", "2021-", "2022-", "2023-", "chia -", "lassig",
     "regular expression", "regex", "python", "match a string", "chef",
     "renowned", "chengcheng", "amaciyla", "amacyla", "amac",
     "reevaluation", "evaluate results", "implement changes"
   };
   for (size_t i = 0; i < sizeof(bad_keys) / sizeof(bad_keys[0]); i++) {
     if (low.indexOf(bad_keys[i]) >= 0) return false;
   }
   const char* bad_meta[] = {
     "sentence 1", "sentence 2", "first sentence", "second sentence",
     "output only", "analysis:"
   };
   for (size_t i = 0; i < sizeof(bad_meta) / sizeof(bad_meta[0]); i++) {
     if (low.indexOf(bad_meta[i]) >= 0) return false;
   }
   if (count_substr(low, "chia") >= 3) return false;
   if (low.indexOf("injuries or injuries") >= 0) return false;
   if (low.indexOf("risk of risk") >= 0) return false;
   if (ends_with_bad_tail(s)) return false;
 
   // first non-space must be a letter
   int first_idx = -1;
   for (int i = 0; i < (int)s.length(); i++) {
     if (s[i] != ' ') { first_idx = i; break; }
   }
   if (first_idx < 0) return false;
   char first_ch = s[first_idx];
   if (!((first_ch >= 'A' && first_ch <= 'Z') || (first_ch >= 'a' && first_ch <= 'z'))) {
     return false;
   }
 
  // sentence structure
  String parts[3];
  int cnt = split_sentences(s, parts, 3);
  if (cnt < 1) return false;
 
   String p0 = parts[0];
   p0.trim();
   if (count_words(p0) < 4) return false;
  if (isA && cnt >= 2) {
    String p1 = parts[1];
    p1.trim();
    if (count_words(p1) < 4) return false;
  }

  return true;
}
 
 static bool is_duplicate_recent_a_s1(const String& s1) {
   String t = s1;
   t.trim();
   if (t.length() < 14) return false;
   String key = t.substring(0, std::min(28, (int)t.length()));
   for (int i = 0; i < DUP_A_S1_HISTORY; i++) {
     if (g_hist_a_s1[i].length() == 0) continue;
     String hk = g_hist_a_s1[i].substring(0, std::min(28, (int)g_hist_a_s1[i].length()));
     if (hk == key) return true;
   }
   return false;
 }
 
 static void remember_recent_a_s1(const String& s1) {
   g_hist_a_s1[g_hist_a_s1_idx % DUP_A_S1_HISTORY] = s1;
   g_hist_a_s1_idx++;
 }
 
static void remember_recent_text_ac(const String& text, const char* role) {
  String r = String(role);
  if (r == "a") {
     g_hist_a[g_hist_a_idx % DUP_HISTORY_SIZE] = text;
     g_hist_a_idx++;
     String parts[3];
     int cnt = split_sentences(text, parts, 3);
     if (cnt >= 1) {
       String s1 = parts[0];
       s1.trim();
       if (s1.length() > 0) remember_recent_a_s1(s1);
     }
   } else if (r == "b") {
     g_hist_b[g_hist_b_idx % DUP_HISTORY_SIZE] = text;
     g_hist_b_idx++;
   } else if (r == "c") {
     g_hist_c[g_hist_c_idx % DUP_HISTORY_SIZE] = text;
     g_hist_c_idx++;
  }
}

static String make_dup_key(const String& text) {
  String k = text;
  k.toLowerCase();
  normalize_inline(k, 200);
  k.trim();
  if ((int)k.length() > 64) k = k.substring(0, 64);
  return k;
}

static bool is_duplicate_recent_bc(const String& text, const char* role) {
  String key = make_dup_key(text);
  if (key.length() < 18) return false;

  const String* hist = nullptr;
  if (String(role) == "b") hist = g_hist_b;
  else if (String(role) == "c") hist = g_hist_c;
  else return false;

  int same = 0;
  for (int i = 0; i < DUP_HISTORY_SIZE; i++) {
    if (hist[i].length() == 0) continue;
    if (make_dup_key(hist[i]) == key) same++;
  }
  return (same >= 2);
}

static bool c_has_human_risk_token(const String& text) {
  String low = text;
  low.toLowerCase();
  const char* toks[] = {
    "risk", "danger", "hazard", "unsafe", "injury", "injured", "hurt", "fatal",
    "evacuat", "rescue", "trapped", "bury", "blocked access", "access", "people",
    "pedestrian", "nearby residents", "public safety", "life-threatening"
  };
  for (size_t i = 0; i < sizeof(toks) / sizeof(toks[0]); i++) {
    if (low.indexOf(toks[i]) >= 0) return true;
  }
  return false;
}

static bool a_s1_too_repetitive(const String& s1) {
  String key = make_dup_key(s1);
  if (key.length() < 12) return false;
  if (key == g_last_a_s1_key) g_last_a_s1_repeat++;
  else {
    g_last_a_s1_key = key;
    g_last_a_s1_repeat = 1;
  }
  return (g_last_a_s1_repeat >= 3);
}

static bool bc_text_too_repetitive(const String& text, const char* role) {
  String key = make_dup_key(text);
  if (key.length() < 12) return false;

  bool isB = (String(role) == "b");
  if (isB) {
    if (key == g_last_b_key) g_last_b_repeat++;
    else {
      g_last_b_key = key;
      g_last_b_repeat = 1;
    }
    return (g_last_b_repeat >= 3);
  }

  if (key == g_last_c_key) g_last_c_repeat++;
  else {
    g_last_c_key = key;
    g_last_c_repeat = 1;
  }
  return (g_last_c_repeat >= 3);
}

static bool enforce_two_sentences_strict(String& text) {
  String parts[3];
  int cnt = split_sentences(text, parts, 3);
  if (cnt < 2) return false;
  String s1 = parts[0];
  String s2 = parts[1];
  s1.trim();
  s2.trim();
  if (count_words(s1) < 4 || count_words(s2) < 4) return false;
  trim_sentence_words(s1, 25);
  trim_sentence_words(s2, 25);
  if (!s1.endsWith(".")) s1 += ".";
  if (!s2.endsWith(".")) s2 += ".";
  text = s1 + " " + s2;
  text.trim();
  return true;
}
 
 // ============================================================
 //  Garbage / hallucination detection (1B model artifacts)
 // ============================================================
 static bool is_garbage_text(const String& text) {
   String low = text;
   low.toLowerCase();
   const char* garbage_markers[] = {
     "###", "chapter", "introduction", "machine learning",
     "data set", "dataset", "chakra", "venus", "python",
     "regex", "http://", "https://", "2020-", "chia -",
     "lassig", "evaluate results", "implement changes", "chef",
     "renowned", "chengcheng", "amac",
     "regular expression", "match a string",
     "in this chapter", "we will discuss"
   };
   for (size_t i = 0; i < sizeof(garbage_markers) / sizeof(garbage_markers[0]); i++) {
     if (low.indexOf(garbage_markers[i]) >= 0) return true;
   }
   return false;
 }
 
 // ============================================================
 //  Disaster relevance check
 // ============================================================
 static bool is_disaster_relevant(const String& text) {
   String low = text;
   low.toLowerCase();
   const char* disaster_words[] = {
     "damage", "destroy", "collaps", "block", "crack", "break", "broke",
     "rock", "boulder", "debris", "rubble", "landslide", "mudslide",
     "flood", "erosion", "eroded", "erode",
     "fire", "burn", "blaze",
     "earthquake", "quake",
     "hurricane", "storm", "cyclone", "typhoon",
     "harm", "injur", "danger", "risk", "hazard", "threat", "fatal",
     "rescue", "evacuate", "emergency", "respond",
     "road", "bridge", "building", "structure", "infrastructure",
     "fallen", "scatter", "swept", "buried", "trapped",
     "unstable", "unsafe", "severe", "massive",
     "cliff", "slope", "hillside", "trench", "sinkhole",
     "vehicle", "car", "truck"
   };
   for (size_t i = 0; i < sizeof(disaster_words) / sizeof(disaster_words[0]); i++) {
     if (low.indexOf(disaster_words[i]) >= 0) return true;
   }
   return false;
 }
 
 // ============================================================
 //  Lightweight label stripping for A
 // ============================================================
 static void strip_a_labels(String& s) {
   if (s.startsWith("Yes, ")) s = s.substring(5);
   if (s.startsWith("Yes. ")) s = s.substring(5);
   if (s.startsWith("No, "))  s = s.substring(4);
   if (s.startsWith("No. "))  s = s.substring(4);
   s.replace("In a disaster scene, ", "");
   s.replace("In the disaster scene, ", "");
   s.replace("In this disaster scene, ", "");
   s.replace("In a disaster scene ", "");
   s.replace("In the disaster scene ", "");
   s.replace("In this disaster scene ", "");
   s.replace("What should be done? ", "");
   s.replace("What happened? ", "");
   s.replace("What happened here? ", "");
   s.replace("Immediate measures include ", "");
   s.replace("Immediate measures should be taken to ", "");
   s.replace("Immediate measures: ", "");
   s.replace("Immediate action includes ", "");
   s.replace("Immediate action is needed to ", "");
   s.replace("Immediate action should be taken to ", "");
   s.replace("Immediate action: ", "");
   s.replace("Immediate response actions include ", "");
   s.replace("Immediate response action: ", "");
   s.replace("Response actions include ", "");
   s.replace("The response action needed is ", "");
   s.replace("The response actions needed are ", "");
   s.replace("The life safety threat in this scene is ", "");
   s.replace("The life safety threat is ", "");
   s.replace("The life safety threat here is ", "");
   s.replace("a significant life safety threat, ", "");
   s.replace("a life safety threat, ", "");
   s.replace("significant life safety threat", "significant danger");
   s.replace("life safety threat", "danger");
   s.replace("The presence of ", "");
   s.replace("Sentence 1: ", "");
   s.replace("Sentence 2: ", "");
   s.replace("Sentence 1:", "");
   s.replace("Sentence 2:", "");
   s.replace("Sentence1: ", "");
   s.replace("Sentence2: ", "");
   s.replace("First: ", "");
   s.replace("Second: ", "");
   s.replace("First sentence: ", "");
   s.replace("Second sentence: ", "");
   s.replace("Overall Damage Situation: ", "");
   s.replace("Overall damage situation: ", "");
   s.replace("overall damage situation: ", "");
   s.replace("The overall damage situation appears ", "");
   s.replace("The overall damage situation is ", "");
   s.replace("- Overall Damage Situation: ", "");
   s.replace("The image shows ", "");
   s.replace("the image shows ", "");
   s.replace("This image shows ", "");
   s.replace("this image shows ", "");
   s.replace("The image depicts ", "");
   s.replace("This image depicts ", "");
   s.replace("**", "");
   s.replace("__", "");
   s.replace("You see. ", "");
   s.replace("You see, ", "");
   s.replace("You see ", "");
   s.replace("you see. ", "");
   s.replace("you see, ", "");
   s.replace("you see ", "");
   s.replace("I see ", "");
   s.replace("I can see ", "");
   s.replace("Describe the key visible damage.", "");
   s.replace("Describe the key visible damage", "");
   s.replace("State one response action for that damage.", "");
   s.replace("State one response action for that damage", "");
   s.replace("State the immediate needs of responders, such as ", "Responders should ");
   s.replace("State the immediate needs of responders", "");
   s.replace("State responders should ", "");
   s.replace("State responders ", "");
   s.replace("Describe the damage ", "");
   s.replace("Recommend a response action ", "");
   s.replace("Responders should ", "");
   s.replace("responders should ", "");
   s.replace("Emergency responders should ", "");
   s.replace("emergency responders should ", "");
   s.replace("first sentence: ", "");
   s.replace("second sentence: ", "");
   s.replace("Monitoring: ", "");
   s.replace("monitoring: ", "");
   s.replace("Emergency action: ", "");
   s.replace("emergency action: ", "");
   s.replace("Emergency Action: ", "");
   s.replace("Emergency action needed: ", "");
   s.replace("Emergency Action Needed: ", "");
   s.replace("emergency action needed: ", "");
   s.replace("Damage: ", "");
   s.replace("damage: ", "");
   s.replace("Action: ", "");
   s.replace("action: ", "");
   s.replace("Response: ", "");
   s.replace("response: ", "");
   if (s.startsWith("1. ")) s = s.substring(3);
   if (s.indexOf(" 2. ") >= 0) s.replace(" 2. ", " ");
   s.trim();
 }
 
 // ============================================================
 //  Keyword-based classification
 // ============================================================
static void classify_from_text(const String& text, String& cat, int& sev) {
  String low = text;
  low.toLowerCase();

  // ── Category ──
  // 1단계: 프롬프트 힌트로 모델이 직접 쓴 disaster type 단어 체크 (가장 신뢰도 높음)
  bool direct_hit = false;
  if      (low.indexOf("earthquake") >= 0 || low.indexOf("quake") >= 0)
    { cat = "earthquake"; direct_hit = true; }
  else if (low.indexOf("hurricane") >= 0 || low.indexOf("cyclone") >= 0 ||
           low.indexOf("typhoon") >= 0  || low.indexOf("tornado") >= 0)
    { cat = "hurricane";  direct_hit = true; }
  else if (low.indexOf("landslide") >= 0 || low.indexOf("mudslide") >= 0 ||
           low.indexOf("rockslide") >= 0 || low.indexOf("rockfall") >= 0)
    { cat = "landslide";  direct_hit = true; }
  else if (low.indexOf("flood") >= 0 || low.indexOf("inundat") >= 0)
    { cat = "flood";      direct_hit = true; }
  else if (low.indexOf("fire") >= 0 || low.indexOf("wildfire") >= 0 ||
           low.indexOf("blaze") >= 0 || low.indexOf("flame") >= 0)
    { cat = "fire";       direct_hit = true; }

  // 1.5단계: 모순 감지 — direct_hit이 있어도 시각 서술과 충돌하면 재분류
  // 핵심: 모델이 "landslide"를 썼지만 실제 묘사가 구조물 붕괴면 earthquake로 교정
  if (direct_hit) {
    // landslide → earthquake 교정
    // "landslide"를 썼는데 사면/토사 증거가 없고 구조물 붕괴 증거만 있으면
    if (cat == "landslide") {
      bool slope_evidence =
        low.indexOf("slope")       >= 0 || low.indexOf("hillside")   >= 0 ||
        low.indexOf("mud")         >= 0 || low.indexOf("soil")        >= 0 ||
        low.indexOf("debris flow") >= 0 || low.indexOf("mudflow")     >= 0 ||
        low.indexOf("rockfall")    >= 0 || low.indexOf("erosion")     >= 0 ||
        low.indexOf("cliff face")  >= 0 || low.indexOf("embankment")  >= 0;
      bool eq_evidence =
        low.indexOf("concrete")    >= 0 || low.indexOf("structural")  >= 0 ||
        low.indexOf("rubble")      >= 0 || low.indexOf("crumbled")    >= 0 ||
        low.indexOf("earthquake")  >= 0 || low.indexOf("aftershock")  >= 0 ||
        low.indexOf("collapsed building") >= 0 || low.indexOf("cracked wall") >= 0 ||
        low.indexOf("collapsed structure") >= 0;
      if (!slope_evidence && eq_evidence) {
        cat = "earthquake";
      }
    }

  }

  // 2단계: 직접 단어가 없으면 간접 신호 scoring으로 폴백
  if (!direct_hit) {
    // "burn" 계열 (smoke 등 fire 문맥일 때만)
    if (low.indexOf("burn") >= 0 &&
        (low.indexOf("burning") >= 0 || low.indexOf("burned") >= 0 ||
         low.indexOf("burnt") >= 0   || low.indexOf("smoke") >= 0))
      { cat = "fire"; }
    else {
      int s_eq = 0, s_hurr = 0, s_flood = 0, s_land = 0;

      // ── Earthquake ──
      if (low.indexOf("seismic") >= 0 || low.indexOf("tremor") >= 0 ||
          low.indexOf("aftershock") >= 0)                               s_eq += 4;
      if (low.indexOf("rubble") >= 0 || low.indexOf("crumbled") >= 0)  s_eq += 3;
      if (low.indexOf("fissure") >= 0 || low.indexOf("sinkhole") >= 0) s_eq += 4;
      if (low.indexOf("collapsed building") >= 0 ||
          low.indexOf("collapsed structure") >= 0 ||
          low.indexOf("building collapse") >= 0)                        s_eq += 3;
      if (low.indexOf("structural") >= 0 &&
          (low.indexOf("collapse") >= 0 || low.indexOf("damage") >= 0)) s_eq += 2;
      if (low.indexOf("cracked") >= 0 &&
          (low.indexOf("wall") >= 0 || low.indexOf("road") >= 0 ||
           low.indexOf("foundation") >= 0))                             s_eq += 2;
      if (low.indexOf("concrete") >= 0 &&
          (low.indexOf("collapse") >= 0 || low.indexOf("damage") >= 0 ||
           low.indexOf("crumble") >= 0))                                s_eq += 2;

      // ── Hurricane ──
      if (low.indexOf("storm surge") >= 0 || low.indexOf("storm damage") >= 0 ||
          low.indexOf("tropical storm") >= 0)                           s_hurr += 4;
      if (low.indexOf("wind") >= 0 || low.indexOf("gust") >= 0)        s_hurr += 2;
      if (low.indexOf("uprooted") >= 0)                                 s_hurr += 3;
      if (low.indexOf("fallen tree") >= 0 || low.indexOf("tree fell") >= 0 ||
          low.indexOf("toppled tree") >= 0)                             s_hurr += 3;
      if (low.indexOf("roof") >= 0 && (low.indexOf("blown") >= 0 ||
          low.indexOf("torn") >= 0))                                    s_hurr += 2;
      if (low.indexOf("power line") >= 0 || low.indexOf("powerline") >= 0 ||
          low.indexOf("downed") >= 0)                                   s_hurr += 2;
      if (low.indexOf("coastal") >= 0 &&
          (low.indexOf("damage") >= 0 || low.indexOf("flood") >= 0))   s_hurr += 3;

      // ── Flood ──
      if (low.indexOf("submerge") >= 0 || low.indexOf("waterlog") >= 0) s_flood += 4;
      if (low.indexOf("overflow") >= 0 || low.indexOf("water level") >= 0) s_flood += 3;
      if (low.indexOf("inundat") >= 0)                                  s_flood += 4;
      if (low.indexOf("water") >= 0 &&
          (low.indexOf("rising") >= 0 || low.indexOf("rose") >= 0))    s_flood += 2;

      // ── Landslide ──
      if (low.indexOf("slope") >= 0 || low.indexOf("hillside") >= 0)   s_land += 3;
      if (low.indexOf("cliff") >= 0 || low.indexOf("boulder") >= 0)    s_land += 3;
      if (low.indexOf("erosion") >= 0 || low.indexOf("eroded") >= 0)   s_land += 3;
      if (low.indexOf("debris flow") >= 0 || low.indexOf("mudflow") >= 0) s_land += 4;

      // ── 약한 공통 신호: 다른 강한 신호 없을 때만 반영 ──
      if (low.indexOf("collapsed") >= 0 || low.indexOf("collapse") >= 0)
        if (s_land < 3 && s_flood < 3 && s_hurr < 3) s_eq += 1;
      if (low.indexOf("debris") >= 0)
        if (s_land < 3 && s_flood < 3 && s_hurr < 3) s_eq += 1;

      int best = s_eq;
      if (s_hurr  > best) best = s_hurr;
      if (s_flood > best) best = s_flood;
      if (s_land  > best) best = s_land;

      // 증거 불충분(best < 2)이면 other_disaster로 분류
      if      (best < 2)                                                 cat = "other_disaster";
      else if (s_eq >= s_hurr && s_eq >= s_flood && s_eq >= s_land)     cat = "earthquake";
      else if (s_flood >= s_hurr && s_flood >= s_land)                   cat = "flood";
      else if (s_hurr >= s_land)                                         cat = "hurricane";
      else                                                                cat = "landslide";
    }
  }

  // ── Severity ──
  // 1단계: 프롬프트 힌트로 모델이 직접 쓴 severity 단어 체크
  // "not severe"/"no severe" 부정문은 제외
  bool sev_negated = (low.indexOf("not severe") >= 0 || low.indexOf("no severe") >= 0 ||
                      low.indexOf("not a severe") >= 0);
  if      (low.indexOf("severe") >= 0 && !sev_negated)              sev = 2;
  else if (low.indexOf("moderate") >= 0)                             sev = 1;
  else if (low.indexOf("mild") >= 0 || low.indexOf("low severity") >= 0 ||
           low.indexOf("minor") >= 0 || low.indexOf("low damage") >= 0 ||
           low.indexOf("minimal damage") >= 0 || low.indexOf("little or no") >= 0 ||
           low.indexOf("little to no") >= 0)                         sev = 0;
  else {
    // 2단계: 직접 단어 없으면 기존 scoring 폴백
    int score = 0;
    if (low.indexOf("massive") >= 0 || low.indexOf("devastating") >= 0 ||
        low.indexOf("catastroph") >= 0 || low.indexOf("widespread") >= 0) score += 2;
    if (low.indexOf("casualt") >= 0 || low.indexOf("killed") >= 0 ||
        low.indexOf("fatal") >= 0 || low.indexOf("dead") >= 0 ||
        low.indexOf("death") >= 0)                                      score += 2;
    // "significant/extensive/destruction" → +2 (raised from +1)
    if (low.indexOf("significant") >= 0 || low.indexOf("major") >= 0 ||
        low.indexOf("extensive") >= 0 || low.indexOf("destruct") >= 0) score += 2;
    if (low.indexOf("destroy") >= 0 || low.indexOf("collaps") >= 0 ||
        low.indexOf("rubble") >= 0)                                     score += 1;
    if (low.indexOf("injur") >= 0 || low.indexOf("trapped") >= 0 ||
        low.indexOf("evacuat") >= 0 || low.indexOf("rescue") >= 0)     score += 1;
    if (low.indexOf("block") >= 0 || low.indexOf("scatter") >= 0 ||
        low.indexOf("displace") >= 0)                                   score += 1;
    // severe 임계값 4→3으로 낮춤 (GT의 65%가 severe)
    sev = (score >= 3) ? 2 : (score >= 2) ? 1 : 0;
  }
}

// conf: fine-grained score → quintile-based confidence (1~5)
// - 키워드 카운트 기반 연속 점수 → 실측 분포 quintile 매핑
// - Pi/Jetson의 logprob quintile과 동일한 구조
//
// Quintile boundaries (from pilot_300_run1, N=914):
//   M5_A: q20=0.97, q40=1.20, q60=1.39, q80=1.62
//   M5_B: q20=0.55, q40=0.76, q60=0.93, q80=1.13
//   M5_C: q20=0.40, q40=0.57, q60=0.75, q80=0.98
static float compute_fine_score(const String& text, const String& cat) {
  String low = text;
  low.toLowerCase();
  int wc = count_words(text);

  // ── 1) Category keyword count ──
  int cat_count = 0;
  if (cat == "landslide") {
    const char* kw[] = {"landslide","mudslide","rockslide","rockfall","slope","cliff","erosion","boulder","mudflow"};
    for (size_t i = 0; i < sizeof(kw)/sizeof(kw[0]); i++)
      if (low.indexOf(kw[i]) >= 0) cat_count++;
  } else if (cat == "earthquake") {
    const char* kw[] = {"earthquake","quake","seismic","fissure","sinkhole","rubble","crumbled","tremor"};
    for (size_t i = 0; i < sizeof(kw)/sizeof(kw[0]); i++)
      if (low.indexOf(kw[i]) >= 0) cat_count++;
  } else if (cat == "flood") {
    const char* kw[] = {"flood","submerge","inundat","waterlog","overflow","water level"};
    for (size_t i = 0; i < sizeof(kw)/sizeof(kw[0]); i++)
      if (low.indexOf(kw[i]) >= 0) cat_count++;
  } else if (cat == "hurricane") {
    const char* kw[] = {"hurricane","cyclone","typhoon","storm","wind","gust","surge"};
    for (size_t i = 0; i < sizeof(kw)/sizeof(kw[0]); i++)
      if (low.indexOf(kw[i]) >= 0) cat_count++;
  } else if (cat == "fire") {
    const char* kw[] = {"fire","wildfire","burn","blaze","flame","smoke","ash","scorch"};
    for (size_t i = 0; i < sizeof(kw)/sizeof(kw[0]); i++)
      if (low.indexOf(kw[i]) >= 0) cat_count++;
  }

  // ── 2) Contradiction penalty ──
  bool contradict = false;
  if (cat == "flood" && (low.indexOf("burning") >= 0 || low.indexOf("fire") >= 0 ||
      low.indexOf("blaze") >= 0 || low.indexOf("smoke") >= 0 || low.indexOf("flame") >= 0))
    contradict = true;
  if (cat == "fire" && (low.indexOf("flood") >= 0 || low.indexOf("submerge") >= 0 ||
      low.indexOf("overflow") >= 0 || low.indexOf("waterlog") >= 0))
    contradict = true;
  if (cat == "earthquake" && low.indexOf("flood") >= 0 && low.indexOf("quake") < 0)
    contradict = true;

  // ── 3) Specificity count ──
  const char* specific[] = {
    "blocked","buried","trapped","injured","displaced","evacuate",
    "shatter","toppled","scattered","piled","damaged","severed"
  };
  int spec_count = 0;
  for (size_t i = 0; i < sizeof(specific)/sizeof(specific[0]); i++)
    if (low.indexOf(specific[i]) >= 0) spec_count++;

  // ── 4) Grounding anchor count ──
  const char* anchors[] = {
    "road","bridge","building","highway","vehicle","house",
    "wall","roof","river","forest","town","village","street"
  };
  int anchor_count = 0;
  for (size_t i = 0; i < sizeof(anchors)/sizeof(anchors[0]); i++)
    if (low.indexOf(anchors[i]) >= 0) anchor_count++;

  // ── 5) Hedging count ──
  const char* hedge[] = {
    " may "," might "," could "," possibly "," appears ","appear ",
    " seem","seems ","seemed "," perhaps "," likely "," unlikely "
  };
  int hedge_count = 0;
  for (size_t i = 0; i < sizeof(hedge)/sizeof(hedge[0]); i++)
    if (low.indexOf(hedge[i]) >= 0) hedge_count++;

  // ── Fine-grained score (continuous) ──
  float score = 0.30f * cat_count
              + 0.25f * spec_count
              + 0.25f * anchor_count
              - 0.30f * hedge_count
              - 0.50f * (contradict ? 1 : 0)
              + 0.02f * (wc < 40 ? wc : 40);
  return score;
}

// Quintile boundaries per device role (from pilot_300_run1, N=914)
static int score_to_conf_quintile(float score) {
  // Device-specific quintile boundaries
  #if defined(DEVICE_NAME)
    #define _DEV_IS(x) (strcmp(DEVICE_NAME, x) == 0)
  #else
    #define _DEV_IS(x) false
  #endif

  float q20, q40, q60, q80;
  if (_DEV_IS("M5_B")) {
    q20 = 0.55f; q40 = 0.76f; q60 = 0.93f; q80 = 1.13f;
  } else if (_DEV_IS("M5_C")) {
    q20 = 0.40f; q40 = 0.57f; q60 = 0.75f; q80 = 0.98f;
  } else { // M5_A (default)
    q20 = 0.97f; q40 = 1.20f; q60 = 1.39f; q80 = 1.62f;
  }

  if (score <= q20) return 1;
  if (score <= q40) return 2;
  if (score <= q60) return 3;
  if (score <= q80) return 4;
  return 5;
}

static int compute_conf_from_text(const String& text, const String& cat, int sev) {
  float score = compute_fine_score(text, cat);
  return score_to_conf_quintile(score);
}

static String default_action_for_cat(const String& cat, int sev) {
  String c = cat;
  c.toLowerCase();
  if (c == "landslide") {
    return (sev >= 2)
      ? "Responders should block the slope area and clear critical routes immediately."
      : "Responders should restrict access near unstable slopes and clear debris from paths.";
  }
  if (c == "flood") {
    return (sev >= 2)
      ? "Responders should start evacuation from low-lying zones and cut off hazardous utilities."
      : "Responders should secure flooded paths and guide people to safer ground.";
  }
  if (c == "earthquake") {
    return (sev >= 2)
      ? "Responders should isolate damaged structures and prioritize rapid search and rescue."
      : "Responders should keep people away from cracked structures and inspect key routes.";
  }
  if (c == "fire") {
    return (sev >= 2)
      ? "Responders should establish fire lines and evacuate areas near active flames."
      : "Responders should control fire spread and keep people clear of smoke exposure.";
  }
  if (c == "hurricane") {
    return (sev >= 2)
      ? "Responders should close hazardous roads and evacuate high-risk coastal zones."
      : "Responders should secure damaged infrastructure and monitor rising weather risks.";
  }
  return "Responders should secure the area and perform a rapid field safety assessment.";
}

static void ensure_a_two_sentences(String& text, const String& cat_hint, int sev_hint) {
  String parts[3];
  int cnt = split_sentences(text, parts, 3);
  if (cnt >= 2) {
    String s1 = parts[0];
    String s2 = parts[1];
    s1.trim(); s2.trim();
    trim_sentence_words(s1, 25);
    trim_sentence_words(s2, 25);
    if (!s1.endsWith(".")) s1 += ".";
    if (!s2.endsWith(".")) s2 += ".";
    text = s1 + " " + s2;
    text.trim();
    return;
  }

  if (cnt == 1) {
    String s1 = parts[0];
    s1.trim();
    if (s1.length() == 0) return;
    trim_sentence_words(s1, 25);
    if (!s1.endsWith(".")) s1 += ".";

    String s2 = default_action_for_cat(cat_hint, sev_hint);
    s2.trim();
    trim_sentence_words(s2, 25);
    if (!s2.endsWith(".")) s2 += ".";

    text = s1 + " " + s2;
    text.trim();
  }
}

 // ============================================================
 //  Structured output parser (new prompt format)
 //  Parses: text: / cat: / sev: / conf: fields from VLM output
 // ============================================================
struct VlmParsed {
  String text;
  String cat;
  int sev;
  int conf;
  bool ok_text;
  bool ok_cat;
  bool ok_sev;
  bool ok_conf;
};

 static String normalize_cat_value(const String& raw) {
   String low = raw;
   low.toLowerCase();
   low.trim();
   // Remove surrounding quotes
   if (low.startsWith("\"") || low.startsWith("'")) low = low.substring(1);
   if (low.endsWith("\"") || low.endsWith("'"))     low = low.substring(0, low.length() - 1);
   low.trim();

   // Exact matches
   const char* valid[] = {"earthquake","fire","flood","hurricane","landslide","other_disaster"};
   for (size_t i = 0; i < 6; i++) {
     if (low == valid[i]) return String(valid[i]);
   }
   // Fuzzy matching
   if (low.indexOf("flood") >= 0)                                             return "flood";
   if (low.indexOf("earthquake") >= 0 || low.indexOf("quake") >= 0)           return "earthquake";
   if (low.indexOf("wildfire") >= 0 || low.indexOf("fire") >= 0 ||
       low.indexOf("burn") >= 0 || low.indexOf("blaze") >= 0)                 return "fire";
   if (low.indexOf("landslide") >= 0 || low.indexOf("mudslide") >= 0 ||
       low.indexOf("rockslide") >= 0 || low.indexOf("debris") >= 0 ||
       low.indexOf("slope") >= 0 || low.indexOf("erosion") >= 0)              return "landslide";
   if (low.indexOf("hurricane") >= 0 || low.indexOf("cyclone") >= 0 ||
       low.indexOf("typhoon") >= 0 || low.indexOf("storm") >= 0)              return "hurricane";
   if (low.indexOf("other") >= 0 || low.indexOf("disaster") >= 0)             return "other_disaster";
   return "";
 }

static int extract_first_digit(const String& val) {
  for (int i = 0; i < (int)val.length(); i++) {
    if (val[i] >= '0' && val[i] <= '9') return val[i] - '0';
  }
  return -1;
}

static bool parse_conf_value(const String& raw, int& out_conf) {
  String s = raw;
  s.trim();
  s.replace(",", ".");

  int st = -1;
  for (int i = 0; i < (int)s.length(); i++) {
    char c = s[i];
    if ((c >= '0' && c <= '9') || c == '.') {
      st = i;
      break;
    }
  }
  if (st < 0) return false;

  int ed = st;
  bool dot_seen = false;
  while (ed < (int)s.length()) {
    char c = s[ed];
    if (c >= '0' && c <= '9') {
      ed++;
      continue;
    }
    if (c == '.' && !dot_seen) {
      dot_seen = true;
      ed++;
      continue;
    }
    break;
  }
  if (ed <= st) return false;

  String tok = s.substring(st, ed);
  tok.trim();
  float v = tok.toFloat();

  // Preferred: confidence in [1,5]
  if (v >= 1.0f && v <= 5.0f) {
    int iv = (int)(v + 0.5f);
    if (iv < 1) iv = 1;
    if (iv > 5) iv = 5;
    out_conf = iv;
    return true;
  }
  // Backward compatibility: old normalized [0,1] -> [1,5]
  if (v >= 0.0f && v <= 1.0f) {
    float mapped = 1.0f + (4.0f * v);
    int iv = (int)(mapped + 0.5f);
    if (iv < 1) iv = 1;
    if (iv > 5) iv = 5;
    out_conf = iv;
    return true;
  }
  // Percent-like values (e.g., 87) -> [1,5]
  if (v > 5.0f && v <= 100.0f) {
    float mapped = 1.0f + (4.0f * (v / 100.0f));
    int iv = (int)(mapped + 0.5f);
    if (iv < 1) iv = 1;
    if (iv > 5) iv = 5;
    out_conf = iv;
    return true;
  }
  return false;
}

 // field 검색 (case-insensitive, "\nfield:" 또는 문자열 시작)
 static int find_field_pos(const String& low, const char* field_name) {
   String needle = String("\n") + field_name;
   int pos = low.indexOf(needle);
   if (pos >= 0) return pos + 1;  // skip \n
   if (low.startsWith(String(field_name))) return 0;
   return -1;
 }

 // field 뒤의 단일 라인 값 추출
 static String extract_line_val(const String& s, int val_start) {
   int end = s.indexOf('\n', val_start);
   if (end < 0) end = s.length();
   String v = s.substring(val_start, end);
   v.trim();
   return v;
 }

 // text: 필드 값 추출 (다음 필드(cat:/sev:/conf:)까지)
static String extract_text_val(const String& s, const String& low, int val_start) {
   int next = s.length();
   const char* fields[] = {"cat:", "sev:", "conf:"};
   for (int i = 0; i < 3; i++) {
     int p = find_field_pos(low.substring(val_start), fields[i]);
     if (p >= 0 && (val_start + p) < next) next = val_start + p;
   }
   // Also try \ncat: etc. from val_start
   for (int i = 0; i < 3; i++) {
     String nl_field = String("\n") + fields[i];
     int p = low.indexOf(nl_field, val_start);
     if (p >= 0 && p < next) next = p;
   }
   String v = s.substring(val_start, next);
   v.trim();
   return v;
 }

static String extract_unlabeled_text_block(const String& s) {
  String out = "";
  int start = 0;
  while (start <= (int)s.length()) {
    int nl = s.indexOf('\n', start);
    int end = (nl < 0) ? (int)s.length() : nl;
    String line = s.substring(start, end);
    line.trim();
    if (line.length() > 0) {
      String low = line;
      low.toLowerCase();
      bool is_label =
        low.startsWith("text:") ||
        low.startsWith("cat:") ||
        low.startsWith("category:") ||
        low.startsWith("disaster_category:") ||
        low.startsWith("disaster category:") ||
        low.startsWith("sev:") ||
        low.startsWith("severity:") ||
        low.startsWith("conf:") ||
        low.startsWith("confidence:") ||
        low.startsWith("conf_score:") ||
        low.startsWith("confidence_score:");
      if (!is_label) {
        if (out.length() > 0) out += " ";
        out += line;
      }
    }
    if (nl < 0) break;
    start = nl + 1;
  }
  out.trim();
  return out;
}

static bool parse_structured_output(const String& raw, const char* role, VlmParsed& out) {
   out.text = ""; out.cat = ""; out.sev = -1; out.conf = -1;
   out.ok_text = false; out.ok_cat = false; out.ok_sev = false; out.ok_conf = false;

   String s = raw;
   s.replace("\r\n", "\n");
   s.replace("\r", "\n");
   String low = s;
   low.toLowerCase();

   bool isA = (String(role) == "a");

   // text: field (A only)
   int text_pos = find_field_pos(low, "text:");
   if (text_pos >= 0) {
     String tv = extract_text_val(s, low, text_pos + 5);
     if (tv.length() > 0) {
       out.text = tv;
       out.ok_text = true;
     }
   }
   if (isA && !out.ok_text) {
     String tv2 = extract_unlabeled_text_block(s);
     if (tv2.length() > 0) {
       out.text = tv2;
       out.ok_text = true;
     }
   }

  // cat/category field
  int cat_pos = find_field_pos(low, "cat:");
  int cat_skip = 4;
  if (cat_pos < 0) {
    cat_pos = find_field_pos(low, "category:");
    cat_skip = 9;
  }
  if (cat_pos < 0) {
    cat_pos = find_field_pos(low, "disaster_category:");
    cat_skip = 18;
  }
  if (cat_pos < 0) {
    cat_pos = find_field_pos(low, "disaster category:");
    cat_skip = 18;
  }
  if (cat_pos >= 0) {
    String cv = extract_line_val(s, cat_pos + cat_skip);
    String norm = normalize_cat_value(cv);
    if (norm.length() > 0) {
      out.cat = norm;
      out.ok_cat = true;
    }
  }

  // sev/severity field
  int sev_pos = find_field_pos(low, "sev:");
  int sev_skip = 4;
  if (sev_pos < 0) {
    sev_pos = find_field_pos(low, "severity:");
    sev_skip = 9;
  }
  if (sev_pos >= 0) {
    String sv = extract_line_val(s, sev_pos + sev_skip);
    int val = extract_first_digit(sv);
    if (val >= 0) {
      if (val > 2) val = 2;  // sev is normalized to 0/1/2
      out.sev = val;
      out.ok_sev = true;
    }
   }

   // conf: / confidence: field
  int conf_pos = find_field_pos(low, "conf:");
  if (conf_pos < 0) conf_pos = find_field_pos(low, "confidence:");
  if (conf_pos < 0) conf_pos = find_field_pos(low, "conf_score:");
  if (conf_pos < 0) conf_pos = find_field_pos(low, "confidence_score:");
  if (conf_pos >= 0) {
    int skip = 5;
    String tail = low.substring(conf_pos);
    if (tail.startsWith("confidence_score:")) skip = 17;
    else if (tail.startsWith("confidence:")) skip = 11;
    else if (tail.startsWith("conf_score:")) skip = 11;
    String cv = extract_line_val(s, conf_pos + skip);
    int conf_v = -1;
    if (parse_conf_value(cv, conf_v)) {
      out.conf = conf_v;
      out.ok_conf = true;
    }
   }

   // Minimum requirements
   bool fields_ok = out.ok_cat && out.ok_sev && out.ok_conf;
   return fields_ok;
 }

 static int split_sentences(const String& s, String out[], int max_out) {
   int count = 0;
   int start = 0;
   int n = s.length();
   for (int i = 0; i < n && count < max_out; i++) {
     char ch = s[i];
     if (ch == '.' || ch == '!' || ch == '?') {
       String part = s.substring(start, i + 1);
       part.trim();
       if (part.length() > 0 && count_words(part) > 0) out[count++] = part;
       start = i + 1;
     }
   }
   if (count < max_out) {
     String tail = s.substring(start);
     tail.trim();
     if (tail.length() > 0 && count_words(tail) > 0) {
       if (!tail.endsWith(".")) tail += ".";
       out[count++] = tail;
     }
   }
   return count;
 }
 
 static void strip_leading_template(String& sent) {
   sent.trim();
   if (sent.length() == 0) return;
 
   String low = sent;
   low.toLowerCase();
 
   const char* prefixes[] = {
     "first, ", "second, ", "first ", "second ",
     "the image shows ", "the image depicts ",
     "this image shows ", "this image depicts ",
     "the photo shows ", "the photo depicts ",
     "the visible damage ", "visible damage ",
     "the main damage in the image is ", "the main damage in this image is ",
     "the main damage is ", "the damage in the image is ",
     "the damage shown is ",
     "in this image, ", "in the image, ",
     "in a disaster scene, ", "in the disaster scene, ",
     "in this disaster scene, ", "in this disaster, ",
     "the danger to people in the image is ",
     "the danger to people in this image is ",
     "the danger to people is ",
     "the danger to nearby people is ",
     "the potential danger is ",
     "you see ", "you see. ", "you see, ",
     "i see ", "i can see ",
     "the scene shows ", "this scene shows ",
     "the life safety threat in this scene is ",
     "the life safety threat is ",
     "the life safety threat here is ",
     "the presence of ",
     "immediate measures include ",
     "immediate measures should be taken to ",
     "immediate measures: ",
     "immediate action includes ",
     "immediate action is needed to ",
     "immediate action should be taken to ",
     "immediate action: ",
     "immediate response actions include ",
     "response actions include ",
     "states ", "describes "
   };
   for (size_t i = 0; i < sizeof(prefixes) / sizeof(prefixes[0]); i++) {
     String p = String(prefixes[i]);
     if (low.startsWith(p)) {
       sent = sent.substring(p.length());
       sent.trim();
       low = sent;
       low.toLowerCase();
     }
   }
 }
 
static void postprocess_text_output(String& s, const char* role) {
  normalize_inline(s, 0);
  s.trim();
  if (s.length() == 0) {
    s = "";
    return;
  }

  bool isA = (String(role) == "a");
  if (isA) {
    // Keep A text as close to model output as possible to preserve per-image diversity.
    // Only strip explicit structured labels and keep first 2 sentences.
    String unlabeled = extract_unlabeled_text_block(s);
    if (unlabeled.length() > 0) {
      s = unlabeled;
    } else {
      String lowa = s;
      lowa.toLowerCase();
      if (lowa.indexOf("cat:") >= 0 || lowa.indexOf("category:") >= 0 ||
          lowa.indexOf("sev:") >= 0 || lowa.indexOf("severity:") >= 0 ||
          lowa.indexOf("conf:") >= 0 || lowa.indexOf("confidence:") >= 0) {
        s = "";
        return;
      }
    }
    s.replace("text:", "");
    s.replace("Text:", "");
    s.replace("TEXT:", "");
    s.replace("**", "");
    s.replace("__", "");
    normalize_inline(s, 0);
    s.trim();

    String partsA[3];
    int cntA = split_sentences(s, partsA, 3);
    if (cntA <= 0) {
      s = "";
      return;
    }
    String a1 = partsA[0];
    String a2 = (cntA >= 2) ? partsA[1] : "";
    a1.trim();
    a2.trim();
    if (a1.length() == 0 || count_words(a1) < 4) {
      s = "";
      return;
    }
    if (a1[0] >= 'a' && a1[0] <= 'z') a1.setCharAt(0, (char)(a1[0] - 'a' + 'A'));
    trim_sentence_words(a1, 25);
    if (!a1.endsWith(".")) a1 += ".";
    if (a2.length() > 0) {
      if (a2[0] >= 'a' && a2[0] <= 'z') a2.setCharAt(0, (char)(a2[0] - 'a' + 'A'));
      trim_sentence_words(a2, 25);
      if (!a2.endsWith(".")) a2 += ".";
      s = a1 + " " + a2;
    } else {
      s = a1;
    }
    s.trim();
    return;
  }

  s.replace("Sentence 1: ", "");
  s.replace("Sentence 2: ", "");
   s.replace("Sentence 1:", "");
   s.replace("Sentence 2:", "");
   s.replace("First sentence ", "");
   s.replace("Second sentence ", "");
   s.replace("First, ", "");
   s.replace("Second, ", "");
   s.replace("**", "");
   s.replace("Monitoring: ", "");
   s.replace("monitoring: ", "");
   s.replace("Emergency action: ", "");
   s.replace("emergency action: ", "");
   s.replace("Emergency action needed: ", "");
   s.replace("Emergency Action Needed: ", "");
   s.replace("emergency action needed: ", "");
   s.replace("Damage: ", "");
   s.replace("damage: ", "");
   s.replace("Action: ", "");
   s.replace("action: ", "");
   s.replace("Response: ", "");
   s.replace("response: ", "");
   s.replace("State the immediate needs of responders, such as ", "");
   s.replace("State the immediate needs of responders", "");
   s.replace("State responders should ", "");
   s.replace("State responders ", "");
   s.replace("Responders should ", "");
   s.replace("responders should ", "");
   s.replace("Emergency responders should ", "");
   s.replace("emergency responders should ", "");
 
   s.replace("bould ers", "boulders");
   s.replace("Bould ers", "Boulders");
   s.replace("injuries or injuries", "injuries");
   s.replace("risk of risk", "risk");
 
  String parts[3];
  int cnt = split_sentences(s, parts, 3);
  if (cnt < 1) {
    s = "";
    return;
  }
 
   String s1 = (cnt >= 1) ? parts[0] : "";
   String s2 = (cnt >= 2) ? parts[1] : "";
   s1.trim();
   s2.trim();
 
   strip_leading_template(s1);
   strip_leading_template(s2);
 
  if (s1.length() == 0) {
    s = "";
    return;
  }
  if (count_words(s1) < 4) {
    s = "";
    return;
  }
  if (isA && s2.length() > 0 && count_words(s2) < 4) {
    s = "";
    return;
  }
 
   if (s1.length() > 0 && s1[0] >= 'a' && s1[0] <= 'z') {
     s1.setCharAt(0, (char)(s1[0] - 'a' + 'A'));
   }
  if (isA && s2.length() > 0 && s2[0] >= 'a' && s2[0] <= 'z') {
    s2.setCharAt(0, (char)(s2[0] - 'a' + 'A'));
  }
 
   s1.replace("  ", " ");
   s2.replace("  ", " ");
 
   trim_sentence_words(s1, 25);
  if (isA && s2.length() > 0) trim_sentence_words(s2, 25);
 
   if (!s1.endsWith(".")) s1 += ".";
  if (isA && s2.length() > 0 && !s2.endsWith(".")) s2 += ".";

  if (isA && s2.length() > 0) s = s1 + " " + s2;
  else if (isA) s = s1;
  else s = s1;
}
 
 static void sanitize_a_final(String& s, const String& hint, const String& imgid) {
   String src = s;
   if (src.length() < 20 && hint.length() > src.length()) src = hint;
   normalize_inline(src, 500);
   strip_a_labels(src);
   src.trim();
 
   String parts[4];
   int cnt = split_sentences(src, parts, 4);
 
   String s1 = "";
   for (int i = 0; i < cnt; i++) {
     String cand = parts[i];
     strip_leading_template(cand);
     strip_a_labels(cand);
     cand.trim();
     if (starts_with_alpha(cand) && count_words(cand) >= 4 && !is_garbage_text(cand)) {
       s1 = cand;
       break;
     }
   }
 
   if (s1.length() == 0) return;
 
   String s2 = "";
   for (int i = 1; i < cnt; i++) {
     String cand = parts[i];
     strip_leading_template(cand);
     strip_a_labels(cand);
     cand.trim();
     if (starts_with_alpha(cand) && count_words(cand) >= 4 && !is_garbage_text(cand)) {
       s2 = cand;
       break;
     }
   }
 
   if (s1[0] >= 'a' && s1[0] <= 'z') s1.setCharAt(0, (char)(s1[0] - 'a' + 'A'));
   trim_sentence_words(s1, 25);
   if (!s1.endsWith(".")) s1 += ".";
 
   if (s2.length() > 0) {
     if (s2[0] >= 'a' && s2[0] <= 'z') s2.setCharAt(0, (char)(s2[0] - 'a' + 'A'));
     trim_sentence_words(s2, 25);
     if (!s2.endsWith(".")) s2 += ".";
     s = s1 + " " + s2;
   } else {
     s = s1;
   }
 }
 
static void sanitize_b_text(String& s) {
  String parts[4];
  int cnt = split_sentences(s, parts, 4);
  String best = "";
  int best_score = -1000;

  for (int i = 0; i < cnt; i++) {
    String cand = parts[i];
    cand.trim();
    if (cand.length() == 0) continue;

    strip_leading_template(cand);
    strip_a_labels(cand);
    cand.replace("The scene shows ", "");
    cand.replace("the scene shows ", "");
    cand.replace("One visible damage is ", "");
    cand.replace("one visible damage is ", "");
    cand.replace("The visible damage is ", "");
    cand.replace("the visible damage is ", "");
    cand.trim();
    if (!starts_with_alpha(cand)) continue;
    if (cand.length() > 0 && cand[0] >= 'a' && cand[0] <= 'z') {
      cand.setCharAt(0, (char)(cand[0] - 'a' + 'A'));
    }
    trim_sentence_words(cand, 25);
    while (ends_with_bad_tail(cand)) {
      int cut = cand.lastIndexOf(' ');
      if (cut <= 0) break;
      cand = cand.substring(0, cut);
      cand.trim();
    }
    if (!starts_with_alpha(cand) || count_words(cand) < 4) continue;
    if (!cand.endsWith(".")) cand += ".";

    int score = 0;
    if (b_has_damage_token(cand)) score += 6;
    if (b_has_anchor_token(cand)) score += 4;
    String low = cand;
    low.toLowerCase();
    if (low.indexOf("road") >= 0 || low.indexOf("bridge") >= 0 || low.indexOf("path") >= 0 ||
        low.indexOf("lane") >= 0 || low.indexOf("barrier") >= 0 || low.indexOf("shore") >= 0) {
      score += 3;
    }
    if (b_has_speculation_token(cand)) score -= 3;
    if (b_has_scenic_token(cand)) score -= 4;
    if (has_bad_generic_phrase(cand)) score -= 5;

    if (score > best_score) {
      best_score = score;
      best = cand;
    }
  }

  if (best.length() == 0) {
    s = "";
    return;
  }
  s = best;
}
 
static void sanitize_c_text(String& s) {
  String parts[4];
  int cnt = split_sentences(s, parts, 4);
  String best = "";
  int best_score = -1000;
  for (int i = 0; i < cnt; i++) {
    String o = parts[i];
    o.trim();
    strip_leading_template(o);
    strip_a_labels(o);
    o.trim();
    if (!starts_with_alpha(o)) continue;
    if (o.length() > 0 && o[0] >= 'a' && o[0] <= 'z') {
      o.setCharAt(0, (char)(o[0] - 'a' + 'A'));
    }
    trim_sentence_words(o, 25);
    while (ends_with_bad_tail(o)) {
      int cut = o.lastIndexOf(' ');
      if (cut <= 0) break;
      o = o.substring(0, cut);
      o.trim();
    }
    if (!starts_with_alpha(o) || count_words(o) < 4) continue;
    if (!o.endsWith(".")) o += ".";

    String low = o;
    low.toLowerCase();
    int score = 0;
    // Prioritize direct human-risk wording for C.
    if (c_has_human_risk_token(low)) score += 8;
    if (low.indexOf("evacuat") >= 0 || low.indexOf("rescue") >= 0) score += 4;
    if (low.indexOf("injury") >= 0 || low.indexOf("danger") >= 0 || low.indexOf("hazard") >= 0) score += 4;
    // Hazard cues are secondary support.
    if (low.indexOf("collapse") >= 0 || low.indexOf("collapsed") >= 0) score += 3;
    if (low.indexOf("rockfall") >= 0 || low.indexOf("boulder") >= 0 || low.indexOf("debris") >= 0) score += 2;
    if (low.indexOf("cliff") >= 0 || low.indexOf("slope") >= 0 || low.indexOf("hillside") >= 0) score += 2;
    if (low.indexOf("blocked") >= 0 || low.indexOf("buried") >= 0 || low.indexOf("trapped") >= 0) score += 2;
    // Penalize peripheral context-only descriptions.
    if (low.indexOf("shoreline") >= 0 || low.indexOf("partially submerged") >= 0) score -= 2;
    if (low.indexOf("vegetation") >= 0 || low.indexOf("background") >= 0) score -= 2;

    if (score > best_score) {
      best_score = score;
      best = o;
    }
  }
  if (best.length() == 0) {
    s = "";
    return;
  }
  s = best;
}

static void refine_c_text_priority(String& text, const String& raw, const String& cat_hint) {
  String merged = text;
  if (raw.length() > 0) {
    if (merged.length() > 0) merged += " ";
    merged += raw;
  }
  merged.trim();
  if (merged.length() == 0) return;

  String best = text;
  sanitize_c_text(best);

  String cat_low = cat_hint;
  cat_low.toLowerCase();

  String parts[8];
  int cnt = split_sentences(merged, parts, 8);
  for (int i = 0; i < cnt; i++) {
    String cand = parts[i];
    sanitize_c_text(cand);
    if (cand.length() == 0) continue;

    String c_low = cand;
    c_low.toLowerCase();

    int sc = 0;
    if (c_has_human_risk_token(c_low)) sc += 10;
    if (c_low.indexOf("evacuat") >= 0 || c_low.indexOf("rescue") >= 0) sc += 5;
    if (c_low.indexOf("injury") >= 0 || c_low.indexOf("danger") >= 0 ||
        c_low.indexOf("hazard") >= 0 || c_low.indexOf("unsafe") >= 0) sc += 4;
    if (cat_low == "landslide") {
      if (c_low.indexOf("collapse") >= 0 || c_low.indexOf("collapsed") >= 0) sc += 3;
      if (c_low.indexOf("cliff") >= 0 || c_low.indexOf("slope") >= 0 || c_low.indexOf("rockfall") >= 0) sc += 3;
      if (c_low.indexOf("shoreline") >= 0 || c_low.indexOf("partially submerged") >= 0) sc -= 2;
    }

    String b_low = best;
    b_low.toLowerCase();
    int bsc = 0;
    if (c_has_human_risk_token(b_low)) bsc += 10;
    if (b_low.indexOf("evacuat") >= 0 || b_low.indexOf("rescue") >= 0) bsc += 5;
    if (b_low.indexOf("injury") >= 0 || b_low.indexOf("danger") >= 0 ||
        b_low.indexOf("hazard") >= 0 || b_low.indexOf("unsafe") >= 0) bsc += 4;
    if (cat_low == "landslide") {
      if (b_low.indexOf("collapse") >= 0 || b_low.indexOf("collapsed") >= 0) bsc += 3;
      if (b_low.indexOf("cliff") >= 0 || b_low.indexOf("slope") >= 0 || b_low.indexOf("rockfall") >= 0) bsc += 3;
      if (b_low.indexOf("shoreline") >= 0 || b_low.indexOf("partially submerged") >= 0) bsc -= 2;
    }

    if (best.length() == 0 || sc > bsc) best = cand;
  }

  if (best.length() > 0) text = best;
}

static void ensure_c_human_impact(String& text, const String& cat_hint, int sev_hint) {
  text.trim();
  if (text.length() == 0) return;
  if (c_has_human_risk_token(text)) return;

  String c = cat_hint;
  c.toLowerCase();

  String hazard = "falling debris and unstable ground";
  if (c == "flood") hazard = "rising water and unstable footing";
  else if (c == "fire") hazard = "heat, smoke exposure, and rapid fire spread";
  else if (c == "earthquake") hazard = "aftershocks and structural instability";
  else if (c == "hurricane") hazard = "strong winds, debris impact, and flooding";

  String impact;
  if (sev_hint >= 2) {
    impact = "this creates a high injury risk for nearby people and requires evacuation with strict access control";
  } else if (sev_hint == 1) {
    impact = "this can injure nearby people and requires restricted access with safety monitoring";
  } else {
    impact = "this poses a potential safety risk to nearby people and requires caution with controlled access";
  }

  String base = text;
  if (base.endsWith(".")) base = base.substring(0, base.length() - 1);
  base.trim();
  if (base.length() == 0) {
    base = "The visible hazard suggests danger from " + hazard;
  } else if (base.indexOf("because") < 0 && base.indexOf("due to") < 0) {
    base += ", with danger from " + hazard;
  }

  text = base + ", and " + impact + ".";
  normalize_inline(text, 220);
  text.trim();
}

static bool salvage_vlm_text(const String& raw, const char* role, String& out) {
  String s = raw;
  normalize_inline(s, 500);
  strip_a_labels(s);
  s.trim();
  if (s.length() == 0 || has_non_ascii(s)) return false;

  bool isA = (String(role) == "a");
  String parts[4];
  int cnt = split_sentences(s, parts, 4);
  if (cnt <= 0) return false;

  String s1 = "";
  String s2 = "";
  for (int i = 0; i < cnt; i++) {
    String cand = parts[i];
    strip_leading_template(cand);
    strip_a_labels(cand);
    cand.trim();
    String low = cand;
    low.toLowerCase();
    if (low.indexOf("cat:") >= 0 || low.indexOf("sev:") >= 0 ||
        low.indexOf("conf:") >= 0 || low.indexOf("confidence:") >= 0 ||
        low.indexOf("category:") >= 0 || low.indexOf("severity:") >= 0) {
      continue;
    }
    if (!starts_with_alpha(cand)) continue;
    if (count_words(cand) < 4) continue;
    if (is_garbage_text(cand)) continue;
    if (s1.length() == 0) s1 = cand;
    else if (isA && s2.length() == 0) s2 = cand;
    if (!isA || (s1.length() > 0 && s2.length() > 0)) break;
  }

  if (s1.length() == 0) return false;
  if (s1[0] >= 'a' && s1[0] <= 'z') s1.setCharAt(0, (char)(s1[0] - 'a' + 'A'));
  trim_sentence_words(s1, 25);
  if (!s1.endsWith(".")) s1 += ".";

  if (isA && s2.length() > 0) {
    if (s2[0] >= 'a' && s2[0] <= 'z') s2.setCharAt(0, (char)(s2[0] - 'a' + 'A'));
    trim_sentence_words(s2, 25);
    if (!s2.endsWith(".")) s2 += ".";
    out = s1 + " " + s2;
  } else {
    out = s1;
  }
  out.trim();
  return (out.length() >= 12 && count_words(out) >= 4);
}
 
 // ============================================================
 //  WiFi  (blocking reconnect with backoff)
 // ============================================================
static void wifi_connect() {
  if (WiFi.status() == WL_CONNECTED) return;
 
  Serial.printf("[WIFI] connecting %s ...\n", WIFI_SSID);
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
 
  unsigned long t0 = millis();
  while (WiFi.status() != WL_CONNECTED) {
    wdt_kick();
    delay(250);
    if (millis() - t0 > 15000) {
      Serial.println("[WIFI] timeout -> retry");
      WiFi.disconnect(true);
      delay(500);
       WiFi.begin(WIFI_SSID, WIFI_PASS);
       t0 = millis();
     }
   }
   Serial.printf("[WIFI] connected  IP=%s\n", WiFi.localIP().toString().c_str());
 }
 
static void wifi_recover_hard() {
  Serial.println("[WIFI] hard recover start");
  WiFi.disconnect(true, true);
  wdt_kick();
  delay(700);
  WiFi.mode(WIFI_OFF);
  wdt_kick();
  delay(300);
  WiFi.mode(WIFI_STA);
  wdt_kick();
  delay(200);
  wifi_connect();
  Serial.println("[WIFI] hard recover done");
}
 
 // ============================================================
 //  LLM Module Health Check
 // ============================================================
 static bool llm_health_check() {
   return module_llm.checkConnection();
 }
 
 static bool llm_quick_infer_check(const uint8_t* probe_img, size_t probe_size) {
   if (!probe_img || probe_size < 16) {
     Serial.println("[LLM] infer health-check skipped (no probe image)");
     return true;
   }
 
   if (probe_size > 2 * 1024 * 1024) {
     Serial.printf("[LLM] infer health-check skipped (probe too large: %uKB)\n",
                   (unsigned)(probe_size / 1024));
     return true;
   }
 
   String req_id = String("h_") + String((uint32_t)esp_random(), HEX);
   String out;
   unsigned long t0 = millis();
 
   module_llm.vlm.inference(vlm_work_id, (uint8_t*)probe_img, probe_size, req_id);
   vTaskDelay(pdMS_TO_TICKS(120));
   module_llm.vlm.inferenceAndWaitResult(
       vlm_work_id,
       String("Describe one visible damage in one short sentence."),
       [&](String& chunk) { append_chunk(out, chunk); },
       12000,
       String("hc_") + req_id);
 
   out.trim();
   unsigned long dt = millis() - t0;
   bool ok = (out.length() >= 8);
   Serial.printf("[LLM] infer health-check %s (%lums, len=%d)\n",
                 ok ? "ok" : "fail", dt, (int)out.length());
   return ok;
 }
 
static int compute_vlm_wait_ms(size_t img_size, const char* role) {
  // Balance false timeout avoidance with overnight progress.
  int wait_ms = 12000;
  if (img_size > 80 * 1024)   wait_ms = 15000;
  if (img_size > 200 * 1024)  wait_ms = 18000;
  if (img_size > 400 * 1024)  wait_ms = 22000;
  if (img_size > 800 * 1024)  wait_ms = 26000;
  // Role A prompt is longer (2 sentences + action + structured fields).
  if (role && role[0] == 'a') wait_ms += 14000;

  if (wait_ms < VLM_WAIT_MS) wait_ms = VLM_WAIT_MS;
  if (wait_ms > VLM_WAIT_MAX_MS) wait_ms = VLM_WAIT_MAX_MS;
  return wait_ms;
}

static bool ensure_uart_921600() {
  // 이미 921600이면 바로 OK
  Serial2.updateBaudRate(921600);
  delay(20);
  if (module_llm.checkConnection()) return true;

  // 115200으로 떨어져 있으면 연결 후 업그레이드
  Serial2.updateBaudRate(115200);
  delay(30);
  if (!module_llm.checkConnection()) return false;

  module_llm.sys.setBaudRate(921600);
  delay(120);
  Serial2.updateBaudRate(921600);
  delay(30);
  return module_llm.checkConnection();
}

static constexpr uint8_t CORE_S3_AW9523_ADDR = 0x58;
static constexpr uint8_t CORE_S3_PORT0_REG = 0x02;
static constexpr uint8_t CORE_S3_BUS_EN_BIT = 0x02;
static constexpr uint8_t CORE_S3_BOOST_EN_BIT = 0x80;
static constexpr uint32_t CORE_S3_I2C_FREQ = 400000;

static bool read_core_s3_power_regs(uint8_t& p0, uint8_t& p1) {
  uint8_t buf[2] = {0, 0};
  if (!M5.In_I2C.readRegister(CORE_S3_AW9523_ADDR, CORE_S3_PORT0_REG, buf, sizeof(buf), CORE_S3_I2C_FREQ)) {
    return false;
  }
  p0 = buf[0];
  p1 = buf[1];
  return true;
}

static bool force_core_s3_bus_power(bool enable) {
  uint8_t p0 = 0, p1 = 0;
  if (!read_core_s3_power_regs(p0, p1)) return false;

  uint8_t n0 = p0;
  uint8_t n1 = p1;
  if (enable) {
    n0 |= CORE_S3_BUS_EN_BIT;
    n1 |= CORE_S3_BOOST_EN_BIT;
  } else {
    n0 &= ~CORE_S3_BUS_EN_BIT;
    if ((n0 & CORE_S3_BUS_EN_BIT) == 0) n1 &= ~CORE_S3_BOOST_EN_BIT;
  }

  uint8_t wbuf[2] = { n0, n1 };
  bool w_ok = M5.In_I2C.writeRegister(CORE_S3_AW9523_ADDR, CORE_S3_PORT0_REG, wbuf, sizeof(wbuf), CORE_S3_I2C_FREQ);
  delay(80);

  uint8_t rp0 = 0, rp1 = 0;
  bool r_ok = read_core_s3_power_regs(rp0, rp1);
  bool bus_on = r_ok && ((rp0 & CORE_S3_BUS_EN_BIT) != 0);
  bool boost_on = r_ok && ((rp1 & CORE_S3_BOOST_EN_BIT) != 0);

  if (!w_ok || !r_ok) return false;
  return enable ? (bus_on && boost_on) : !bus_on;
}

static bool set_ext_output_with_fallback(bool enable) {
  M5.Power.setExtOutput(enable);
  delay(100);
  bool ext = M5.Power.getExtOutput();
  if (ext == enable) return true;

  bool forced = force_core_s3_bus_power(enable);
  delay(100);
  bool ext2 = M5.Power.getExtOutput();
  return forced && (ext2 == enable);
}

// LLM Mate external power 환경에서는 M5 ext-output power cycle이 실제 모듈
// 전원 차단과 일치하지 않을 수 있다. 따라서 런타임 복구는 UART 재초기화와
// 모듈 소프트 리셋 중심으로 수행한다.
static bool recover_module(const uint8_t* probe_img, size_t probe_size) {
  if (g_next_recover_allowed_ms > millis()) {
    unsigned long remain = (g_next_recover_allowed_ms - millis()) / 1000UL;
    Serial.printf("[RECOVER] in cooldown, skip soft recover (%lus left)\n", remain);
    return false;
  }

  Serial.println("[LLM] === RECOVER: UART re-init + module soft reset ===");
  wdt_kick();

  int rxd = M5.getPin(m5::pin_name_t::port_c_rxd);
  int txd = M5.getPin(m5::pin_name_t::port_c_txd);

  // 1. 연결되어 있으면 모듈 내부 상태를 먼저 soft reset.
  Serial2.updateBaudRate(921600);
  delay(50);
  module_llm.begin(&Serial2);
  delay(50);
  if (module_llm.checkConnection()) {
    Serial.println("[LLM] Soft reset on active link...");
    module_llm.sys.reset(true);
    delay(700);
    wdt_kick();
  }

  // 2. UART를 115200 cold-boot 기본값으로 재오픈.
  Serial2.end();
  delay(80);
  Serial2.begin(115200, SERIAL_8N1, rxd, txd);
  delay(500);
  module_llm.begin(&Serial2);
  delay(200);

  // 3. 115200 연결 대기.
  bool connected = false;
  for (int r = 0; r < 16; r++) {
    if (module_llm.checkConnection()) {
      connected = true;
      break;
    }
    wdt_kick();
    delay(250);
  }

  if (!connected) {
    // 이미 921600으로 살아있을 수 있으니 한 번 더 fast probe.
    Serial.println("[LLM] 115200 reconnect failed. Trying 921600 direct...");
    Serial2.end();
    delay(80);
    Serial2.begin(921600, SERIAL_8N1, rxd, txd);
    delay(200);
    module_llm.begin(&Serial2);
    delay(100);
    for (int r = 0; r < 12; r++) {
      if (module_llm.checkConnection()) {
        connected = true;
        break;
      }
      wdt_kick();
      delay(250);
    }
    if (!connected) {
      Serial.println("[RECOVER] Module reconnect failed after UART re-init");
      mark_recover_fail_cooldown();
      return false;
    }
  } else {
    Serial.println("[LLM] Connected at 115200. Upgrading to 921600...");
    module_llm.sys.setBaudRate(921600);
    delay(120);
    Serial2.updateBaudRate(921600);
    delay(120);
  }

  // 4. 921600 상태에서 VLM 세션 재설정.
  wdt_kick();
  m5_module_llm::ApiVlmSetupConfig_t cfg;
  cfg.model = VLM_MODEL_ID;
  cfg.max_token_len = 300;
  vlm_work_id = module_llm.vlm.setup(cfg);
  vlm_work_id.trim();
  if (vlm_work_id.length() == 0) vlm_work_id = "vlm_work";
  delay(300);
  wdt_kick();

  bool health_ok = llm_health_check();
  if (health_ok && probe_img && probe_size >= 16 && probe_size <= (2 * 1024 * 1024)) {
    health_ok = llm_quick_infer_check(probe_img, probe_size);
  }

  if (health_ok) {
    g_recover_fail_streak = 0;
    g_timeout_streak_global = 0;
    g_next_recover_allowed_ms = 0;
    Serial.printf("[LLM] Recovery OK. work_id=%s (921600 baud)\n", vlm_work_id.c_str());
    return true;
  }

  // If UART link is alive but quick infer-check failed, continue to avoid long stall loops.
  if (module_llm.checkConnection()) {
    Serial.println("[RECOVER] health check failed but link alive -> continue");
    return true;
  }

  Serial.println("[RECOVER] Health check failed after soft recovery");
  mark_recover_fail_cooldown();
  return false;
}


 // ============================================================
 //  JPEG resize for VLM
 // ============================================================
static uint16_t* g_rgb565_buf = nullptr;
static int g_rgb565_w = 0;
static int g_rgb565_h = 0;

static int jpeg_decode_cb(JPEGDRAW *pDraw) {
  if (!g_rgb565_buf || g_rgb565_w <= 0 || g_rgb565_h <= 0) return 0;
  if (pDraw->x < 0 || pDraw->y < 0) return 0;
  if (pDraw->x >= g_rgb565_w || pDraw->y >= g_rgb565_h) return 1;

  int copy_w = pDraw->iWidthUsed;
  int copy_h = pDraw->iHeight;
  if (copy_w <= 0 || copy_h <= 0) return 1;

  if (copy_w > (g_rgb565_w - pDraw->x)) copy_w = g_rgb565_w - pDraw->x;
  if (copy_h > (g_rgb565_h - pDraw->y)) copy_h = g_rgb565_h - pDraw->y;
  if (copy_w <= 0 || copy_h <= 0) return 1;

  for (int j = 0; j < copy_h; j++) {
    memcpy(&g_rgb565_buf[(pDraw->y + j) * g_rgb565_w + pDraw->x],
           &pDraw->pPixels[j * pDraw->iWidth],
           copy_w * sizeof(uint16_t));
  }
  return 1;
}

 static bool resize_jpeg_for_vlm(const uint8_t* src, size_t src_len,
                                  uint8_t* dst, size_t dst_cap, size_t& dst_len) {
   JPEGDEC* dec = new JPEGDEC();
   if (!dec) return false;

   if (!dec->openRAM((uint8_t*)src, src_len, jpeg_decode_cb)) {
     delete dec;
     return false;
   }

  int orig_w = dec->getWidth();
  int orig_h = dec->getHeight();
  int orig_long = (orig_w > orig_h) ? orig_w : orig_h;
  int orig_short = (orig_w < orig_h) ? orig_w : orig_h;

  const int divisors[4] = {1, 2, 4, 8};
  int best_div = 0;
  unsigned long best_score = 0xFFFFFFFFUL;

  for (int i = 0; i < 4; i++) {
    int d = divisors[i];
    int sw = orig_w / d;
    int sh = orig_h / d;
    if (sw <= 0 || sh <= 0) continue;
    int long_side = (sw > sh) ? sw : sh;
    int short_side = (sw < sh) ? sw : sh;
    if (short_side < VLM_TARGET_MIN_SHORT_SIDE) continue;
    if (long_side > VLM_TARGET_LONG_SIDE) continue;

    unsigned long score = (unsigned long)(VLM_TARGET_LONG_SIDE - long_side);
    if (best_div == 0 || score < best_score) {
      best_div = d;
      best_score = score;
    }
  }

  if (best_div == 0) {
    for (int i = 0; i < 4; i++) {
      int d = divisors[i];
      int sw = orig_w / d;
      int sh = orig_h / d;
      if (sw <= 0 || sh <= 0) continue;
      int long_side = (sw > sh) ? sw : sh;
      int short_side = (sw < sh) ? sw : sh;
      if (short_side < VLM_TARGET_MIN_SHORT_SIDE) continue;

      unsigned long delta = (long_side > VLM_TARGET_LONG_SIDE)
                              ? (unsigned long)(long_side - VLM_TARGET_LONG_SIDE)
                              : (unsigned long)(VLM_TARGET_LONG_SIDE - long_side);
      if (long_side > VLM_TARGET_LONG_SIDE) delta += 1000UL;

      if (best_div == 0 || delta < best_score) {
        best_div = d;
        best_score = delta;
      }
    }
  }

  if (best_div == 0) best_div = 1;

  int divisor = best_div;
  int scale_flag = 0; 
  if (divisor == 2) scale_flag = JPEG_SCALE_HALF;
  else if (divisor == 4) scale_flag = JPEG_SCALE_QUARTER;
  else if (divisor == 8) scale_flag = JPEG_SCALE_EIGHTH;

  if (divisor == 1 && orig_long <= VLM_TARGET_LONG_SIDE && orig_short <= VLM_TARGET_LONG_SIDE) {
    dec->close();
    delete dec;
    return false;
  }

   int scaled_w = orig_w / divisor;
   int scaled_h = orig_h / divisor;
   size_t rgb_bytes = (size_t)scaled_w * scaled_h * sizeof(uint16_t);

  g_rgb565_buf = (uint16_t*)ps_malloc(rgb_bytes);
  if (!g_rgb565_buf) {
    dec->close();
    delete dec;
    return false;
  }
  g_rgb565_w = scaled_w;
  g_rgb565_h = scaled_h;

   dec->setPixelType(RGB565_LITTLE_ENDIAN);
  if (!dec->decode(0, 0, scale_flag)) {
    free(g_rgb565_buf); g_rgb565_buf = nullptr;
    g_rgb565_w = 0;
    g_rgb565_h = 0;
    dec->close();
    delete dec;
    return false;
  }
   dec->close();
   delete dec;
   dec = nullptr;

   JPEGENC* enc = new JPEGENC();
  if (!enc) {
    free(g_rgb565_buf); g_rgb565_buf = nullptr;
    g_rgb565_w = 0;
    g_rgb565_h = 0;
    return false;
  }

   int rc = enc->open(dst, dst_cap);
  if (rc != JPEGE_SUCCESS) {
    free(g_rgb565_buf); g_rgb565_buf = nullptr;
    g_rgb565_w = 0;
    g_rgb565_h = 0;
    delete enc;
    return false;
  }

   JPEGENCODE jpe;
   rc = enc->encodeBegin(&jpe, scaled_w, scaled_h, JPEGE_PIXEL_RGB565,
                         JPEGE_SUBSAMPLE_420, JPEGE_Q_MED);
  if (rc != JPEGE_SUCCESS) {
    free(g_rgb565_buf); g_rgb565_buf = nullptr;
    g_rgb565_w = 0;
    g_rgb565_h = 0;
    enc->close();
    delete enc;
    return false;
  }

   int pitch = scaled_w * sizeof(uint16_t);
   rc = enc->addFrame(&jpe, (uint8_t*)g_rgb565_buf, pitch);
  if (rc != JPEGE_SUCCESS) {
    free(g_rgb565_buf); g_rgb565_buf = nullptr;
    g_rgb565_w = 0;
    g_rgb565_h = 0;
    enc->close();
    delete enc;
    return false;
  }

   int encoded_size = enc->close();
  delete enc;
  enc = nullptr;
  free(g_rgb565_buf);
  g_rgb565_buf = nullptr;
  g_rgb565_w = 0;
  g_rgb565_h = 0;

   if (encoded_size <= 0) return false;

   dst_len = (size_t)encoded_size;
   return true;
 }

 // ============================================================
 //  VLM query
 // ============================================================
static bool vlm_query(const uint8_t* img, size_t img_size,
                      const char* prompt,
                      const char* role,
                      String& out_text,
                      unsigned long& out_ms,
                      unsigned long& out_uart_ms,
                      unsigned long& out_infer_ms) {
  unsigned long t0 = millis();
  size_t raw_len = img_size;
  g_last_vlm_fail_reason = "";
  out_uart_ms = 0;
  out_infer_ms = 0;
  wdt_kick();
 
  String req_id = String("v_") + String((uint32_t)esp_random(), HEX);
 
  unsigned long push_t0 = millis();
  module_llm.vlm.inference(vlm_work_id, (uint8_t*)img, raw_len, req_id);
   unsigned long push_ms = millis() - push_t0;
   out_uart_ms = push_ms;
   g_last_push_ms = push_ms;
   unsigned long kb = (unsigned long)(img_size / 1024);
   if (kb == 0) kb = 1;
   g_last_push_kbps = (push_ms > 0) ? ((kb * 1000UL) / push_ms) : 0;
   vTaskDelay(pdMS_TO_TICKS(250));
 
  if (g_last_push_kbps > 0 && g_last_push_kbps < UART_MIN_PUSH_KBPS) {
    bool hs_ok = ensure_uart_921600();
    g_last_vlm_fail_reason = "uart_link_slow";
    out_ms = millis() - t0;
    out_text = "";
    wdt_kick();
    return false;
  }
 
  int wait_ms = compute_vlm_wait_ms(img_size, role);
  String buf;
  unsigned long infer_t0 = millis();
  module_llm.vlm.inferenceAndWaitResult(
      vlm_work_id,
      String(prompt),
      [&](String& chunk) {
        append_chunk(buf, chunk);
        wdt_kick();
      },
      wait_ms,
      String("w_") + req_id);
  out_infer_ms = millis() - infer_t0;
  wdt_kick();
 
  out_ms = millis() - t0;
  out_text = buf;
  out_text.trim();
 
   if (out_ms < 1000) {
     g_last_vlm_fail_reason = "too_fast_invalid";
     return false;
   }
   if (out_text.length() == 0) {
     if (out_ms + 200 >= (unsigned long)wait_ms) g_last_vlm_fail_reason = "timeout_empty";
     else g_last_vlm_fail_reason = "empty_text";
     return false;
   }
   return true;
 }
 
 // ============================================================
 //  HTTP Multipart POST
 // ============================================================
 static bool http_post_multipart(const String& meta_json,
                                 const uint8_t* img, size_t img_size,
                                 const String& image_id,
                                 int&           out_code,
                                 String&        out_resp,
                                 unsigned long& out_http_ms,
                                 size_t&        out_tx,
                                 size_t&        out_rx,
                                 const char* host = PI_HOST,
                                 int port = PI_PORT,
                                 const char* endpoint = PI_ENDPOINT_INGEST) {
   out_code    = -1;
   out_resp    = "";
   out_http_ms = 0;
   out_tx      = 0;
   out_rx      = 0;
 
   wifi_connect();
 
   WiFiClient cli;
   cli.setTimeout(30000);  
 
   unsigned long t0 = millis();
 
   if (!cli.connect(host, port)) {
     out_resp = "connect_fail";
     return false;
   }

   String boundary = "----m5bound7x";

   String part_meta =
     "--" + boundary + "\r\n"
     "Content-Disposition: form-data; name=\"meta\"\r\n"
     "Content-Type: application/json\r\n\r\n"
     + meta_json + "\r\n";

   String part_img_hdr =
     "--" + boundary + "\r\n"
     "Content-Disposition: form-data; name=\"image\"; filename=\"" + image_id + "\"\r\n"
     "Content-Type: image/jpeg\r\n\r\n";

   String part_end = "\r\n--" + boundary + "--\r\n";

   size_t content_length = part_meta.length()
                         + part_img_hdr.length()
                         + img_size
                         + part_end.length();

   String hdr;
   hdr.reserve(300);
   hdr += "POST ";
   hdr += endpoint;
   hdr += " HTTP/1.1\r\nHost: ";
   hdr += host;
   hdr += "\r\nConnection: close\r\n";
   hdr += "Content-Type: multipart/form-data; boundary=";
   hdr += boundary;
   hdr += "\r\nContent-Length: ";
   hdr += String((unsigned long)content_length);
   hdr += "\r\n\r\n";
 
   cli.print(hdr);
   cli.print(part_meta);
   cli.print(part_img_hdr);
 
   {
     const size_t CHUNK = 4096;
     const uint8_t* ptr = img;
     size_t remaining = img_size;
     int stall = 0;
 
     while (remaining > 0) {
       size_t to_write = (remaining > CHUNK) ? CHUNK : remaining;
       size_t written = cli.write(ptr, to_write);
 
       if (written > 0) {
         ptr       += written;
         remaining -= written;
         stall = 0;
       } else {
         stall++;
         if (stall > 2000) {   
           cli.stop();
           out_resp = "img_write_stall";
           out_http_ms = millis() - t0;
           return false;
         }
         delay(10);
       }
       if ((img_size - remaining) % (64 * 1024) < CHUNK) {
         yield();
       }
     }
   }
 
   cli.print(part_end);
   out_tx = hdr.length() + content_length;   
 
   unsigned long send_ms = millis() - t0;
 
   yield();
   delay(50);                
   cli.setTimeout(5000);     
 
   unsigned long deadline = millis() + 10000; 
   while (!cli.available() && millis() < deadline) delay(10);
 
   if (!cli.available()) {
     cli.stop();
     out_resp = "response_timeout";
     out_http_ms = millis() - t0;
     return false;
   }
 
   String status_line = cli.readStringUntil('\n');
   status_line.trim();
   if (status_line.startsWith("HTTP/1.")) {
     int sp1 = status_line.indexOf(' ');
     int sp2 = status_line.indexOf(' ', sp1 + 1);
     if (sp1 > 0) {
       String cs = (sp2 > sp1)
         ? status_line.substring(sp1 + 1, sp2)
         : status_line.substring(sp1 + 1);
       out_code = cs.toInt();
     }
   }
 
   out_rx = status_line.length() + 1;
   while (cli.available()) {
     String line = cli.readStringUntil('\n');
     out_rx += line.length() + 1;
     line.trim();
     if (line.length() == 0) break; 
   }
   String body = "";
   unsigned long body_dl = millis() + 3000;
   while (millis() < body_dl) {
     if (cli.available()) {
       char c = (char)cli.read();
       body += c;
       out_rx++;
       if (c == '}') break;
     } else {
       delay(5);
     }
   }
   out_resp = status_line;
   cli.stop();
   out_http_ms = millis() - t0;
 
   return (out_code == 200);
 }
 
 // ============================================================
 //  HTTP PATCH
 // ============================================================
 static bool http_post_patch(const String& image_id,
                             unsigned long http_e2e_ms,
                             size_t tx_bytes,
                             size_t rx_bytes,
                             const char* host = PI_HOST,
                             int port = PI_PORT,
                             const char* endpoint = PI_ENDPOINT_PATCH) {
   wifi_connect();

   WiFiClient cli;
   cli.setTimeout(5000);

   if (!cli.connect(host, port)) return false;

   JsonDocument doc;
   doc["run_id"]             = RUN_ID;
   doc["device"]             = DEVICE_ID;
   doc["image_id"]           = image_id;
   doc["m5_pi_http_e2e_ms"]  = (unsigned long)http_e2e_ms;
   doc["m5_pi_tx_bytes"]     = (unsigned long)tx_bytes;
   doc["m5_pi_rx_bytes"]     = (unsigned long)rx_bytes;
   doc["ts"]                 = now_kst_iso_ms();

   String payload;
   serializeJson(doc, payload);

   String req;
   req.reserve(payload.length() + 200);
   req += "POST ";
   req += endpoint;
   req += " HTTP/1.1\r\nHost: ";
   req += host;
   req += "\r\nConnection: close\r\n";
   req += "Content-Type: application/json\r\n";
   req += "Content-Length: ";
   req += String((unsigned long)payload.length());
   req += "\r\n\r\n";
 
   cli.print(req);
   cli.print(payload);
 
   unsigned long dl = millis() + 5000;
   while (!cli.available() && millis() < dl) delay(10);
   String sl = cli.readStringUntil('\n');
 
   cli.stop();
   return (sl.indexOf("200") >= 0);
 }
 
static void append_failed_id(const String& image_id, const char* role, const char* reason) {
  if (!g_sd_write_enabled) return;

  JsonDocument doc;
   doc["run_id"] = RUN_ID;
   doc["device"] = DEVICE_ID;
   doc["role"] = role;
   doc["image_id"] = image_id;
   doc["reason"] = (reason && reason[0] != '\0') ? reason : "unknown";
   doc["ts"] = now_kst_iso_ms();
   doc["push_ms"] = (unsigned long)g_last_push_ms;
   doc["push_kbps"] = (unsigned long)g_last_push_kbps;
 
   String line;
   serializeJson(doc, line);
   line += "\n";
 
   File f = SD.open(g_failed_ids_path.c_str(), FILE_WRITE);
   if (!f) return;
   f.seek(f.size());
   f.print(line);
   f.flush();
   f.close();
 }
 
 static bool load_retry_targets(std::vector<String>& out_ids) {
   out_ids.clear();
   if (!SD.exists(g_failed_ids_path.c_str())) return false;
 
   File f = SD.open(g_failed_ids_path.c_str(), FILE_READ);
   if (!f) return false;
 
   while (f.available()) {
     String line = f.readStringUntil('\n');
     line.trim();
     if (line.length() == 0) continue;
 
     String image_id = "";
     if (line.startsWith("{")) {
       JsonDocument d;
       DeserializationError e = deserializeJson(d, line);
       if (!e && d["image_id"].is<const char*>()) {
         image_id = String((const char*)d["image_id"]);
       }
     } else {
       image_id = line;
     }
     image_id = basename_of(image_id);
     image_id.trim();
     if (image_id.length() == 0) continue;
     if (std::find(out_ids.begin(), out_ids.end(), image_id) == out_ids.end()) {
       out_ids.push_back(image_id);
     }
   }
   f.close();
   return !out_ids.empty();
 }
 
 static void filter_files_by_retry_targets() {
   if (!SD.exists(g_retry_only_flag_path.c_str())) return;
 
   if (!load_retry_targets(g_retry_targets)) return;
 
   std::vector<String> filtered;
   filtered.reserve(g_retry_targets.size());
   for (size_t i = 0; i < g_files.size(); i++) {
     String id = basename_of(g_files[i]);
     if (std::find(g_retry_targets.begin(), g_retry_targets.end(), id) != g_retry_targets.end()) {
       filtered.push_back(g_files[i]);
     }
   }
   g_files.swap(filtered);
 }
 
 // ============================================================
 //  Process one image
 // ============================================================
 enum ProcessResult {
   PROCESS_OK = 0,
   PROCESS_RETRYABLE_FAIL = 1,
   PROCESS_SKIP_FAIL = 2,
 };
 

// ============================================================
//  Process one image (시리얼 청소 및 선제적 복구 로직 포함)
// ============================================================
 static ProcessResult process_one(int idx) {
  int total    = (int)g_files.size();
  String fp    = g_files[idx];
  String imgid = basename_of(fp);
  Serial.printf("[PROC] %s idx=%d/%d file=%s\n",
                DEVICE_ID, idx + 1, total, fp.c_str());

  bool isA = (String(DEVICE_ID) == "M5_A");
  bool isB = (String(DEVICE_ID) == "M5_B");

  const char* role   = isA ? "a" : (isB ? "b" : "c");
  const char* prompt_primary = isA ? PROMPT_TEXT_A : (isB ? PROMPT_TEXT_B : PROMPT_TEXT_C);
  const char* prompt_backup  = isA ? PROMPT_TEXT_A_BACKUP : (isB ? PROMPT_TEXT_B_BACKUP : PROMPT_TEXT_C_BACKUP);
  const bool flip_prompt_order = ((idx & 1) == 1);

  M5.Display.fillScreen(BLACK);
  M5.Display.setCursor(0, 0);
  M5.Display.setTextSize(2);
  M5.Display.printf("[%s] %d/%d\n%s\n", DEVICE_ID, idx + 1, total, imgid.c_str());
  M5.Display.print("SD Read...");

  unsigned long t_start = millis();

  // 1. SD 카드에서 이미지 읽기
  size_t fsz = 0;
  unsigned long sd_ms = 0;
  String rd_err;
  if (!read_image_to_buffer(fp, jpg_buffer, MAX_JPG_SIZE, fsz, sd_ms, rd_err)) {
    if (rd_err == "size_bad" || rd_err == "too_small" || rd_err == "jpg_magic_bad") {
      return PROCESS_SKIP_FAIL;
    }
    return PROCESS_RETRYABLE_FAIL;
  }

  M5.Display.println("OK");

  // 2. VLM용 리사이징
  const uint8_t* vlm_img = jpg_buffer;
  size_t vlm_img_size = fsz;
  bool was_resized = false;

  unsigned long resize_ms = 0;
  if (vlm_jpg_buffer) {
    size_t resized_len = 0;
    unsigned long resize_t0 = millis();
    bool r_ok = resize_jpeg_for_vlm(jpg_buffer, fsz,
                                     vlm_jpg_buffer, VLM_JPG_BUF_SIZE, resized_len);
    resize_ms = millis() - resize_t0;
    if (r_ok && resized_len > 0) {
      vlm_img = vlm_jpg_buffer;
      vlm_img_size = resized_len;
      was_resized = true;
    }
  }

  M5.Display.print("VLM Infer...");

  // -- 추론 직전: UART 청소 + baud 확인 + 모듈 상태 점검 --
  while (Serial2.available()) Serial2.read();
  Serial2.flush();

  bool uart_ok = ensure_uart_921600();
  // checkConnection() is noisy right after heavy UART traffic; use UART readiness as primary precheck.
  bool llm_ok = uart_ok;
  if (!llm_ok) {
    g_precheck_fail_streak++;
    Serial.printf("[CLEAN] precheck fail (streak=%d) uart_ok=%d llm_ok=%d\n",
                  g_precheck_fail_streak, (int)uart_ok, (int)llm_ok);

    // 매 실패마다 하드복구를 돌리면 장시간 블로킹되므로, 간헐적으로만 복구 시도
    if ((g_precheck_fail_streak % PRECHECK_RECOVER_EVERY) != 0) {
      append_failed_id(imgid, role, "precheck_not_ready");
      return PROCESS_RETRYABLE_FAIL;
    }

    Serial.println("[CLEAN] precheck recover attempt...");
    if (!recover_module(vlm_img, vlm_img_size)) {
      append_failed_id(imgid, role, "recover_failed_precheck");
      return PROCESS_RETRYABLE_FAIL;
    }
    g_precheck_fail_streak = 0;
  } else {
    g_precheck_fail_streak = 0;
  }

  // 3. VLM 추론 진행 (0312 스타일 단순 루프 + 구조화 cat/sev/conf 파싱)
  String text = "";
  unsigned long vlm_roundtrip_ms = 0;
  unsigned long uart_ms = 0;
  unsigned long vlm_infer_ms = 0;
  unsigned long recover_ms = 0;
  unsigned long postprocess_ms = 0;
  String cat = "";
  int sev = -1;
  int conf = -1;
  bool vlm_ok = false;
  String fail_reason = "";
  int timeout_empty_streak_img = 0;
  int infer_attempts = 0;
  const int tries = RETRY_VLM + 1;

  for (int t = 0; t < tries; t++) {
    infer_attempts++;
    wdt_kick();
    const char* attempt_prompt = (t == 0) ? prompt_primary : prompt_backup;
    if (flip_prompt_order) {
      attempt_prompt = (t == 0) ? prompt_backup : prompt_primary;
    }
    String out;
    unsigned long ms = 0;
    unsigned long q_uart_ms = 0;
    unsigned long q_infer_ms = 0;
    bool q_ok = vlm_query(vlm_img, vlm_img_size, attempt_prompt, role, out, ms, q_uart_ms, q_infer_ms);
    vlm_roundtrip_ms += ms;
    uart_ms += q_uart_ms;
    vlm_infer_ms += q_infer_ms;
    wdt_kick();

    if (q_ok) {
      unsigned long pp_t0 = millis();
      String raw_out = out;

      // --- 구조화 필드 파싱 (cat/sev/conf) ---
      VlmParsed parsed;
      parse_structured_output(out, role, parsed);

      // --- 텍스트 추출 및 후처리 (0312 스타일: 단순) ---
      // text: 필드가 있으면 그것을 사용, 없으면 raw output에서 label 제거
      String proc_text = "";
      if (parsed.ok_text && parsed.text.length() > 0) {
        proc_text = parsed.text;
      } else {
        proc_text = extract_unlabeled_text_block(out);
        if (proc_text.length() == 0) proc_text = out;
      }

      postprocess_text_output(proc_text, role);
      if (isA) {
        sanitize_a_final(proc_text, raw_out, imgid);
        ensure_a_two_sentences(proc_text, parsed.ok_cat ? parsed.cat : "", parsed.ok_sev ? parsed.sev : 1);
      } else if (isB) {
        sanitize_b_text(proc_text);
      } else {
        sanitize_c_text(proc_text);
      }
      normalize_inline(proc_text, 500);
      proc_text.trim();

      // --- 텍스트 품질 체크 (0312 스타일: 단순) ---
      bool relevant_ok = isA ? is_disaster_relevant(proc_text) : true;
      bool usable = text_quality_ok(proc_text, role) &&
                    !is_garbage_text(proc_text) &&
                    !has_non_ascii(proc_text) &&
                    relevant_ok;

      // 텍스트 실패 시 salvage 시도
      if (!usable) {
        String rescued = "";
        if (salvage_vlm_text(raw_out, role, rescued)) {
          if (isA) {
            ensure_a_two_sentences(rescued, parsed.ok_cat ? parsed.cat : "", parsed.ok_sev ? parsed.sev : 1);
          }
          bool rescued_relevant = isA ? is_disaster_relevant(rescued) : true;
          if (text_quality_ok(rescued, role) && !is_garbage_text(rescued) &&
              !has_non_ascii(rescued) && rescued_relevant) {
            proc_text = rescued;
            usable = true;
          }
        }
      }

      // A: 중복 로깅만 (reject 하지 않음 — 1B 모델 특성상 유사 패턴 불가피)
      if (usable && isA) {
        String parts_chk[3];
        int sc = split_sentences(proc_text, parts_chk, 3);
        if (sc >= 1) {
          String s1 = parts_chk[0]; s1.trim();
          if (a_s1_too_repetitive(s1) || is_duplicate_recent_a_s1(s1)) {
            Serial.printf("[A] duplicate_s1_warn id=%s (accepted)\n", imgid.c_str());
          }
        }
      }

      postprocess_ms += millis() - pp_t0;

      if (usable) {
        text = proc_text;
        // cat/sev: 항상 텍스트 기반 분류 (프롬프트에 구조화 필드 없음)
        classify_from_text(text, cat, sev);
        if (sev > 2) sev = 2;
        if (sev < 0) sev = 0;
        // conf: 텍스트 specificity 기반 계산
        conf = compute_conf_from_text(text, cat, sev);
        vlm_ok = true;
        g_infer_fail_streak = 0;
        g_timeout_streak_global = 0;
        Serial.printf("[OK] %s cat=%s sev=%d conf=%d text_len=%d\n",
                      role, cat.c_str(), sev, conf, (int)text.length());
        break;
      }

      if (fail_reason.length() == 0) fail_reason = "quality_fail";
      delay(COOL_BAD_MS);
      wdt_kick();
      continue;
    }

    // VLM query 자체 실패 (timeout, empty, UART 등)
    fail_reason = (g_last_vlm_fail_reason.length() > 0) ? g_last_vlm_fail_reason : "vlm_fail";
    Serial.printf("[VLM] fail role=%s try=%d/%d reason=%s ms=%lu\n",
                  role, t + 1, tries, fail_reason.c_str(), ms);
    bool was_timeout = (fail_reason == "timeout_empty" || fail_reason == "empty_text");
    if (was_timeout) {
      timeout_empty_streak_img++;
      g_timeout_streak_global++;
    }

    if (t + 1 < tries) {
      bool need_recover = false;
      if (was_timeout) {
        need_recover = (timeout_empty_streak_img >= TIMEOUT_RESET_STREAK);
      } else if (fail_reason == "uart_link_slow" || fail_reason == "vlm_fail") {
        need_recover = true;
      }

      if (need_recover) {
        // Do not block inside per-image retry path. Recovery is deferred to precheck
        // of subsequent images to keep the main loop advancing overnight.
        bool quick_ok = true;
        if (fail_reason == "uart_link_slow") {
          quick_ok = ensure_uart_921600();
        }
        if (!quick_ok) {
          g_precheck_fail_streak = PRECHECK_RECOVER_EVERY - 1;
        }
        Serial.printf("[RECOVER] deferred to precheck (reason=%s try=%d/%d)\n",
                      fail_reason.c_str(), t + 1, tries);
      } else {
        Serial.printf("[RECOVER] skip for reason=%s (try=%d/%d)\n",
                      fail_reason.c_str(), t + 1, tries);
      }
    }
    if (was_timeout || fail_reason == "uart_link_slow" || fail_reason == "vlm_fail") {
      break;
    }
    delay(COOL_BAD_MS);
    wdt_kick();
  }

  if (!vlm_ok) {
    if (fail_reason == "timeout_empty" || fail_reason == "empty_text" ||
        fail_reason == "uart_link_slow" || fail_reason == "vlm_fail") {
      unsigned long r_t0 = millis();
      bool r_ok = recover_module(vlm_img, vlm_img_size);
      recover_ms += millis() - r_t0;
      if (!r_ok) {
        Serial.printf("[RECOVER] failed after %s, continue without board restart\n", fail_reason.c_str());
      }
    }
    M5.Display.println("FAIL");
    Serial.printf("[FAIL] role=%s id=%s reason=%s\n", role, imgid.c_str(), fail_reason.c_str());
    append_failed_id(imgid, role, fail_reason.c_str());
    g_infer_fail_streak++;
    return PROCESS_RETRYABLE_FAIL;
  } else {
    M5.Display.println("OK");
    g_infer_fail_streak = 0;
  }
 
  unsigned long m5_total_ms = millis() - t_start;
 
  // 5. 결과 JSON 생성 및 전송 (모든 role에 cat/sev/conf 포함)
  JsonDocument doc;
  doc["run_id"]   = RUN_ID;
  doc["fw"]       = FW_TAG;
  doc["device"]   = DEVICE_ID;
  doc["image_id"] = imgid;
  doc["role"]     = role;
  doc["status"]   = vlm_ok ? "ok" : "vlm_fail";

  // text: A는 text 출력, B/C는 text 있으면 포함 (CoT)
  if (vlm_ok && text.length() > 0) {
    doc["text"] = text;
  } else {
    doc["text"] = JsonVariant();
  }

  // cat/sev/conf: 모든 role (agreement 판정을 위해)
  if (vlm_ok) {
    doc["cat"]  = cat;
    doc["sev"]  = sev;
    doc["conf"] = conf;
  } else {
    doc["cat"]  = JsonVariant();
    doc["sev"]  = JsonVariant();
    doc["conf"] = JsonVariant();
  }

  doc["ts"] = now_kst_iso_ms();
  if (!vlm_ok) doc["fail_reason"] = fail_reason;
  doc["uart_push_ms"] = (unsigned long)g_last_push_ms;
  doc["retry_count"] = (infer_attempts > 0) ? (infer_attempts - 1) : 0;

  JsonObject lat = doc["latency"].to<JsonObject>();
  lat["m5_sd_read_ms"]       = (unsigned long)sd_ms;
  lat["m5_decode_resize_ms"] = (unsigned long)resize_ms;
  lat["m5_uart_ms"]          = (unsigned long)uart_ms;
  lat["m5_vlm_infer_ms"]     = (unsigned long)vlm_infer_ms;
  lat["m5_postprocess_ms"]   = (unsigned long)postprocess_ms;
  lat["m5_recover_ms"]       = (unsigned long)recover_ms;
  lat["m5_vlm_roundtrip_ms"] = (unsigned long)vlm_roundtrip_ms;
  lat["m5_total_ms"]  = (unsigned long)m5_total_ms;

  JsonObject sig = doc["signal_overhead_bytes"].to<JsonObject>();
  sig["m5_pi_tx_bytes"] = JsonVariant();
  sig["m5_pi_rx_bytes"] = JsonVariant();
  sig["m5_pi_http_e2e_ms"] = JsonVariant();

  JsonObject pwr = doc["power"].to<JsonObject>();
  pwr["energy_j"] = JsonVariant();
  pwr["power_w"] = JsonVariant();
 
  doc["image_bytes"] = (unsigned long)fsz;   
  if (was_resized) doc["vlm_image_bytes"] = (unsigned long)vlm_img_size;
 
  String meta_json;
  serializeJson(doc, meta_json);
 
  int    http_code = -1;
  String resp;
  unsigned long http_ms = 0;
  size_t tx = 0, rx = 0;
  bool   sent = false;
 
  M5.Display.print("Pi_A...");
  // ---- Pi_A 전송 (primary) ----
  for (int r = 0; r < RETRY_NET; r++) {
    wdt_kick();
    sent = http_post_multipart(meta_json, jpg_buffer, fsz, imgid,
                               http_code, resp, http_ms, tx, rx,
                               PI_HOST, PI_PORT, PI_ENDPOINT_INGEST);
    if (sent) break;
    wifi_recover_hard();
    wdt_kick();
  }

  if (!sent) {
    M5.Display.println("FAIL");
    g_consecutive_fails++;
    return PROCESS_RETRYABLE_FAIL;
  }
  M5.Display.println("OK");
  g_consecutive_fails = 0;

  // Pi_A E2E PATCH (retry up to 3 times - 259/914 failures in pilot run)
  for (int pr = 0; pr < 3; pr++) {
    if (http_post_patch(imgid, http_ms, tx, rx, PI_HOST, PI_PORT, PI_ENDPOINT_PATCH)) break;
    delay(200);
    wdt_kick();
  }

  Serial.printf("[%s] %s role=%s cat=%s sev=%d conf=%d infer=%lums pi_a_http=%lums img=%uKB\n",
                DEVICE_ID, imgid.c_str(), role,
                cat.c_str(), sev, conf,
                vlm_infer_ms, http_ms, (unsigned)(fsz / 1024));

  // ---- Pi_B 전송 (sequential, latency 측정 독립) ----
  wdt_kick();
  M5.Display.print("Pi_B...");
  int    pib_code = -1;
  String pib_resp;
  unsigned long pib_http_ms = 0;
  size_t pib_tx = 0, pib_rx = 0;
  bool   pib_sent = false;

  for (int r = 0; r < RETRY_NET; r++) {
    wdt_kick();
    pib_sent = http_post_multipart(meta_json, jpg_buffer, fsz, imgid,
                                   pib_code, pib_resp, pib_http_ms, pib_tx, pib_rx,
                                   PIB_HOST, PIB_PORT, PI_ENDPOINT_INGEST);
    if (pib_sent) break;
    delay(500);
    wdt_kick();
  }

  if (pib_sent) {
    M5.Display.println("OK");
    for (int pr = 0; pr < 3; pr++) {
      if (http_post_patch(imgid, pib_http_ms, pib_tx, pib_rx, PIB_HOST, PIB_PORT, PI_ENDPOINT_PATCH)) break;
      delay(200);
      wdt_kick();
    }
    Serial.printf("[%s] Pi_B %s http=%lums tx=%u rx=%u\n",
                  DEVICE_ID, imgid.c_str(), pib_http_ms, (unsigned)pib_tx, (unsigned)pib_rx);
  } else {
    M5.Display.println("FAIL(skip)");
    Serial.printf("[%s] Pi_B send FAIL %s (continue)\n", DEVICE_ID, imgid.c_str());
    // Pi_B 실패해도 전체 파이프라인은 계속 진행 (Pi_A 성공이 기준)
  }

  g_last_progress_ms = millis();

  if (vlm_ok && text.length() > 0) remember_recent_text_ac(text, role);
 
  return PROCESS_OK;
} 
 

 
 // ============================================================
 //  tier1_init
 // ============================================================
 void tier1_init() {
   M5.Display.fillScreen(BLACK);
   M5.Display.setCursor(0, 0);
   M5.Display.setTextSize(2);
   M5.Display.println("[INIT] Booting...");
   Serial.printf("[FW] %s (%s %s)\n", FW_TAG, __DATE__, __TIME__);

   M5.Display.print("WiFi...");
   wifi_connect();
   M5.Display.println("OK");

   bool ntp_ok = ntp_sync_kst();
   Serial.printf("[NTP] %s\n", ntp_ok ? "synced" : "fail(will retry later)");

   M5.Display.print("SD Card...");
   SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
   if (!SD.begin(SD_CS, SPI, 15000000)) {
     M5.Display.println("FAIL!");
     Serial.println("[FATAL] SD mount failed -- check card");
     while (1) delay(1000);
   }
   M5.Display.println("OK");

   M5.Display.print("Scan imgs...");
   if (!scan_recursive(String(SCAN_ROOT), g_files)) {
     M5.Display.println("FAIL!");
     Serial.printf("[FATAL] no images in %s\n", SCAN_ROOT);
     while (1) delay(5000);
   }
   std::sort(g_files.begin(), g_files.end());
   filter_files_by_retry_targets();
   M5.Display.printf("OK (%d)\n", (int)g_files.size());
   Serial.printf("[SCAN] %d images in %s\n", (int)g_files.size(), SCAN_ROOT);

   jpg_buffer = (uint8_t*)ps_malloc(MAX_JPG_SIZE);
   if (!jpg_buffer) jpg_buffer = (uint8_t*)malloc(MAX_JPG_SIZE);
   if (!jpg_buffer) {
     M5.Display.println("MEM FAIL!");
     Serial.println("[FATAL] jpg_buffer alloc failed");
     while (1) delay(1000);
   }

   vlm_jpg_buffer = (uint8_t*)ps_malloc(VLM_JPG_BUF_SIZE);
   if (!vlm_jpg_buffer) vlm_jpg_buffer = (uint8_t*)malloc(VLM_JPG_BUF_SIZE);

   // ESP.restart() 후 모듈이 이미 921600으로 동작 중일 수 있음
   int rxd = M5.getPin(m5::pin_name_t::port_c_rxd);
   int txd = M5.getPin(m5::pin_name_t::port_c_txd);

   M5.Display.print("LLM Module...");
   bool already_fast = false;
   bool soft_connected = false;

   // 1단계: 921600으로 먼저 시도 (선제적 리부팅 후)
   Serial.println("[LLM] Try 921600 first (module may already be running)...");
   M5.Display.print("921k.");
   Serial2.begin(921600, SERIAL_8N1, rxd, txd);
   module_llm.begin(&Serial2);
  for (int r = 0; r < INIT_TRY_921_LOOPS; r++) {  // 0.5s * INIT_TRY_921_LOOPS
     if (module_llm.checkConnection()) {
       already_fast = true;
       soft_connected = true;
       break;
     }
     if (r % 10 == 0) {
       Serial.printf("[LLM] 921600 wait... %ds\n", r / 2);
     }
     delay(500);
   }

   // 2단계: 921600 실패 → 115200으로 전환 (콜드 부팅)
   if (!already_fast) {
     Serial.println("[LLM] 921600 failed. Trying 115200 (cold boot)...");
     M5.Display.print("115k.");
     Serial2.end();
     delay(50);
     Serial2.begin(115200, SERIAL_8N1, rxd, txd);
     module_llm.begin(&Serial2);
    for (int r = 0; r < INIT_TRY_115_LOOPS; r++) {  // 0.5s * INIT_TRY_115_LOOPS
       if (module_llm.checkConnection()) {
         soft_connected = true;
         break;
       }

       // 115200 대기 중에도 921600 재프로브를 섞어서 baud mismatch 교착을 회피
       if ((r % 8) == 7) {
         Serial2.end();
         delay(30);
         Serial2.begin(921600, SERIAL_8N1, rxd, txd);
         module_llm.begin(&Serial2);
         delay(60);
         if (module_llm.checkConnection()) {
           already_fast = true;
           soft_connected = true;
           break;
         }
         Serial2.end();
         delay(30);
         Serial2.begin(115200, SERIAL_8N1, rxd, txd);
         module_llm.begin(&Serial2);
       }

       if (r % 10 == 0) {
         Serial.printf("... %ds\n", r / 2);
         M5.Display.printf("%ds.", r / 2);
       }
       delay(500);
     }
   }

   if (soft_connected && already_fast) {
     // 이미 921600으로 연결됨 (선제적 리부팅 후)
     M5.Display.println("OK(fast)");
     Serial.println("[LLM] Already at 921600. Resetting module state...");

     // 이전 VLM 세션 정리 → 모듈 소프트 리셋
     M5.Display.print("Reset...");
     module_llm.sys.reset(true);  // 모듈 내부 상태 초기화, 완료 대기
     delay(500);
     M5.Display.println("OK");

     // 리셋 후 다시 연결 확인 (리셋하면 115200으로 돌아갈 수 있음)
     bool reconnected = false;
     // 먼저 921600 확인
     for (int r = 0; r < 6; r++) {
       if (module_llm.checkConnection()) { reconnected = true; break; }
       delay(500);
     }
     if (!reconnected) {
       // 리셋 후 115200으로 돌아갔을 수 있음
       Serial.println("[LLM] Post-reset: trying 115200...");
       Serial2.end();
       delay(50);
       Serial2.begin(115200, SERIAL_8N1, rxd, txd);
       module_llm.begin(&Serial2);
       for (int r = 0; r < 20; r++) {  // 10초 대기
         if (module_llm.checkConnection()) { reconnected = true; break; }
         delay(500);
       }
       if (reconnected) {
         // 다시 921600으로 업그레이드
         module_llm.sys.setBaudRate(921600);
         delay(100);
         Serial2.updateBaudRate(921600);
         delay(100);
       }
     }

     if (!reconnected) {
       M5.Display.println("RECONNECT FAIL!");
       Serial.println("[WARN] Post-reset reconnect failed -> try recover_module");
       if (!recover_module(nullptr, 0)) {
         Serial.println("[WARN] recover_module failed after reconnect fail; continue");
       }
     }

     if (!module_llm.checkConnection()) {
       Serial.println("[WARN] LLM not connected after warm reconnect; skip setup for now");
       M5.Display.println("LLM not ready");
       vlm_work_id = "vlm_work";
     } else {
       Serial.println("[LLM] Setting up VLM...");
       M5.Display.print("VLM...");
       m5_module_llm::ApiVlmSetupConfig_t cfg;
       cfg.model = VLM_MODEL_ID;
       cfg.max_token_len = 300;
       vlm_work_id = module_llm.vlm.setup(cfg);
       vlm_work_id.trim();
       if (vlm_work_id.length() == 0) vlm_work_id = "vlm_work";
       M5.Display.println("OK");
       Serial.printf("[LLM] Ready. work_id=%s (921600, warm)\n", vlm_work_id.c_str());
     }
   } else if (soft_connected) {
     M5.Display.println("OK");
     // 115200에서 연결 → 921600 업그레이드
     Serial.println("[LLM] Connected at 115200. Upgrading to 921600...");
     M5.Display.print("Baud 921600...");
     module_llm.sys.setBaudRate(921600);
     delay(100);
     Serial2.updateBaudRate(921600);
     delay(100);

     if (!module_llm.checkConnection()) {
       Serial.println("[WARN] LLM not connected after baud upgrade; skip setup for now");
       M5.Display.println("LLM not ready");
       vlm_work_id = "vlm_work";
     } else {
       m5_module_llm::ApiVlmSetupConfig_t cfg;
       cfg.model = VLM_MODEL_ID;
       cfg.max_token_len = 300;
       vlm_work_id = module_llm.vlm.setup(cfg);
       vlm_work_id.trim();
       if (vlm_work_id.length() == 0) vlm_work_id = "vlm_work";
       M5.Display.println("OK");
       Serial.printf("[LLM] Ready. work_id=%s (921600 baud)\n", vlm_work_id.c_str());
     }
  } else {
    M5.Display.println("FAIL");
    M5.Display.println("Recover...");
    // 소프트 연결 실패 → 하드 복구 시도
    Serial.println("[LLM] Soft connect failed. Trying hard power cycle...");
    bool ready = recover_module(nullptr, 0);
    if (!ready) {
      M5.Display.println("LLM init fail");
      Serial.println("[WARN] LLM module init failed, continue and retry in loop");
    }
    else M5.Display.println("Recovered OK");
  }

  // checkpoint/failed_ids 경로를 먼저 확정한다.
  g_out_dir_path = String(OUT_DIR);
  g_failed_ids_path = String(FAILED_IDS_PATH);
  g_retry_only_flag_path = String(RETRY_ONLY_FLAG_PATH);

  // mkdir 시도 → 실패하면 root "/" 로 fallback (A/C SD카드는 mkdir 불가)
  bool out_ok = ensure_dir(g_out_dir_path.c_str());
  if (!out_ok && g_out_dir_path != "/") {
    Serial.printf("[WARN] mkdir fail %s -> fallback to /\n", g_out_dir_path.c_str());
    g_out_dir_path = "/";
    g_failed_ids_path = "/failed_ids.jsonl";
    g_retry_only_flag_path = "/retry_failed_only.flag";
  }
  g_sd_write_enabled = true;

  // SD 쓰기 가능 여부 확인 (probe)
  String probe_path = join_sd_path(g_out_dir_path, "write_probe.tmp");
  if (!probe_sd_write_capability(probe_path)) {
    Serial.printf("[WARN] SD write probe failed at %s\n", probe_path.c_str());
    // root에서도 한 번 더 시도
    if (g_out_dir_path != "/") {
      g_out_dir_path = "/";
      probe_path = "/write_probe.tmp";
      g_sd_write_enabled = probe_sd_write_capability(probe_path);
    } else {
      g_sd_write_enabled = false;
    }
    if (!g_sd_write_enabled) {
      Serial.println("[WARN] SD write disabled -> checkpoint will not persist");
    }
  }

  // 부팅 직후 1회 probe 추론으로 VLM 세션 워밍업/검증
  // LLM이 연결되지 않은 상태에서는 probe를 건너뛰어 init 프리즈를 방지
  if (!g_files.empty() && module_llm.checkConnection()) {
    size_t probe_sz = 0;
    unsigned long probe_sd_ms = 0;
    String probe_err;
    const String& probe_fp = g_files[0];
    bool probe_loaded = read_image_to_buffer(probe_fp, jpg_buffer, MAX_JPG_SIZE,
                                             probe_sz, probe_sd_ms, probe_err);
    if (probe_loaded) {
      String probe_id = basename_of(probe_fp);
      Serial.printf("[LLM] init probe image=%s size=%uKB\n",
                    probe_id.c_str(), (unsigned)(probe_sz / 1024));
      bool probe_ok = llm_quick_infer_check(jpg_buffer, probe_sz);
      if (!probe_ok) {
        Serial.println("[LLM] init probe failed -> recovering module...");
        if (!recover_module(jpg_buffer, probe_sz)) {
          Serial.println("[WARN] init probe recover failed, continue and retry in loop");
        }
      }
    } else {
      Serial.printf("[LLM] init probe skipped (read fail: %s)\n", probe_err.c_str());
      llm_quick_infer_check(nullptr, 0);
    }
  } else {
    if (!module_llm.checkConnection()) {
      Serial.println("[LLM] init probe skipped (LLM not connected)");
    } else {
      llm_quick_infer_check(nullptr, 0);
    }
  }
 
   Serial.println("========================================");
   Serial.printf("  DEVICE : %s\n", DEVICE_ID);
   Serial.printf("  IMAGES : %d\n", (int)g_files.size());
   Serial.printf("  Pi     : %s:%d%s\n", PI_HOST, (int)PI_PORT, PI_ENDPOINT_INGEST);
   Serial.printf("  PATCH  : %s:%d%s\n", PI_HOST, (int)PI_PORT, PI_ENDPOINT_PATCH);
   Serial.printf("  RUN_ID : %s\n", RUN_ID);
   Serial.printf("  MAX_IMG: %uKB\n", (unsigned)(MAX_JPG_SIZE / 1024));
   Serial.printf("  OUTDIR : %s\n", g_out_dir_path.c_str());
   Serial.printf("  FAILID : %s\n", g_failed_ids_path.c_str());
   Serial.println("========================================");
 }
 
 // ============================================================
 //  tier1_run

void tier1_run() {
  if (g_done) return;

  int total = (int)g_files.size();

  // ===== 강제 리셋: 0부터 시작하려면 아래 줄 주석 해제 후 빌드, 완료 후 다시 주석 =====
  //save_ckpt_nvs(DEVICE_ID, 0);

  // ===== NVS reset flag: SD카드에 /reset_ckpt.flag 파일이 있으면 지정 번호부터 시작 =====
  // 파일 내용이 숫자면 그 번호부터, 비어있거나 0이면 처음부터
  // 파일이 없으면 NVS 체크포인트(프리즈 번호)부터 이어서 시작
  if (SD.exists("/reset_ckpt.flag")) {
    int reset_to = 0;
    File rf = SD.open("/reset_ckpt.flag", FILE_READ);
    if (rf) {
      String s = rf.readString();
      rf.close();
      s.trim();
      if (s.length() > 0) reset_to = s.toInt();
    }
    if (reset_to < 0) reset_to = 0;
    Serial.printf("[CKPT-NVS] reset flag found -> set checkpoint to %d\n", reset_to);
    save_ckpt_nvs(DEVICE_ID, reset_to);
    // 삭제 실패 시 재시도 — 파일이 남아있으면 매 부팅마다 0으로 리셋되는 버그 방지
    for (int rm_try = 0; rm_try < 5; rm_try++) {
      if (SD.remove("/reset_ckpt.flag")) break;
      delay(50);
    }
    if (SD.exists("/reset_ckpt.flag")) {
      Serial.println("[WARN] reset_ckpt.flag remove failed! Will be re-applied on next boot.");
    }
  }

  // ===== NVS checkpoint (ESP32 내부 플래시) =====
  // NVS 무결성 probe: 임시 값 쓰고 읽어서 다르면 파티션 erase 후 재초기화
  {
    const int PROBE_VAL = 0xAB;
    bool ok = save_ckpt_nvs("_nvs_probe", PROBE_VAL);
    int probe_read = load_ckpt_nvs("_nvs_probe");
    if (!ok || probe_read != PROBE_VAL) {
      Serial.printf("[NVS] integrity check FAIL (ok=%d read=%d) -> erasing NVS partition\n",
                    (int)ok, probe_read);
      nvs_flash_erase();
      nvs_flash_init();
      Serial.println("[NVS] erased and re-initialized");
    } else {
      Serial.println("[NVS] integrity OK");
    }
  }

  int i = load_ckpt_nvs(DEVICE_ID);
  Serial.printf("[CKPT-NVS] device=%s loaded_idx=%d total=%d\n", DEVICE_ID, i, total);

  if (i < 0) i = 0;

  // LCD에 체크포인트 값 표시 (3초 대기)
  M5.Display.fillScreen(BLACK);
  M5.Display.setCursor(0, 0);
  M5.Display.setTextSize(2);
  M5.Display.printf("CKPT=%d/%d\n%s (NVS)", i, total, DEVICE_ID);
  delay(3000);

  if (i >= total) {
    // WDT가 아직 설정되지 않은 상태이므로 deregister 불필요
    // 단, 이전 부팅에서 WDT가 남아있을 수 있으므로 명시적으로 해제
    esp_task_wdt_deinit();
    M5.Display.fillScreen(BLACK);
    M5.Display.setCursor(0, 0);
    M5.Display.setTextSize(3);
    M5.Display.printf("[%s]\nDONE %d imgs", DEVICE_ID, total);
    g_done = true;
    while (1) delay(5000);  // WDT 없으므로 안전
  }

  Serial.printf("[SYSTEM] Starting from index %d (NVS ckpt, device=%s)\n", i, DEVICE_ID);
  g_last_progress_ms = millis();

  // 워치독 설정: 90초 무응답 시 강제 리부팅 (VLM 최대 75초 + 여유)
  esp_task_wdt_init(90, true);
  esp_task_wdt_add(NULL);
  g_wdt_armed = true;
  Serial.println("[WDT] Hardware watchdog enabled (90s)");

  std::vector<uint8_t> same_fail_count(total, 0);

  auto persist_ckpt = [&](int idx_to_write, const char* why) -> bool {
    if (idx_to_write < 0) idx_to_write = 0;
    if (idx_to_write > total) idx_to_write = total;
    bool ok = save_ckpt_nvs(DEVICE_ID, idx_to_write);
    if (!ok) {
      Serial.printf("[ERROR] NVS checkpoint save fail idx=%d (%s)\n", idx_to_write, why);
    }
    return ok;
  };

  for (; i < total; ) {
    wdt_kick();
    if (millis() - g_last_progress_ms > PROGRESS_RESTART_MS) {
      Serial.printf("[SYSTEM] no progress for %lus -> board restart\n",
                    PROGRESS_RESTART_MS / 1000UL);
      delay(500);
      ESP.restart();
    }
    wifi_connect();

    if (WIFI_RECYCLE_EVERY_N > 0 && i > 0 && (i % WIFI_RECYCLE_EVERY_N == 0)) {
      wifi_recover_hard();
    }

    // 처리 직전 현재 인덱스를 먼저 저장.
    // 프리즈/리셋이 여기서 발생해도 같은 이미지부터 재시작된다.
    if (!persist_ckpt(i, "pre_process")) {
      // SD write glitch가 있어도 추론 파이프라인은 계속 진행한다.
      // (즉시 재부팅하면 같은 이미지 반복 업로드가 발생할 수 있음)
      Serial.println("[WARN] pre_process checkpoint save failed -> continue");
    }

    ProcessResult res = process_one(i);

    if (res == PROCESS_OK || res == PROCESS_SKIP_FAIL) {
      same_fail_count[i] = 0;
      i++;
      if (!persist_ckpt(i, "commit_success")) {
        Serial.println("[WARN] commit_success checkpoint save failed -> continue");
      }
      g_last_progress_ms = millis();
      if (g_restart_requested) {
        Serial.printf("[SYSTEM] restarting after ckpt=%d: %s\n", i, g_restart_reason.c_str());
        delay(500);
        ESP.restart();
      }
      delay(COOL_OK_MS);
    }
    else {
      same_fail_count[i]++;
      Serial.printf("[RETRY] idx=%d/%d fail_count=%u/%d (keep_same_index)\n",
                    i + 1, total, same_fail_count[i], MAX_RETRY_SAME_INDEX);

      // retryable 실패는 스킵하지 않는다.
      // 동일 인덱스를 유지한 채 재시도하고, 일정 횟수 도달 시 보드 리부팅으로 복구.
      if (same_fail_count[i] >= MAX_RETRY_SAME_INDEX) {
        append_failed_id(basename_of(g_files[i]), "sys", "max_retry_reboot");
        same_fail_count[i] = 0;
        if (!persist_ckpt(i, "retryable_reboot_same_index")) {
          Serial.println("[WARN] retryable_reboot_same_index checkpoint save failed");
        }
        Serial.printf("[RETRY-HARD] idx=%d/%d image=%s -> reboot (same index)\n",
                      i + 1, total, basename_of(g_files[i]).c_str());
        delay(500);
        ESP.restart();
      } else {
        delay(1000);
      }
    }
  }

  if (!persist_ckpt(total, "all_done")) {
    Serial.println("[WARN] checkpoint save fail at all_done");
  }

  M5.Display.fillScreen(BLACK);
  M5.Display.setCursor(0, 0);
  M5.Display.setTextSize(3);
  M5.Display.printf("[%s]\nALL DONE\n%d imgs", DEVICE_ID, total);
  g_done = true;
  while (1) delay(5000);
}
