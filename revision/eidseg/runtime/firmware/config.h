#pragma once
#include <cstddef>

// ===== Device Identity (set per env in platformio.ini) =====
#ifndef DEVICE_NAME
  #define DEVICE_NAME "M5_Default"
#endif

// ===== WiFi =====
#ifndef WIFI_SSID
  #define WIFI_SSID "YOUR_WIFI_SSID"
#endif
#ifndef WIFI_PASS
  #define WIFI_PASS "YOUR_WIFI_PASSWORD"
#endif

// ===== Pi_A Ingest Server =====
#ifndef PI_HOST
  #define PI_HOST "192.168.0.60"
#endif
#ifndef PI_PORT
  #define PI_PORT 8000
#endif
#ifndef PI_ENDPOINT_INGEST
  #define PI_ENDPOINT_INGEST "/ingest_m5"
#endif
#ifndef PI_ENDPOINT_PATCH
  #define PI_ENDPOINT_PATCH "/patch_e2e"
#endif
#ifndef PI_COORD_PORT
  #define PI_COORD_PORT 8000
#endif
#ifndef PI_ENDPOINT_FINAL_ROUTE
  #define PI_ENDPOINT_FINAL_ROUTE "/ingest_final_route"
#endif

// ===== Pi_B Ingest Server =====
#ifndef PIB_HOST
  #define PIB_HOST "192.168.0.93"
#endif
#ifndef PIB_PORT
  #define PIB_PORT 8000
#endif

// ===== M5_A Stage-1 Coordinator =====
// M5_B/C send only lightweight JSON outputs to M5_A. M5_A then runs the
// Stage-1 router and uploads the image to Pi_A/Pi_B when escalation is needed.
// Set M5A_HOST to the actual WiFi IP of M5_A in platformio.ini or via build flags.
#ifndef M5A_HOST
  #define M5A_HOST "192.168.0.30"
#endif
#ifndef M5A_PORT
  #define M5A_PORT 8010
#endif
#ifndef M5A_ENDPOINT_AUX
  #define M5A_ENDPOINT_AUX "/m5_aux"
#endif

// 0 means M5_A waits indefinitely until both B/C outputs for the same image arrive.
static const unsigned long M5A_AUX_WAIT_MS = 0UL;
// 0 means B/C keep retrying the same already-computed output until M5_A accepts it.
static const unsigned long M5A_AUX_SEND_WAIT_MS = 0UL;

// ===== Run ID =====
#ifndef RUN_ID
  #define RUN_ID "eidseg_runtime600"
#endif
// Image index range [RUN_START, RUN_END) for one batch. Each newly flashed firmware starts at RUN_START.
#ifndef RUN_START
  #define RUN_START 0
#endif
#ifndef RUN_END
  #define RUN_END 100000
#endif

// ===== SD image path =====
// EIDSeg held-out 600 routing run (ids 000001-000600; all under 1.1 MB, below MAX_JPG_SIZE)
static const char* SCAN_ROOT = "/EIDSeg_eval/heldout600/earthquake";
static const char* OUT_DIR   = "/";
static const char* FAILED_IDS_PATH = "/failed_ids.jsonl";
static const char* RETRY_ONLY_FLAG_PATH = "/retry_failed_only.flag";

// ===== VLM Model =====
static const char* VLM_MODEL_ID = "internvl2.5-1b-364-ax630c";

// ===== Buffer =====
static const size_t MAX_JPG_SIZE = 5 * 1024 * 1024;  // 5MB

// ===== Retry / Cooldown =====
static const int RETRY_VLM = 1;
static const int RETRY_NET = 5;
static const int MAX_RETRY_SAME_INDEX = 5;
static const int MAX_DEFER_ROUNDS = 2;

// Base VLM wait timeout.
static const int VLM_WAIT_MS = 12000;
static const int VLM_WAIT_MAX_MS = 50000;

static const int COOL_OK_MS  = 5000;
static const int COOL_BAD_MS = 800;

// Recovery tuning.
static const int RECOVER_RESET_DELAY_MS = 8000;
static const int RECOVER_RETRY_DELAY_MS = 3000;
static const int RECOVER_FAIL_RESTART_THRESHOLD = 3;
static const int MAX_RECOVER_PER_IMAGE = 2;
static const int TIMEOUT_RESET_STREAK = 2;
static const unsigned long UART_MIN_PUSH_KBPS = 24;  // below this => likely 115200-like link
static const int RECOVER_FAIL_COOLDOWN_MS = 45000;
static const int PRECHECK_RECOVER_EVERY = 3;

// If false, A-role parse failure does NOT use text-only fallback (prevents generic repeated pseudo outputs).
static const bool ALLOW_A_PARSE_FALLBACK = false;

// 0 = disable size-based skip (collect all images, fail-closed on VLM timeout/fail).
static const size_t MAX_VLM_IMAGE_SIZE = 0;

// If true, do not emit rows with text=null.
// On VLM failure, retry same image (with restart recovery) until text is produced.
static const bool REQUIRE_VLM_TEXT_NONEMPTY = true;

// VLM input normalization target.
// - Keep original upload path to Pi unchanged.
// - Only VLM input is downscaled toward this long-side target.
static const int VLM_TARGET_LONG_SIDE = 640;
static const int VLM_TARGET_MIN_SHORT_SIDE = 300;
static const size_t VLM_JPG_BUF_SIZE = 512 * 1024;  // buffer for normalized VLM JPEG

// C role: if true, append risk phrase when human-impact wording is missing.
// Enabled to prevent C from drifting into plain scene description.
static const bool C_ENFORCE_HUMAN_RISK_AUGMENT = true;

// Init handshake tuning (0.5s per loop)
static const int INIT_TRY_921_LOOPS = 16;  // 8s
static const int INIT_TRY_115_LOOPS = 16;  // 8s

// If a device makes no forward progress for too long, force a board restart.
static const unsigned long PROGRESS_RESTART_MS = 180000UL;

// ===== Periodic LLM module reset =====
// 0 = 주기적 리셋 비활성화 (recover 과다 호출이 오히려 모듈 불안정 유발)
static const int RESET_EVERY_N = 0;
