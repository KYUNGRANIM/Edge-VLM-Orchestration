#include <M5Unified.h>
#include "tier1.h"

void setup() {
  // ★ 부팅 삐리링 소리 끄기
  auto cfg = M5.config();
  cfg.internal_spk = false;
  M5.begin(cfg);

  // 혹시 남아있을 스피커 완전히 끄기
  M5.Speaker.end();

  Serial.begin(115200);
  delay(300);

  Serial.println("==============================");
  Serial.println(" M5 Oracle Pipeline (v2-http)");
  Serial.println("==============================");

  tier1_init();
}

void loop() {
  tier1_run();
  delay(1000);
}
