#pragma once
#include <Arduino.h>

struct M5Stage1RouterInput {
  String m5a_cat;
  String m5b_cat;
  String m5c_cat;
  int m5a_sev;
  int m5b_sev;
  int m5c_sev;
  int m5a_conf;
  int m5b_conf;
  int m5c_conf;
};

// Returns true when the learned Stage-1 policy would END at M5.
//
// Important: the trained policy uses M5_A/B/C agreement features. In the
// current three-M5 collection architecture, a single M5 device does not have
// all three outputs locally. Call this only in a coordinator that has already
// received A/B/C outputs, or after adding inter-M5 exchange.
bool router_stage1_end_m5(const M5Stage1RouterInput& in);

