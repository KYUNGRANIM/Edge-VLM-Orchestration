#include "router_policy.h"

static float zscore(float v, float mean, float scale) {
  return (v - mean) / scale;
}

static float majority3(const String& a, const String& b, const String& c) {
  if (a == b && b == c) return 1.0f;
  if (a == b || a == c || b == c) return 2.0f / 3.0f;
  return 1.0f / 3.0f;
}

static float majority3_int(int a, int b, int c) {
  if (a == b && b == c) return 1.0f;
  if (a == b || a == c || b == c) return 2.0f / 3.0f;
  return 1.0f / 3.0f;
}

bool router_stage1_end_m5(const M5Stage1RouterInput& in) {
  const float m5_cat_agree = majority3(in.m5a_cat, in.m5b_cat, in.m5c_cat);
  const float m5_conf_mean = (in.m5a_conf + in.m5b_conf + in.m5c_conf) / 3.0f;
  const int min_sev = min(in.m5a_sev, min(in.m5b_sev, in.m5c_sev));
  const int max_sev = max(in.m5a_sev, max(in.m5b_sev, in.m5c_sev));
  const float m5_sev_range = (float)(max_sev - min_sev);
  const float m5_sev_agree = majority3_int(in.m5a_sev, in.m5b_sev, in.m5c_sev);

  const bool cat_earthquake = (in.m5a_cat == "earthquake");
  const bool cat_fire = (in.m5a_cat == "fire");
  const bool cat_hurricane = (in.m5a_cat == "hurricane");
  const bool cat_landslide = (in.m5a_cat == "landslide");
  const bool cat_other = (in.m5a_cat == "other_disaster");

  const float z_cat_agree = zscore(m5_cat_agree, 0.77180601f, 0.21209395f);
  const float z_conf_mean = zscore(m5_conf_mean, 2.48799614f, 1.03325038f);
  const float z_sev_agree = zscore(m5_sev_agree, 0.69127760f, 0.20032729f);
  const float z_m5a_sev = zscore((float)in.m5a_sev, 0.95150199f, 0.89908895f);

  // Exported from revision/first_submission/routers/stage1_m5_router.joblib.
  // class 1 = END@M5, class 0 = ESCALATE@Pi.
  if (z_m5a_sev <= 0.61005981f) {
    if (z_conf_mean <= -0.95620206f) {
      if (!cat_other) {
        if (!cat_hurricane) {
          return false;
        }
        return true;
      }
      if (z_m5a_sev <= -0.50217722f) {
        return true;
      }
      return false;
    }
    if (!cat_earthquake) {
      if (!cat_fire) {
        return false;
      }
      return false;
    }
    if (z_cat_agree <= -1.28153586f) {
      return false;
    }
    return false;
  }

  if (!cat_other) {
    if (z_cat_agree <= 0.29009467f) {
      if (z_cat_agree <= -1.28153586f) {
        return true;
      }
      return true;
    }
    if (!cat_landslide) {
      return true;
    }
    return true;
  }

  if (z_sev_agree <= -0.95482551f) {
    if (z_conf_mean <= -0.63359553f) {
      return true;
    }
    return false;
  }
  if (z_conf_mean <= 1.30204380f) {
    return false;
  }
  return false;
}

