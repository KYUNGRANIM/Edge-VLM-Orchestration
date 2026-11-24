#pragma once

// cat/sev/conf는 텍스트 기반 후처리로 생성 (1B 모델 템플릿 고정 방지)
// Describe-first 구조: 이미지에서 시각적 증거를 먼저 서술하게 한 뒤 분류
// → "The disaster type is X" 템플릿 방지, earthquake/landslide 오분류 감소
// → 각 카테고리에 시각적 anchor 제공 (earthquake=구조물 붕괴 vs landslide=사면 토사)

// M5_A — Representative: 피해 묘사 + 즉시 대처 (2문장)
// v1 스타일: 카테고리 목록 없음 → "The disaster type is X" 템플릿 방지
// cat/sev는 자연스러운 묘사에서 post-processing으로 추출
static const char* PROMPT_TEXT_A =
  "You are a disaster expert.\n"
  "Describe the damage and the most urgent action to take. Two sentences.\n"
  "Do not start with Assessing.\n";

static const char* PROMPT_TEXT_A_BACKUP =
  "You are a disaster expert.\n"
  "Describe what happened and what should be done immediately. Two sentences.\n";

// M5_B — Auxiliary: 구조물/인프라 물리 손상 (1문장)
static const char* PROMPT_TEXT_B =
  "You are a disaster expert.\n"
  "Describe the physical destruction or collapse visible in this image. One sentence.\n";

static const char* PROMPT_TEXT_B_BACKUP =
  "You are a disaster expert.\n"
  "Describe the broken or blocked structures here. One sentence.\n";

// M5_C — Auxiliary: 인명 위험/영향 (1문장)
static const char* PROMPT_TEXT_C =
  "You are a disaster expert.\n"
  "Describe how people could be hurt or killed here. One sentence.\n";

static const char* PROMPT_TEXT_C_BACKUP =
  "You are a disaster expert.\n"
  "Describe the danger to people near this scene. One sentence.\n";
