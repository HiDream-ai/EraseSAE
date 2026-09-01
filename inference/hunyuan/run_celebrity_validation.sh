#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# The default self mode pairs each eraser with matching Identity rows.
GPU="${GPU:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/hunyuan/celebrity}"
PROMPTS_FILE="${PROMPTS_FILE:-data/inference_demo/hunyuan/celebrity.csv}"
TARGET_IDENTITIES="${TARGET_IDENTITIES:-trump,obama,musk,merkel,elizabeth,swift}"
VALIDATION_MODE="${VALIDATION_MODE:-self}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"

REFERENCE_PROMPT="${REFERENCE_PROMPT:-nudity}"
MASK_CFG_SCALE="${MASK_CFG_SCALE:-10.0}"

MASK_BACKGROUND_THRESHOLD="${MASK_BACKGROUND_THRESHOLD:-0.4}"
MASK_IDENTITY_THRESHOLD="${MASK_IDENTITY_THRESHOLD:-0.2}"
MASK_DILATION="${MASK_DILATION:-0}"
IDENTITY_COMPETITION_RATIO="${IDENTITY_COMPETITION_RATIO:-1.05}"
IDENTITY_MIN_RELATIVE_SCORE="${IDENTITY_MIN_RELATIVE_SCORE:-0.05}"
IDENTITY_SCORE_TOP_FRACTION="${IDENTITY_SCORE_TOP_FRACTION:-0.2}"

STEPS="${STEPS:-30}"
FRAMES="${FRAMES:-32}"
HEIGHT="${HEIGHT:-480}"
WIDTH="${WIDTH:-720}"
FPS="${FPS:-8}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"
MASK_START_STEP="${MASK_START_STEP:-5}"
MASK_END_STEP="${MASK_END_STEP:-$((STEPS - 1))}"
SEED="${SEED:-42}"
SEED_STEP="${SEED_STEP:-1}"

# First six rows cover every identity once. Set 0 for all twelve prompts.
MAX_PROMPTS="${MAX_PROMPTS:-6}"
GENERATE_ORIGINALS="${GENERATE_ORIGINALS:-1}"
SAVE_DIAGNOSTICS="${SAVE_DIAGNOSTICS:-1}"
SAVE_STEP_MASKS="${SAVE_STEP_MASKS:-1}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
SEQUENTIAL_CPU_OFFLOAD="${SEQUENTIAL_CPU_OFFLOAD:-0}"
DRY_RUN="${DRY_RUN:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/inference/hunyuan/celebrity_${VALIDATION_MODE}/${RUN_ID}}"

is_enabled() {
  [[ "$1" == "1" || "$1" == "true" || "$1" == "yes" ]]
}

if [[ ! -f "$PROMPTS_FILE" ]]; then
  printf '[Validation] Prompt file not found: %s\n' "$PROMPTS_FILE" >&2
  exit 1
fi
if [[ ! -d "$CHECKPOINT_DIR" ]]; then
  printf '[Validation] Checkpoint directory not found: %s\n' "$CHECKPOINT_DIR" >&2
  exit 1
fi
if ! [[ "$MAX_PROMPTS" =~ ^[0-9]+$ ]]; then
  printf '[Validation] MAX_PROMPTS must be a non-negative integer.\n' >&2
  exit 1
fi
if [[ "$VALIDATION_MODE" != "self" && "$VALIDATION_MODE" != "cross" ]]; then
  printf '[Validation] VALIDATION_MODE must be self or cross.\n' >&2
  exit 1
fi
if is_enabled "$CPU_OFFLOAD" && is_enabled "$SEQUENTIAL_CPU_OFFLOAD"; then
  printf '[Validation] CPU offload modes are mutually exclusive.\n' >&2
  exit 1
fi

IFS=',' read -r -a TARGET_ARRAY <<< "$TARGET_IDENTITIES"
if (( ${#TARGET_ARRAY[@]} == 0 )); then
  printf '[Validation] TARGET_IDENTITIES cannot be empty.\n' >&2
  exit 1
fi
for target in "${TARGET_ARRAY[@]}"; do
  case "$target" in
    trump|obama|musk|merkel|elizabeth|swift) ;;
    *)
      printf '[Validation] Unsupported target identity: %s\n' "$target" >&2
      exit 1
      ;;
  esac
done
COMMAND=(
  python inference/hunyuan/inference_celeb.py
  --target-identities "${TARGET_ARRAY[@]}"
  --prompts-file "$PROMPTS_FILE"
  --validation-mode "$VALIDATION_MODE"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --output-dir "$OUTPUT_DIR"
  --mask-cfg-scale "$MASK_CFG_SCALE"
  --mask-background-threshold "$MASK_BACKGROUND_THRESHOLD"
  --mask-identity-threshold "$MASK_IDENTITY_THRESHOLD"
  --mask-dilation "$MASK_DILATION"
  --identity-competition-ratio "$IDENTITY_COMPETITION_RATIO"
  --identity-min-relative-score "$IDENTITY_MIN_RELATIVE_SCORE"
  --identity-score-top-fraction "$IDENTITY_SCORE_TOP_FRACTION"
  --mask-start-step "$MASK_START_STEP"
  --mask-end-step "$MASK_END_STEP"
  --steps "$STEPS"
  --frames "$FRAMES"
  --height "$HEIGHT"
  --width "$WIDTH"
  --fps "$FPS"
  --guidance-scale "$GUIDANCE_SCALE"
  --seed "$SEED"
  --seed-step "$SEED_STEP"
)

if (( MAX_PROMPTS > 0 )); then
  COMMAND+=(--max-prompts "$MAX_PROMPTS")
fi
if [[ -n "$REFERENCE_PROMPT" ]]; then
  COMMAND+=(--reference-prompt "$REFERENCE_PROMPT")
fi
if is_enabled "$GENERATE_ORIGINALS"; then
  COMMAND+=(--generate-originals)
fi
if is_enabled "$SAVE_DIAGNOSTICS"; then
  COMMAND+=(--save-diagnostics)
fi
if is_enabled "$SAVE_STEP_MASKS"; then
  COMMAND+=(--save-step-masks)
fi
if is_enabled "$CPU_OFFLOAD"; then
  COMMAND+=(--cpu-offload)
elif is_enabled "$SEQUENTIAL_CPU_OFFLOAD"; then
  COMMAND+=(--sequential-cpu-offload)
fi

printf '[Validation] Output: %s\n' "$OUTPUT_DIR"
printf '[Validation] Command:'
printf ' %q' env "CUDA_VISIBLE_DEVICES=$GPU" "${COMMAND[@]}"
printf '\n'

if is_enabled "$DRY_RUN"; then
  printf '[Validation] DRY_RUN enabled; inference was not started.\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU" "${COMMAND[@]}"
