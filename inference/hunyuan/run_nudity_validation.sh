#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# -----------------------------------------------------------------------------
# Editable validation settings. Environment variables override these defaults.
# -----------------------------------------------------------------------------
GPU="${GPU:-0}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints/hunyuan/nudity}"
PROMPTS_FILE="${PROMPTS_FILE:-data/inference_demo/hunyuan/nudity.csv}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"

REFERENCE_PROMPT="${REFERENCE_PROMPT:-a fully clothed adult model wearing a modest long-sleeved outfit and full-length trousers}"
MASK_CFG_SCALE="${MASK_CFG_SCALE:-3.0}"

MASK_MODE="${MASK_MODE:-intersection}"
MASK_BACKGROUND_THRESHOLD="${MASK_BACKGROUND_THRESHOLD:-0.2}"
MASK_CONCEPT_THRESHOLD="${MASK_CONCEPT_THRESHOLD:-0.5}"
MASK_DILATION="${MASK_DILATION:-2}"
COMPETITION_RATIO="${COMPETITION_RATIO:-1.05}"
MIN_RELATIVE_SCORE="${MIN_RELATIVE_SCORE:-0.05}"
SCORE_TOP_FRACTION="${SCORE_TOP_FRACTION:-0.2}"

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

# Set MAX_PROMPTS=0 to use the complete CSV.
MAX_PROMPTS="${MAX_PROMPTS:-0}"
GENERATE_ORIGINALS="${GENERATE_ORIGINALS:-1}"
SAVE_DIAGNOSTICS="${SAVE_DIAGNOSTICS:-1}"
SAVE_STEP_MASKS="${SAVE_STEP_MASKS:-1}"
CPU_OFFLOAD="${CPU_OFFLOAD:-0}"
DRY_RUN="${DRY_RUN:-0}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/inference/hunyuan/nudity/toward_d${MASK_DILATION}_${RUN_ID}}"

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
COMMAND=(
  python inference/hunyuan/inference_nud.py
  --prompts-file "$PROMPTS_FILE"
  --checkpoint-dir "$CHECKPOINT_DIR"
  --output-dir "$OUTPUT_DIR"
  --reference-prompt "$REFERENCE_PROMPT"
  --mask-cfg-scale "$MASK_CFG_SCALE"
  --mask-mode "$MASK_MODE"
  --mask-background-threshold "$MASK_BACKGROUND_THRESHOLD"
  --mask-concept-threshold "$MASK_CONCEPT_THRESHOLD"
  --mask-dilation "$MASK_DILATION"
  --competition-ratio "$COMPETITION_RATIO"
  --min-relative-score "$MIN_RELATIVE_SCORE"
  --score-top-fraction "$SCORE_TOP_FRACTION"
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
