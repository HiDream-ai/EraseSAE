#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MODEL="${MODEL:-hunyuan}"
TASK="${TASK:-celebrity}"
GPU="${GPU:-0}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"

BATCH_SIZE="${BATCH_SIZE:-1}"
SAE_EPOCHS="${SAE_EPOCHS:-1}"
MAX_SAMPLES_PER_SUBJECT="${MAX_SAMPLES_PER_SUBJECT:-1}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-256}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"

if [[ -d data/train ]]; then
  DEFAULT_RAW_DATA_ROOT="data/train"
else
  DEFAULT_RAW_DATA_ROOT="data/prompts"
fi
RAW_DATA_ROOT="${RAW_DATA_ROOT:-$DEFAULT_RAW_DATA_ROOT}"

case "$MODEL:$TASK" in
  hunyuan:celebrity)
    ENTRYPOINT="training/hunyuan_celebs.py"
    DEFAULT_FRAMES=9
    ;;
  hunyuan:nudity)
    ENTRYPOINT="training/hunyuan_nudity.py"
    DEFAULT_FRAMES=9
    ;;
  cog:celebrity|cogvideo:celebrity)
    MODEL="cog"
    ENTRYPOINT="training/cog_celebs.py"
    DEFAULT_FRAMES=8
    ;;
  cog:nudity|cogvideo:nudity)
    MODEL="cog"
    ENTRYPOINT="training/cog_nudity.py"
    DEFAULT_FRAMES=8
    ;;
  *)
    printf 'MODEL must be hunyuan, cog, or cogvideo; TASK must be celebrity or nudity.\n' >&2
    exit 2
    ;;
esac

FRAMES="${FRAMES:-$DEFAULT_FRAMES}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/smoke/train/${MODEL}_${TASK}/${RUN_ID}}"

if [[ ! -d "$RAW_DATA_ROOT" ]]; then
  printf '[Smoke] Training prompt directory not found: %s\n' "$RAW_DATA_ROOT" >&2
  exit 1
fi
if [[ "$GPU" == *,* ]]; then
  printf '[Smoke] Use exactly one visible GPU for this smoke test.\n' >&2
  exit 2
fi

COMMAND=(
  torchrun
  --standalone
  --nproc_per_node=1
  "$ENTRYPOINT"
  --raw-data-root "$RAW_DATA_ROOT"
  --sae-save-dir "$OUTPUT_DIR/checkpoints"
  --log-dir "$OUTPUT_DIR/logs"
  --batch-size "$BATCH_SIZE"
  --sae-epochs "$SAE_EPOCHS"
  --online-batch-reuse 1
  --normalization-mode none
  --activation-sampling-mode noisy_latent
  --activation-num-frames "$FRAMES"
  --activation-height "$HEIGHT"
  --activation-width "$WIDTH"
  --activation-timesteps-per-video 1
  --activation-trajectory-steps 1
  --max-samples-per-subject "$MAX_SAMPLES_PER_SUBJECT"
  --empty-cache-every-step true
  --wandb-mode disabled
  --skip-attribution
)

if [[ -n "$BASE_MODEL_PATH" ]]; then
  COMMAND+=(--base-model-path "$BASE_MODEL_PATH")
fi

printf '[Smoke] Model/task: %s/%s\n' "$MODEL" "$TASK"
printf '[Smoke] Output: %s\n' "$OUTPUT_DIR"
printf '[Smoke] Command:'
printf ' %q' env "CUDA_VISIBLE_DEVICES=$GPU" "ERASESAE_RUN_ID=$RUN_ID" "${COMMAND[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" || "$DRY_RUN" == "yes" ]]; then
  printf '[Smoke] DRY_RUN enabled; training was not started.\n'
  exit 0
fi

CUDA_VISIBLE_DEVICES="$GPU" ERASESAE_RUN_ID="$RUN_ID" "${COMMAND[@]}"

