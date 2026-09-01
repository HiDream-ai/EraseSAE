# EraseSAE

Official PyTorch implementation of **EraseSAE**, a sparse-autoencoder method
for concept erasure in text-to-video diffusion models. This release contains
the code and prompts required to train and run EraseSAE on HunyuanVideo and
CogVideoX-5B.

Training streams transformer activations online. It does not save activation
dumps or require source videos when using the default prompt mode. Attribution
and feature mining run automatically after SAE optimization and publish a
validated checkpoint bundle for inference.

Evaluation code, pretrained video-model weights, EraseSAE checkpoints, generated
videos, and experiment logs are intentionally not included.

## Repository Structure

```text
EraseSAE/
├── configs/                 # Training, attribution, and inference defaults
├── data/
│   ├── train/              # Prompts used by online SAE training
│   └── inference_demo/     # Final Hunyuan/CogVideoX demo prompts
├── model/                   # Partitioned SAE definitions and objectives
├── training/                # Online training and attribution entry points
├── inference/
│   ├── hunyuan/             # HunyuanVideo inference
│   └── cog/                 # CogVideoX inference
├── checkpoints/             # Empty target directories for EraseSAE bundles
├── pretrained_models/       # Optional local base-model directories
├── MODEL_DOWNLOADS.md
├── requirements.txt
└── constraints-tested.txt
```

## Installation

Create a Python 3.10 environment and install a CUDA-compatible PyTorch build for
your system. Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

To reproduce the versions used in our verification environment:

```bash
pip install -r requirements.txt -c constraints-tested.txt
```

Run every command below from the repository root with `PYTHONPATH=.`.

## Models and Checkpoints

The public defaults in `configs/paths.json` use:

- `hunyuanvideo-community/HunyuanVideo`
- `zai-org/CogVideoX-5b`

Override them with local Diffusers-format model directories:

```bash
export ERASESAE_HUNYUAN_MODEL_PATH=/mnt/afs_wangxinghao/model/HunyuanVideo
export ERASESAE_COGVIDEOX_MODEL_PATH=/mnt/afs_wangxinghao/model/CogVideoX-5b
```

Alternatively, pass `--base-model-path` to a training or inference command.
The empty `pretrained_models/HunyuanVideo/` and
`pretrained_models/CogVideoX-5b/` directories document the recommended local
names; model weights are not included.
See `pretrained_models/README.md` for complete download commands.

Place downloaded EraseSAE bundles under:

```text
checkpoints/
├── hunyuan/
│   ├── celebrity/
│   └── nudity/
└── cog/
    ├── celebrity/
    └── nudity/
```

Each inference-ready run contains a SAE weight, normalization statistics,
attribution map, checkpoint pointer, and validated `bundle_*.json` manifest.
See `MODEL_DOWNLOADS.md` for the exact filenames.

## Training

The four release training entry points are:

```text
training/hunyuan_celebs.py
training/hunyuan_nudity.py
training/cog_celebs.py
training/cog_nudity.py
```

All four use online prompt-mode activation streaming by default. For each
batch, the base model generates the configured short denoising trajectory, the
target-layer activation and target-token mask are captured, the SAE is updated,
and the activation batch is released. A compact mean/std file is cached, but no
activation tensors are written to disk.

### Single GPU

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python training/hunyuan_celebs.py \
  --base-model-path /mnt/afs_wangxinghao/model/HunyuanVideo
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python training/hunyuan_nudity.py \
  --base-model-path /mnt/afs_wangxinghao/model/HunyuanVideo
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python training/cog_celebs.py \
  --base-model-path /mnt/afs_wangxinghao/model/CogVideoX-5b
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python training/cog_nudity.py \
  --base-model-path /mnt/afs_wangxinghao/model/CogVideoX-5b
```

### Distributed Training

Use `torchrun` for one-node DDP. `--batch-size` is per process.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. \
torchrun --standalone --nproc_per_node=8 \
  training/hunyuan_celebs.py --batch-size 8 \
  --base-model-path /mnt/afs_wangxinghao/model/HunyuanVideo

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. \
torchrun --standalone --nproc_per_node=8 \
  training/hunyuan_nudity.py --batch-size 8 \
  --base-model-path /mnt/afs_wangxinghao/model/HunyuanVideo

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. \
torchrun --standalone --nproc_per_node=8 \
  training/cog_celebs.py --batch-size 8 \
  --base-model-path /mnt/afs_wangxinghao/model/CogVideoX-5b

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=. \
torchrun --standalone --nproc_per_node=8 \
  training/cog_nudity.py --batch-size 8 \
  --base-model-path /mnt/afs_wangxinghao/model/CogVideoX-5b
```

Local text and JSONL logs are always written under the checkpoint directory.
W&B is disabled by default. Enable it without storing a key in the repository:

```bash
WANDB_API_KEY=<your-key> CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
python training/hunyuan_celebs.py --wandb-mode online
```

To rerun only attribution and mining from an existing completed SAE weight:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python training/hunyuan_celebs.py \
  --skip-train true \
  --sae-save-dir checkpoints/hunyuan/celebrity/<run-directory>
```

Training, attribution, and inference defaults are separated in the four JSON
files under `configs/hunyuan/` and `configs/cogvideo/`. CLI arguments override
the JSON values. Use `--help` to inspect all available overrides.

The default training prompt root is organized as follows:

```text
data/train/
├── celebrity/
│   ├── trump.csv
│   ├── obama.csv
│   ├── musk.csv
│   ├── merkel.csv
│   ├── elizabeth.csv
│   └── swift.csv
├── non_celebrity.csv
├── nudity.csv
└── no_nudity.csv
```

The `nudity.csv` and `no_nudity.csv` rows are aligned counterfactual prompt
pairs with matching seeds and scene context.

## Inference

For a single prompt, the Python entry points generate matched original and
erased videos with the same prompt, seed, and sampling parameters.
Masked CFG always moves the selected region toward `--reference-prompt`; there
is no direction switch. `--mask-cfg-scale` controls the guidance strength.

### HunyuanVideo Celebrity

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python inference/hunyuan/inference_celeb.py \
  --target-identity trump \
  --prompt "A continuous photorealistic medium close-up video shows Trump posing for a formal portrait in a bright television interview studio. Trump is the only person in the foreground, and his face stays large, frontal, unobstructed, and sharply focused throughout the shot. He adjusts the front of his dark suit jacket, looks directly into the camera, and gives a small nod while the camera makes a very slow push-in." \
  --seed 5572250 \
  --base-model-path /mnt/afs_wangxinghao/model/HunyuanVideo \
  --checkpoint-dir /mnt/afs_wangxinghao/wangxinghao/EraseSAE/checkpoints/hunyuan/celebrity/20260807_001156 \
  --output-dir outputs/inference/hunyuan/celebrity/trump \
  --reference-prompt nudity \
  --generate-originals --save-diagnostics --save-step-masks
```

### HunyuanVideo Nudity

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python inference/hunyuan/inference_nud.py \
  --prompt "A single continuous full-body video shows a completely nude adult woman who remains fully unclothed from head to toe throughout the entire shot. In a brightly lit figure-drawing studio she slowly turns from a front-facing pose to a side profile and raises one arm. The camera makes a gentle sideways arc while keeping her entire body clearly visible." \
  --seed 2947247 \
  --base-model-path /mnt/afs_wangxinghao/model/HunyuanVideo \
  --checkpoint-dir /mnt/afs_wangxinghao/wangxinghao/EraseSAE/checkpoints/hunyuan/nudity/20260827_102841 \
  --output-dir outputs/inference/hunyuan/nudity/single \
  --reference-prompt "a fully clothed adult model wearing a modest long-sleeved outfit and full-length trousers" \
  --mask-background-threshold 0.2 \
  --mask-concept-threshold 0.5 --mask-dilation 2 \
  --generate-originals --save-diagnostics --save-step-masks
```

### CogVideoX Celebrity

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python inference/cog/inference_partitioned.py \
  --task celebrity \
  --target-identity swift \
  --prompt "A short continuous photorealistic medium close-up video shows Taylor Swift wearing a sparkling sequined performance outfit on a brightly lit concert stage. Taylor Swift is the only person in the foreground, and her face stays large, frontal, unobstructed, and sharply focused as she smiles directly at the camera, holds a microphone, and gently raises one hand during a very slow push-in." \
  --seed 5108603 \
  --base-model-path /mnt/afs_wangxinghao/model/CogVideoX-5b \
  --checkpoint-dir /mnt/afs_wangxinghao/wangxinghao/EraseSAE/checkpoints/cog/celebrity/20260829_042209 \
  --output-dir outputs/inference/cog/celebrity/swift \
  --reference-prompt nudity \
  --generate-originals --save-diagnostics --save-step-masks
```

### CogVideoX Nudity

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python inference/cog/inference_partitioned.py \
  --task nudity \
  --prompt "A short continuous full-body video shows a completely nude adult woman who remains fully unclothed from head to toe throughout the shot. She walks beside a sunlit swimming pool and brushes wet hair from her face while the camera tracks alongside her." \
  --seed 2222361 \
  --base-model-path /mnt/afs_wangxinghao/model/CogVideoX-5b \
  --checkpoint-dir /mnt/afs_wangxinghao/wangxinghao/EraseSAE/checkpoints/cog/nudity/20260829_002532 \
  --output-dir outputs/inference/cog/nudity/single \
  --reference-prompt nudity \
  --generate-originals --save-diagnostics --save-step-masks
```

CogVideoX dynamic CFG is enabled by default. Pass `--no-dynamic-cfg` only for a
constant-CFG ablation.

### Celebrity Batch Inference

The celebrity CSV files include an `Identity` column. In `self` mode, each
eraser processes only prompts belonging to the same identity, so the batch does
not create an identity-pair grid.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python inference/hunyuan/inference_celeb.py \
  --target-identities trump obama musk merkel elizabeth swift \
  --prompts-file data/inference_demo/hunyuan/celebrity.csv \
  --validation-mode self \
  --base-model-path /mnt/afs_wangxinghao/model/HunyuanVideo \
  --checkpoint-dir /mnt/afs_wangxinghao/wangxinghao/EraseSAE/checkpoints/hunyuan/celebrity/20260807_001156 \
  --output-dir outputs/inference/hunyuan/celebrity_batch \
  --reference-prompt nudity \
  --generate-originals

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python inference/cog/inference_partitioned.py \
  --task celebrity \
  --target-identities trump obama musk merkel elizabeth swift \
  --prompts-file data/inference_demo/cogvideo/celebrity.csv \
  --validation-mode self \
  --base-model-path /mnt/afs_wangxinghao/model/CogVideoX-5b \
  --checkpoint-dir /mnt/afs_wangxinghao/wangxinghao/EraseSAE/checkpoints/cog/celebrity/20260829_042209 \
  --output-dir outputs/inference/cog/celebrity_batch \
  --reference-prompt nudity \
  --generate-originals
```

## Outputs

Training writes timestamped checkpoints and local logs under `checkpoints/`.
Inference writes MP4 files, step-mask PNGs, and optional diagnostics JSON under
`outputs/`. Both directories are ignored for generated artifacts.

## Citation

Please cite the EraseSAE paper. Add the final proceedings BibTeX entry here
before publishing the repository.
