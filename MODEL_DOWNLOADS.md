# Model and Checkpoint Requirements

Large weights are not included in this source release. Base video models and
released EraseSAE checkpoints are downloaded separately as described below.

## Base Models

The default local model directories are configured in `configs/paths.json`:

- HunyuanVideo: `pretrained_models/HunyuanVideo`
- CogVideoX-5B: `pretrained_models/CogVideoX-5b`

To override them with other repository-relative directories, set:

```bash
export ERASESAE_HUNYUAN_MODEL_PATH=pretrained_models/HunyuanVideo
export ERASESAE_COGVIDEOX_MODEL_PATH=pretrained_models/CogVideoX-5b
```

The optional local directory names reserved by this repository are:

```text
pretrained_models/HunyuanVideo/
pretrained_models/CogVideoX-5b/
```

See `pretrained_models/README.md` for Hugging Face CLI and Git LFS download
commands.

## EraseSAE Checkpoints

Released SAE checkpoints are hosted at
[wxhustc/EraseSAE](https://huggingface.co/wxhustc/EraseSAE). From the EraseSAE
repository root, download all four model/task combinations with:

```bash
hf download wxhustc/EraseSAE \
  --include "checkpoints/**" \
  --local-dir .
```

For only one video-model family, use one of:

```bash
hf download wxhustc/EraseSAE \
  --include "checkpoints/hunyuan/**" \
  --local-dir .

hf download wxhustc/EraseSAE \
  --include "checkpoints/cog/**" \
  --local-dir .
```

The repository is public, so these downloads do not require an access token.
The commands preserve the paths expected by training and inference:

```text
checkpoints/hunyuan/celebrity/
checkpoints/hunyuan/nudity/
checkpoints/cog/celebrity/
checkpoints/cog/nudity/
```

A timestamped run directory has the following inference-ready structure:

```text
<run>/
├── conv_sae_<layer>_<epoch>.pt
├── latest_conv_sae_<layer>.json
├── kernel_identity_map_<layer>.pt
├── bundle_<layer>.json       # required by CogVideoX inference
└── online_stats/
    └── stats_<layer>.pt
```

Keep all files from one run together. Passing the concrete run directory gives
deterministic checkpoint selection. A task root is also accepted: HunyuanVideo
selects the newest matching checkpoint, while CogVideoX selects the newest
immediate child containing a complete validated `bundle_*.json` manifest.
