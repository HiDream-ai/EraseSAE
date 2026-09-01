# Download Base Models

Base-model weights are not included in this repository. Run the commands below
from the EraseSAE repository root.

## Hugging Face CLI (Recommended)

Install the Hugging Face CLI:

```bash
pip install -U huggingface_hub
```

Download CogVideoX-5b:

```bash
hf download zai-org/CogVideoX-5b \
  --local-dir pretrained_models/CogVideoX-5b
```

Download HunyuanVideo:

```bash
hf download hunyuanvideo-community/HunyuanVideo \
  --local-dir pretrained_models/HunyuanVideo
```

These are public repositories, so authentication is normally unnecessary. If
Hugging Face requests authentication or applies anonymous rate limits, run:

```bash
hf auth login
```

Interrupted `hf download` commands can be run again and will reuse already
downloaded files.

## Git LFS Alternative

The models can instead be cloned into the repository-local model directories:

```bash
git lfs install
git clone https://huggingface.co/zai-org/CogVideoX-5b pretrained_models/CogVideoX-5b
git clone https://huggingface.co/hunyuanvideo-community/HunyuanVideo pretrained_models/HunyuanVideo
```

The configured defaults already point to these directories. Equivalent explicit
overrides are:

```bash
export ERASESAE_COGVIDEOX_MODEL_PATH=pretrained_models/CogVideoX-5b
export ERASESAE_HUNYUAN_MODEL_PATH=pretrained_models/HunyuanVideo
```

Each model directory must directly contain its Diffusers files, including
`model_index.json`; avoid an extra nested model directory.

Model pages:

- [CogVideoX-5b](https://huggingface.co/zai-org/CogVideoX-5b)
- [HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo)
