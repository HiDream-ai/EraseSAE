# EraseSAE Checkpoints

Released weights are hosted at
[wxhustc/EraseSAE](https://huggingface.co/wxhustc/EraseSAE). They are not
duplicated in the source archive. From the EraseSAE repository root, run:

```bash
hf download wxhustc/EraseSAE \
  --include "checkpoints/**" \
  --local-dir .
```

The command populates these inference-ready locations:

```text
checkpoints/hunyuan/celebrity/<run>/
checkpoints/hunyuan/nudity/<run>/
checkpoints/cog/celebrity/<run>/
checkpoints/cog/nudity/<run>/
```

See `../MODEL_DOWNLOADS.md` for selective downloads and the expected files.
