import argparse
import json
import os
from copy import deepcopy
from pathlib import Path


CONFIG_ROOT = Path(__file__).resolve().parent

EXPERIMENT_CONFIGS = {
    "hunyuan_nudity": CONFIG_ROOT / "hunyuan" / "nudity.json",
    "hunyuan_celebrity": CONFIG_ROOT / "hunyuan" / "celebrity.json",
    "hunyuan_multiconcept": CONFIG_ROOT / "hunyuan" / "multiconcept.json",
    "cogvideo_nudity": CONFIG_ROOT / "cogvideo" / "nudity.json",
    "cogvideo_celebrity": CONFIG_ROOT / "cogvideo" / "celebrity.json",
}

MODEL_PATH_ENV_VARS = {
    "hunyuan": "ERASESAE_HUNYUAN_MODEL_PATH",
    "cogvideox_5b": "ERASESAE_COGVIDEOX_MODEL_PATH",
}


def str2bool(value):
    if value is None or isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model_paths():
    paths = load_json(CONFIG_ROOT / "paths.json")
    local_path = CONFIG_ROOT / "paths.local.json"
    if local_path.is_file():
        local_paths = load_json(local_path)
        if not isinstance(local_paths, dict):
            raise TypeError(f"Local model paths must be a JSON object: {local_path}")
        paths.update(local_paths)
    return paths


def get_model_path(name):
    env_name = MODEL_PATH_ENV_VARS.get(name)
    if env_name and os.environ.get(env_name):
        return os.environ[env_name]

    paths = load_model_paths()
    if name not in paths:
        raise KeyError(f"Unknown model path key: {name}")
    return paths[name]


def load_experiment_config(name_or_path):
    path = EXPERIMENT_CONFIGS.get(name_or_path, name_or_path)
    cfg = deepcopy(load_json(path))
    model_key = cfg.get("base_model_key")
    if model_key:
        cfg.setdefault("training", {})["base_model_path"] = get_model_path(model_key)
    return cfg


def _section(cfg, name):
    return cfg.get(name, {})


def add_training_args(parser, cfg):
    train = _section(cfg, "training")
    attr = _section(cfg, "attribution")
    infer = _section(cfg, "inference")
    architecture = cfg.get("architecture", "partitioned")

    base = parser.add_argument_group("base")
    base.add_argument(
        "--global-seed",
        type=int,
        default=train.get("global_seed", 42),
        help=(
            "Base seed for Python, NumPy, PyTorch, prompt shuffling, and latent "
            "sampling. Each DDP worker adds its rank where independent streams "
            "are required."
        ),
    )
    base.add_argument(
        "--use-bf16",
        type=str2bool,
        default=train.get("use_bf16", True),
        help=(
            "Use bfloat16 model execution when the selected CUDA device supports "
            "it; otherwise training falls back to float16."
        ),
    )

    paths = parser.add_argument_group("paths")
    paths.add_argument(
        "--base-model-path",
        type=str,
        default=train.get("base_model_path"),
        help=(
            "Repository-relative Diffusers model directory or Hugging Face model "
            "identifier used to generate online activations."
        ),
    )
    paths.add_argument(
        "--sae-save-dir",
        type=str,
        default=train.get("sae_save_dir"),
        help=(
            "Task checkpoint directory for SAE weights, final-checkpoint pointers, "
            "normalization statistics, attribution maps, and bundle manifests."
        ),
    )
    paths.add_argument(
        "--log-dir",
        type=str,
        default=train.get("log_dir"),
        help=(
            "Directory for per-rank stdout/stderr logs, the resolved config JSON, "
            "and JSONL metrics. Defaults to <sae-save-dir>/logs when unset."
        ),
    )
    paths.add_argument(
        "--data-save-dir",
        type=str,
        default=attr.get("data_save_dir"),
        help=(
            "Root containing saved activation tensors and statistics for offline "
            "mode. Online prompt training does not write activation tensors here."
        ),
    )
    if "raw_data_root" in attr:
        paths.add_argument(
            "--raw-data-root",
            type=str,
            default=attr.get("raw_data_root"),
            help=(
                "Input root: CSV prompt files for --activation-input prompt, or "
                "metadata plus source videos for --activation-input video."
            ),
        )

    train_group = parser.add_argument_group("training")
    train_group.add_argument(
        "--target-layers",
        type=str,
        default=train.get("target_layers"),
        help=(
            "Transformer block indices to train. Accepts one index ('20'), a "
            "comma-separated list ('15,17,19'), or an inclusive range with an "
            "optional stride ('15-35:2')."
        ),
    )
    train_group.add_argument(
        "--sae-epochs",
        type=int,
        default=train.get("sae_epochs"),
        help=(
            "Number of effective optimizer passes over the per-rank activation "
            "dataset for each target layer."
        ),
    )
    if architecture == "multi_concept":
        train_group.add_argument(
            "--excl-beta",
            type=float,
            default=train.get("excl_beta"),
            help="Weight of the cross-concept exclusivity loss.",
        )
    train_group.add_argument(
        "--batch-size",
        type=int,
        default=train.get("batch_size"),
        help=(
            "Activation samples per optimizer update on each DDP process. The "
            "effective global batch is this value multiplied by world size."
        ),
    )
    train_group.add_argument(
        "--k-gen",
        type=int,
        default=train.get("k_gen"),
        help="Top-K active channels per spatial token in the generic SAE partition.",
    )
    train_group.add_argument(
        "--k-id",
        type=int,
        default=train.get("k_id"),
        help=(
            "Top-K active channels per spatial token inside the routed concept "
            "partition."
        ),
    )
    train_group.add_argument(
        "--n-gen-kernels",
        type=int,
        default=train.get("n_gen_kernels"),
        help="Number of dictionary channels in the shared generic SAE partition.",
    )
    if architecture == "multi_concept":
        train_group.add_argument(
            "--n-id",
            type=int,
            default=train.get("n_id"),
            help="Total dictionary channels in the multi-concept identity branch.",
        )
    else:
        train_group.add_argument(
            "--n-id-per-celeb",
            type=int,
            default=train.get("n_id_per_celeb"),
            help=(
                "Dictionary channels reserved for each concept. Total concept "
                "channels equal this value times the number of concepts."
            ),
        )
    if architecture == "partitioned" and "partition_beta" in train:
        train_group.add_argument(
            "--partition-beta",
            type=float,
            default=train.get("partition_beta"),
            help=(
                "Weight of partition leakage loss, which suppresses non-owner "
                "concept channels and owner channels outside the target mask."
            ),
        )
        train_group.add_argument(
            "--leak-start-fraction",
            type=float,
            default=train.get("leak_start_fraction", 0.3),
            help=(
                "Fraction of effective training completed before partition leakage "
                "and contrast losses begin; must be in [0, 1)."
            ),
        )
        train_group.add_argument(
            "--partition-contrast-weight",
            type=float,
            default=train.get("partition_contrast_weight", 0.0),
            help=(
                "Weight of supervised owner-versus-other partition separation in "
                "the masked target region."
            ),
        )
        train_group.add_argument(
            "--partition-contrast-margin",
            type=float,
            default=train.get("partition_contrast_margin", 0.1),
            help=(
                "Minimum calibrated activation margin enforced between the owner "
                "partition and competing concept partitions."
            ),
        )
    train_group.add_argument(
        "--face-weight",
        type=float,
        default=train.get("face_weight"),
        help=(
            "Weight of target-region reconstruction by the routed concept branch; "
            "the historical option name is retained for both celebrity and nudity."
        ),
    )
    train_group.add_argument(
        "--aux-coeff",
        type=float,
        default=train.get("aux_coeff"),
        help="Weight of the auxiliary residual-reconstruction loss for sparse features.",
    )
    train_group.add_argument(
        "--bg-exclusive-weight",
        type=float,
        default=train.get("bg_exclusive_weight"),
        help=(
            "Weight for generic-branch background reconstruction and suppression "
            "of generic feature activity inside target regions."
        ),
    )
    train_group.add_argument(
        "--temp-consistency",
        type=float,
        default=train.get("temp_consistency"),
        help=(
            "Weight of adjacent-frame feature consistency for both generic and "
            "concept SAE activations."
        ),
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=train.get("lr"),
        help="AdamW learning rate for SAE parameters.",
    )
    train_group.add_argument(
        "--online-batch-reuse",
        type=int,
        default=train.get("online_batch_reuse", 1),
        help=(
            "Reuse each streamed online activation batch for this many consecutive "
            "optimizer updates. The total update budget is unchanged, so larger "
            "values reduce expensive base-model generation passes. Offline mode "
            "always uses 1."
        ),
    )
    train_group.add_argument(
        "--wandb-name",
        type=str,
        default=train.get("wandb_name"),
        help="Weights & Biases project name; the layer and run ID form the run name.",
    )
    train_group.add_argument(
        "--wandb-mode",
        choices=["disabled", "offline", "online"],
        default=train.get("wandb_mode", os.environ.get("WANDB_MODE", "disabled")),
        help=(
            "Weights & Biases logging mode. 'disabled' keeps only local text and "
            "JSONL logs; 'offline' stores a local W&B run; 'online' uploads metrics."
        ),
    )

    attribution = parser.add_argument_group("attribution and activation collection")
    attribution.add_argument(
        "--top-k",
        type=int,
        default=attr.get("top_k"),
        help=(
            "Maximum number of positively scored kernels retained per concept in "
            "the attribution map and subsequently available to inference."
        ),
    )
    attribution.add_argument(
        "--min-mining-purity",
        type=float,
        default=attr.get("min_mining_purity", 0.0),
        help=(
            "Minimum held-out one-vs-maximum partition purity, in percent, required "
            "for every concept before an inference bundle is published."
        ),
    )
    attribution.add_argument(
        "--min-mining-kernels",
        type=int,
        default=attr.get("min_mining_kernels", 1),
        help=(
            "Minimum number of positive, stable kernels that every concept must "
            "retain after mining filters; otherwise attribution fails."
        ),
    )
    attribution.add_argument(
        "--min-mining-survival",
        type=float,
        default=attr.get("min_mining_survival", 0.1),
        help=(
            "Minimum fraction of a concept's attribution samples on which a kernel "
            "must activate to be considered stable; expressed in [0, 1)."
        ),
    )
    attribution.add_argument(
        "--min-mining-score",
        type=float,
        default=attr.get("min_mining_score", 1e-8),
        help=(
            "Minimum finite one-vs-maximum target-minus-negative mining score for "
            "a kernel to remain eligible."
        ),
    )
    attribution.add_argument(
        "--max-empty-target-mask-fraction",
        type=float,
        default=attr.get("max_empty_target_mask_fraction", 0.05),
        help=(
            "Maximum allowed fraction of attribution samples with an empty target "
            "token-attention mask, evaluated separately for each concept."
        ),
    )
    attribution.add_argument(
        "--skip-train",
        nargs="?",
        const=True,
        type=str2bool,
        default=attr.get("skip_train"),
        help=(
            "Skip optimization and run attribution/mining from an existing final "
            "SAE checkpoint. The checkpoint's normalization contract is reused."
        ),
    )
    attribution.add_argument(
        "--skip-attribution",
        action="store_true",
        help=(
            "Stop after saving the final SAE checkpoint without attribution or an "
            "inference bundle. The resulting checkpoint is not inference-ready."
        ),
    )
    attribution.add_argument(
        "--target-celebs",
        type=str,
        default=attr.get("target_concepts"),
        help=(
            "Comma-separated concept names in partition order. Nudity training "
            "requires exactly 'nudity,no_nudity'."
        ),
    )
    if "activation_source" in attr:
        attribution.add_argument(
            "--activation-source",
            choices=["online", "offline"],
            default=attr.get("activation_source"),
            help=(
                "Activation provider: 'online' runs the frozen video model inside "
                "training and releases each batch; 'offline' reads saved .pt tensors."
            ),
        )
    if "activation_input" in attr:
        attribution.add_argument(
            "--activation-input",
            choices=["prompt", "video"],
            default=attr.get("activation_input"),
            help=(
                "Online input type: 'prompt' synthesizes latent trajectories from "
                "CSV prompts; 'video' encodes source videos referenced by metadata."
            ),
        )
    attribution.add_argument(
        "--normalization-mode",
        choices=["global", "none"],
        default=attr.get("normalization_mode", "global"),
        help=(
            "Activation normalization contract. 'global' streams a preliminary "
            "per-channel mean/std pass and caches compact statistics; 'none' uses "
            "mean=0 and std=1 without a statistics pass."
        ),
    )
    attribution.add_argument(
        "--recompute-online-stats",
        action="store_true",
        help=(
            "Recompute online mean/std even when a cache with the same dataset, "
            "layer, sampling, and shape signature is available."
        ),
    )
    if "activation_num_frames" in attr:
        attribution.add_argument(
            "--activation-num-frames",
            type=int,
            default=attr.get("activation_num_frames"),
            help=(
                "Number of output video frames represented by each online activation "
                "sample before temporal VAE compression."
            ),
        )
        attribution.add_argument(
            "--activation-height",
            type=int,
            default=attr.get("activation_height"),
            help="Pixel height used for online prompt generation or video preprocessing.",
        )
        attribution.add_argument(
            "--activation-width",
            type=int,
            default=attr.get("activation_width"),
            help="Pixel width used for online prompt generation or video preprocessing.",
        )
        attribution.add_argument(
            "--activation-timesteps-per-video",
            type=int,
            default=attr.get("activation_timesteps_per_video"),
            help=(
                "Number of activation/mask samples captured per prompt or source "
                "video. In trajectory mode they are spread across trajectory steps."
            ),
        )
        attribution.add_argument(
            "--activation-timestep-min-index",
            type=int,
            default=attr.get("activation_timestep_min_index"),
            help=(
                "Inclusive lower scheduler-array index for random noisy-latent or "
                "video-input captures. It is not used by prompt trajectory capture."
            ),
        )
        attribution.add_argument(
            "--activation-sampling-mode",
            choices=["trajectory", "noisy_latent"],
            default=attr.get("activation_sampling_mode", "trajectory"),
            help=(
                "Online prompt sampling strategy. 'trajectory' runs a short scheduler "
                "trajectory and captures selected steps; 'noisy_latent' performs "
                "independent single transformer forwards at random noise levels."
            ),
        )
        attribution.add_argument(
            "--activation-trajectory-steps",
            type=int,
            default=attr.get("activation_trajectory_steps", 4),
            help=(
                "Number of denoising updates generated per prompt in trajectory "
                "mode. Must be at least the requested captures per video."
            ),
        )
        attribution.add_argument(
            "--max-samples-per-subject",
            type=int,
            default=attr.get("max_samples_per_subject"),
            help=(
                "Optional cap on input records for each concept before DDP sharding. "
                "The same capped dataset is used for normalization, training, and "
                "the held-out attribution pass."
            ),
        )
        attribution.add_argument(
            "--empty-cache-every-step",
            type=str2bool,
            default=attr.get("empty_cache_every_step"),
            help=(
                "Call torch.cuda.empty_cache() after each generated record to reduce "
                "reserved-memory pressure at the cost of allocator overhead."
            ),
        )
        attribution.add_argument(
            "--attention-mask-quantile",
            type=float,
            default=attr.get("attention_mask_quantile", 0.8),
            help=(
                "Per-frame quantile threshold for target-token attention. Locations "
                "at or above the quantile form the binary training mask; larger "
                "values generally produce smaller masks."
            ),
        )
        attribution.add_argument(
            "--attention-mask-dilation",
            type=int,
            default=attr.get("attention_mask_dilation", 0),
            help=(
                "Spatial dilation radius, measured in activation-grid cells, applied "
                "to each frame of the binary training attention mask."
            ),
        )

    inference = parser.add_argument_group("generation defaults")
    inference.add_argument(
        "--inference-steps",
        type=int,
        default=infer.get("inference_steps"),
        help=(
            "Scheduler length used by online activation generation. This defines "
            "the available denoising/noise-level schedule, not SAE optimizer steps."
        ),
    )
    inference.add_argument(
        "--num-frames",
        type=int,
        default=infer.get("num_frames"),
        help=(
            "Final-inference frame default retained in the experiment config. Online "
            "training uses --activation-num-frames instead."
        ),
    )
    inference.add_argument(
        "--guidance-scale",
        type=float,
        default=infer.get("guidance_scale"),
        help="Text classifier-free guidance scale used for online activation generation.",
    )
    inference.add_argument(
        "--fps",
        type=int,
        default=infer.get("fps"),
        help=(
            "Playback FPS retained for downstream inference output; it does not "
            "change online activation tensors."
        ),
    )

    return parser


def build_training_parser(description, config_name):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.set_defaults(experiment_config=config_name)
    return add_training_args(parser, load_experiment_config(config_name))


def finalize_training_args(args, parse_target_layers):
    import torch

    args.dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_bf16_supported()) else torch.float16
    args.target_layers_list = parse_target_layers(args.target_layers)
    if args.target_celebs:
        args.target_celebs_list = [x.strip() for x in args.target_celebs.split(",") if x.strip()]
    else:
        args.target_celebs_list = None

    if args.skip_train and args.skip_attribution:
        raise ValueError("--skip-train and --skip-attribution cannot be combined.")

    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if args.sae_epochs < 1:
        raise ValueError("sae_epochs must be at least 1.")
    if args.n_gen_kernels < 1:
        raise ValueError("n_gen_kernels must be at least 1.")
    if not 1 <= args.k_gen <= args.n_gen_kernels:
        raise ValueError("k_gen must be in [1, n_gen_kernels].")
    if getattr(args, "n_id_per_celeb", None) is not None:
        if args.n_id_per_celeb < 1:
            raise ValueError("n_id_per_celeb must be at least 1.")
        if not 1 <= args.k_id <= args.n_id_per_celeb:
            raise ValueError("k_id must be in [1, n_id_per_celeb].")
        if not 1 <= args.top_k <= args.n_id_per_celeb:
            raise ValueError("top_k must be in [1, n_id_per_celeb].")
    elif getattr(args, "n_id", None) is not None:
        if args.n_id < 1:
            raise ValueError("n_id must be at least 1.")
        if not 1 <= args.k_id <= args.n_id:
            raise ValueError("k_id must be in [1, n_id].")
    if args.lr <= 0:
        raise ValueError("lr must be positive.")
    for name in (
        "partition_beta",
        "partition_contrast_weight",
        "face_weight",
        "aux_coeff",
        "bg_exclusive_weight",
        "temp_consistency",
    ):
        value = getattr(args, name, None)
        if value is not None and value < 0:
            raise ValueError(f"{name} cannot be negative.")
    if getattr(args, "online_batch_reuse", 1) < 1:
        raise ValueError("online_batch_reuse must be at least 1.")
    if not 0.0 <= getattr(args, "min_mining_purity", 0.0) <= 100.0:
        raise ValueError("min_mining_purity must be in [0, 100].")
    if args.min_mining_kernels < 1 or args.min_mining_kernels > args.top_k:
        raise ValueError("min_mining_kernels must be in [1, top_k].")
    if not 0.0 <= args.min_mining_survival < 1.0:
        raise ValueError("min_mining_survival must be in [0, 1).")
    if args.min_mining_score < 0.0:
        raise ValueError("min_mining_score cannot be negative.")
    if not 0.0 <= args.max_empty_target_mask_fraction <= 1.0:
        raise ValueError("max_empty_target_mask_fraction must be in [0, 1].")
    if hasattr(args, "leak_start_fraction") and not 0.0 <= args.leak_start_fraction < 1.0:
        raise ValueError("leak_start_fraction must be in [0, 1).")
    if getattr(args, "partition_contrast_weight", 0.0) < 0.0:
        raise ValueError("partition_contrast_weight cannot be negative.")
    if getattr(args, "partition_contrast_margin", 0.0) < 0.0:
        raise ValueError("partition_contrast_margin cannot be negative.")
    if hasattr(args, "attention_mask_quantile"):
        if not 0.0 < args.attention_mask_quantile < 1.0:
            raise ValueError("attention_mask_quantile must be in (0, 1).")
        if args.attention_mask_dilation < 0:
            raise ValueError("attention_mask_dilation cannot be negative.")
    if hasattr(args, "activation_num_frames"):
        for name in (
            "activation_num_frames",
            "activation_height",
            "activation_width",
            "activation_timesteps_per_video",
            "activation_trajectory_steps",
        ):
            if getattr(args, name) < 1:
                raise ValueError(f"{name} must be at least 1.")
        if args.activation_timestep_min_index < 0:
            raise ValueError("activation_timestep_min_index cannot be negative.")
        if (
            args.max_samples_per_subject is not None
            and args.max_samples_per_subject < 1
        ):
            raise ValueError("max_samples_per_subject must be at least 1.")
    if (
        getattr(args, "activation_sampling_mode", None) == "trajectory"
        and args.activation_timesteps_per_video > args.activation_trajectory_steps
    ):
        raise ValueError(
            "activation_timesteps_per_video cannot exceed "
            "activation_trajectory_steps."
        )
    if (
        getattr(args, "experiment_config", "").endswith("_nudity")
        and args.target_celebs_list != ["nudity", "no_nudity"]
    ):
        raise ValueError(
            "Nudity training requires --target-celebs nudity,no_nudity in "
            "that order."
        )
    return args
