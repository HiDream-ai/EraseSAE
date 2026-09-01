import torch

from model.attention_masks import (
    binarize_attention_mask,
    compute_cross_attention_mask,
)


class OnlineActivationCollector:
    """Capture visual activations and target-token attention for online training."""

    def __init__(
        self,
        model,
        target_layer_names,
        *,
        text_len,
        text_first,
        block_name_resolver,
    ):
        self.model = model
        self.target_layer_names = list(target_layer_names)
        self.TEXT_LEN = int(text_len)
        self.text_first = bool(text_first)
        self.block_name_resolver = block_name_resolver
        self.current_batch_activations = {
            name: [] for name in self.target_layer_names
        }
        self.temp_attn_buffer = {}
        self.capture_activations = True
        self.capture_attention = True
        self.hook_handles = []

    def register_hooks(self):
        if self.hook_handles:
            raise RuntimeError("Activation collector hooks are already registered.")

        modules = dict(self.model.named_modules())
        try:
            for layer_name in self.target_layer_names:
                if layer_name not in modules:
                    raise RuntimeError(f"Target layer not found: {layer_name}")

                block_name = self.block_name_resolver(layer_name)
                if block_name not in modules:
                    raise RuntimeError(
                        f"Parent transformer block not found for {layer_name}: "
                        f"{block_name}"
                    )

                self.hook_handles.append(
                    modules[layer_name].register_forward_hook(
                        self._make_activation_hook(layer_name)
                    )
                )

                found_q = False
                found_k = False
                for attention_name, module in modules[block_name].named_modules():
                    if attention_name.endswith("to_q"):
                        self.hook_handles.append(
                            module.register_forward_hook(
                                self._make_attention_hook(layer_name, "to_q")
                            )
                        )
                        found_q = True
                    elif attention_name.endswith("to_k"):
                        self.hook_handles.append(
                            module.register_forward_hook(
                                self._make_attention_hook(layer_name, "to_k")
                            )
                        )
                        found_k = True

                if not found_q or not found_k:
                    raise RuntimeError(
                        f"Could not find to_q/to_k modules under {block_name}."
                    )
        except Exception:
            self.remove_hooks(verbose=False)
            raise

        print(
            f"[Collector] Registered activation and attention hooks for "
            f"{len(self.target_layer_names)} layer(s)."
        )

    def _make_activation_hook(self, layer_name):
        def hook(module, inputs, output):
            del module, inputs
            if not self.capture_activations or output.shape[1] <= self.TEXT_LEN:
                return
            if self.text_first:
                visual_output = output[:, self.TEXT_LEN :, :]
            else:
                visual_output = output[:, : -self.TEXT_LEN, :]
            self.current_batch_activations[layer_name].append(
                visual_output.detach().cpu()
            )

        return hook

    def _make_attention_hook(self, layer_name, projection_name):
        def hook(module, inputs, output):
            del module, inputs
            if not self.capture_attention:
                return
            self.temp_attn_buffer.setdefault(layer_name, {})[
                projection_name
            ] = output.detach()

        return hook

    def compute_step_mask(
        self,
        word_indices,
        height,
        width,
        head_num=30,
        quantile=0.8,
        dilation=0,
    ):
        for layer_name, attention in self.temp_attn_buffer.items():
            if "to_q" not in attention or "to_k" not in attention:
                continue

            mask = compute_cross_attention_mask(
                query=attention["to_q"],
                key=attention["to_k"],
                token_idx=word_indices,
                head_num=head_num,
                text_len=self.TEXT_LEN,
                height=height,
                width=width,
                text_first=self.text_first,
            )
            if torch.isnan(mask).any() or torch.isinf(mask).any():
                print(
                    f"[Collector] Ignoring non-finite attention mask for "
                    f"{layer_name}."
                )
                continue

            self.current_batch_activations[layer_name].append(
                binarize_attention_mask(
                    mask,
                    quantile=quantile,
                    dilation=dilation,
                )
            )

        self.temp_attn_buffer = {}

    def remove_hooks(self, verbose=True):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
        if verbose:
            print("[Collector] Hooks removed.")


def create_hunyuan_activation_collector(model, target_layer_names):
    return OnlineActivationCollector(
        model,
        target_layer_names,
        text_len=256,
        text_first=False,
        block_name_resolver=lambda name: name.rsplit(".", 1)[0],
    )


def create_cog_activation_collector(model, target_layer_names):
    return OnlineActivationCollector(
        model,
        target_layer_names,
        text_len=226,
        text_first=True,
        block_name_resolver=lambda name: ".".join(name.split(".")[:2]),
    )
