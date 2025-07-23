#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("dit")
@dataclass
class DiTConfig(PreTrainedConfig):
    """Configuration class for DIT (Diffusion Transformer) Policy.
    This policy implements the Diffusion Transformer Block Policy architecture as described in the paper.
    It uses a transformer-based architecture with diffusion models for action generation.
    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and `output_shapes`.
    Args:
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        horizon: Number of action steps to predict (action chunk size).
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
        normalization_mapping: Dictionary specifying normalization modes for different data types.
        drop_n_last_frames: Number of frames to drop from the end during training to avoid excessive padding.

        # Vision processing
        vision_backbone: Name of the vision backbone to use for encoding images.
        crop_shape: (H, W) shape to crop images to before processing.
        crop_is_random: Whether to use random cropping during training.
        pretrained_backbone_weights: Whether to use pretrained weights for the vision backbone.
        use_group_norm: Whether to use group normalization in the vision backbone.
        spatial_softmax_num_keypoints: Number of keypoints for spatial softmax pooling.
        use_separate_rgb_encoder_per_camera: Whether to use separate encoders for each camera.

        # Transformer architecture
        hidden_dim: Hidden dimension of the transformer.
        time_dim: Dimension of time embeddings.
        num_blocks: Number of transformer blocks.
        nhead: Number of attention heads.
        dim_feedforward: Dimension of feedforward layers.
        dropout: Dropout probability.
        activation: Activation function to use.

        # Diffusion settings
        train_diffusion_steps: Number of diffusion steps during training.
        eval_diffusion_steps: Number of diffusion steps during evaluation.
        beta_schedule: Beta schedule for the diffusion process.
        beta_start: Starting beta value.
        beta_end: Ending beta value.
        clip_sample: Whether to clip samples during generation.
        prediction_type: Type of prediction ("epsilon" or "sample").

        # Training settings
        optimizer_lr: Learning rate for the optimizer.
        optimizer_betas: Beta parameters for Adam optimizer.
        optimizer_eps: Epsilon for Adam optimizer.
        optimizer_weight_decay: Weight decay for the optimizer.
        scheduler_name: Name of the learning rate scheduler.
        scheduler_warmup_steps: Number of warmup steps for the scheduler.
    """

    # Inputs / output structure
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    drop_n_last_frames: int = 7  # horizon - n_action_steps - n_obs_steps + 1

    # Vision processing
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False

    # Transformer architecture
    hidden_dim: int = 512
    time_dim: int = 256
    num_blocks: int = 6
    nhead: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "gelu"

    # Diffusion settings
    train_diffusion_steps: int = 1000
    eval_diffusion_steps: int = 50
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    clip_sample: bool = True
    prediction_type: str = "epsilon"

    # Training settings
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        # Input validation
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )

        supported_activations = ["relu", "gelu", "glu"]
        if self.activation not in supported_activations:
            raise ValueError(f"`activation` must be one of {supported_activations}. Got {self.activation}.")

        if self.eval_diffusion_steps > self.train_diffusion_steps:
            raise ValueError(
                f"`eval_diffusion_steps` ({self.eval_diffusion_steps}) cannot be greater than "
                f"`train_diffusion_steps` ({self.train_diffusion_steps})."
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )

        # Check that all input images have the same shape
        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
