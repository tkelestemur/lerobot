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

"""DIT (Diffusion Transformer) Policy
This implementation is based on the "Diffusion Transformer Block Policy" paper
and adapted for the LeRobot framework.
"""

import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from torch import Tensor

from lerobot.policies.dit.configuration_dit import DiTConfig
from lerobot.policies.normalize import Normalize, Unnormalize
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return nn.GELU(approximate="tanh")
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def _with_pos_embed(tensor, pos=None):
    return tensor if pos is None else tensor + pos


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe[: x.shape[0]]
        pe = pe.repeat((1, x.shape[1], 1))
        return pe.detach().clone()


class _TimeNetwork(nn.Module):
    def __init__(self, time_dim, out_dim, learnable_w=False):
        assert time_dim % 2 == 0, "time_dim must be even!"
        half_dim = int(time_dim // 2)
        super().__init__()

        w = np.log(10000) / (half_dim - 1)
        w = torch.exp(torch.arange(half_dim) * -w).float()
        self.register_parameter("w", nn.Parameter(w, requires_grad=learnable_w))

        self.out_net = nn.Sequential(nn.Linear(time_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, x):
        assert len(x.shape) == 1, "assumes 1d input timestep array"
        x = x[:, None] * self.w[None]
        x = torch.cat((torch.cos(x), torch.sin(x)), dim=1)
        return self.out_net(x)


class _SelfAttnEncoder(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward(self, src, pos):
        q = k = _with_pos_embed(src, pos)
        src2, _ = self.self_attn(q, k, value=src, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class _ShiftScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None] + self.shift(c)[None]

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.scale.weight)
        nn.init.xavier_uniform_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class _ZeroScaleMod(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x, c):
        c = self.act(c)
        return x * self.scale(c)[None]

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


class _DiTDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        self.attn_mod1 = _ShiftScaleMod(d_model)
        self.attn_mod2 = _ZeroScaleMod(d_model)
        self.mlp_mod1 = _ShiftScaleMod(d_model)
        self.mlp_mod2 = _ZeroScaleMod(d_model)

    def forward(self, x, t, cond):
        cond = torch.mean(cond, axis=0)
        cond = cond + t

        x2 = self.attn_mod1(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, need_weights=False)
        x = self.attn_mod2(self.dropout1(x2), cond) + x

        x2 = self.mlp_mod1(self.norm2(x), cond)
        x2 = self.linear2(self.dropout2(self.activation(self.linear1(x2))))
        x2 = self.mlp_mod2(self.dropout3(x2), cond)
        return x + x2

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for s in (self.attn_mod1, self.attn_mod2, self.mlp_mod1, self.mlp_mod2):
            s.reset_parameters()


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_size, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, t, cond):
        cond = torch.mean(cond, axis=0)
        cond = cond + t

        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=1)
        x = x * scale[None] + shift[None]
        x = self.linear(x)
        return x.transpose(0, 1)

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


class _TransformerEncoder(nn.Module):
    def __init__(self, base_module, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(base_module) for _ in range(num_layers)])

        for l in self.layers:
            l.reset_parameters()

    def forward(self, src, pos):
        x, outputs = src, []
        for layer in self.layers:
            x = layer(x, pos)
            outputs.append(x)
        return outputs


class _TransformerDecoder(_TransformerEncoder):
    def forward(self, src, t, all_conds):
        x = src
        for layer, cond in zip(self.layers, all_conds, strict=False):
            x = layer(x, t, cond)
        return x


class _DiTNoiseNet(nn.Module):
    def __init__(
        self,
        ac_dim,
        ac_chunk,
        time_dim=256,
        hidden_dim=512,
        num_blocks=6,
        dropout=0.1,
        dim_feedforward=2048,
        nhead=8,
        activation="gelu",
    ):
        super().__init__()

        self.enc_pos = _PositionalEncoding(hidden_dim)
        self.register_parameter(
            "dec_pos",
            nn.Parameter(torch.empty(ac_chunk, 1, hidden_dim), requires_grad=True),
        )
        nn.init.xavier_uniform_(self.dec_pos.data)

        self.time_net = _TimeNetwork(time_dim, hidden_dim)
        self.ac_proj = nn.Sequential(
            nn.Linear(ac_dim, ac_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ac_dim, hidden_dim),
        )

        encoder_module = _SelfAttnEncoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.encoder = _TransformerEncoder(encoder_module, num_blocks)

        decoder_module = _DiTDecoder(
            hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.decoder = _TransformerDecoder(decoder_module, num_blocks)

        self.eps_out = _FinalLayer(hidden_dim, ac_dim)

    def forward(self, noise_actions, time, obs_enc, enc_cache=None):
        if enc_cache is None:
            enc_cache = self.forward_enc(obs_enc)
        return enc_cache, self.forward_dec(noise_actions, time, enc_cache)

    def forward_enc(self, obs_enc):
        obs_enc = obs_enc.transpose(0, 1)
        pos = self.enc_pos(obs_enc)
        enc_cache = self.encoder(obs_enc, pos)
        return enc_cache

    def forward_dec(self, noise_actions, time, enc_cache):
        time_enc = self.time_net(time)

        ac_tokens = self.ac_proj(noise_actions)
        ac_tokens = ac_tokens.transpose(0, 1)
        dec_in = ac_tokens + self.dec_pos

        dec_out = self.decoder(dec_in, time_enc, enc_cache)
        return self.eps_out(dec_out, time_enc, enc_cache[-1])


class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, num_keypoints, temperature=1.0):
        super().__init__()
        self.height = height
        self.width = width
        self.num_keypoints = num_keypoints
        self.temperature = temperature

        pos_x, pos_y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        pos_x = torch.from_numpy(pos_x.reshape(1, height * width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(1, height * width)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, x):
        assert x.shape[2] == self.height and x.shape[3] == self.width
        x = x.view(x.shape[0], x.shape[1], self.height * self.width)
        attention = F.softmax(x / self.temperature, dim=2)
        expected_x = torch.sum(self.pos_x * attention, dim=2, keepdim=True)
        expected_y = torch.sum(self.pos_y * attention, dim=2, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], dim=2)
        feature_keypoints = expected_xy.view(x.shape[0], self.num_keypoints * 2)
        return feature_keypoints


class DITPolicy(PreTrainedPolicy):
    """
    DIT (Diffusion Transformer) Policy
    This policy implements the Diffusion Transformer Block Policy architecture,
    which combines transformer-based observation encoding with diffusion-based
    action generation.
    """

    def __init__(
        self,
        config: DiTConfig,
        dataset_stats: dict[str, dict[str, Tensor]],
    ):
        super().__init__(config, dataset_stats)

        self.config = config
        self.normalize_inputs = Normalize(
            config.input_shapes, config.input_normalization_modes, dataset_stats
        )
        self.normalize_targets = Normalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_shapes, config.output_normalization_modes, dataset_stats
        )

        self.n_obs_steps = config.n_obs_steps
        self.horizon = config.horizon
        self.n_action_steps = config.n_action_steps

        # Vision encoder
        self.vision_encoder = self._make_vision_encoder()

        # Get visual features dimension
        with torch.no_grad():
            dummy_img = torch.randn(1, 3, *config.crop_shape if config.crop_shape else (224, 224))
            visual_features = self.vision_encoder(dummy_img)
            self.visual_features_dim = visual_features.shape[1]

        # Combine visual and state features
        state_dim = config.input_shapes.get("observation.state", [0])[0]
        self.obs_encoder = nn.Sequential(
            nn.Linear(self.visual_features_dim + state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Action dimension
        self.action_dim = config.output_shapes["action"][0]

        # Diffusion noise network
        self.noise_net = _DiTNoiseNet(
            ac_dim=self.action_dim,
            ac_chunk=self.horizon,
            time_dim=config.time_dim,
            hidden_dim=config.hidden_dim,
            num_blocks=config.num_blocks,
            dropout=config.dropout,
            dim_feedforward=config.dim_feedforward,
            nhead=config.nhead,
            activation=config.activation,
        )

        # Diffusion scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=config.train_diffusion_steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type=config.prediction_type,
        )

        # Observation buffer for n_obs_steps
        self.obs_buffer = deque(maxlen=self.n_obs_steps)

        self.reset()

    def _make_vision_encoder(self):
        """Create vision encoder based on config."""
        backbone = getattr(torchvision.models, self.config.vision_backbone)

        if self.config.pretrained_backbone_weights:
            backbone = backbone(weights=self.config.pretrained_backbone_weights)
        else:
            backbone = backbone()

        # Remove the final classification layer
        if self.config.vision_backbone.startswith("resnet"):
            backbone = nn.Sequential(*list(backbone.children())[:-2])

        # Add spatial softmax
        if self.config.crop_shape:
            h, w = self.config.crop_shape
        else:
            h, w = 224, 224

        # Calculate feature map size after backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, h, w)
            feature_map = backbone(dummy_input)
            feature_h, feature_w = feature_map.shape[-2:]

        spatial_softmax = SpatialSoftmax(
            height=feature_h,
            width=feature_w,
            num_keypoints=self.config.spatial_softmax_num_keypoints,
        )

        return nn.Sequential(backbone, spatial_softmax)

    def reset(self):
        """Reset the policy state."""
        self.obs_buffer.clear()

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select action using the policy."""
        batch = self.normalize_inputs(batch)

        # Process observations
        obs_sequence = []
        for i in range(self.n_obs_steps):
            obs_dict = {key: batch[key][:, i] for key in batch if key.startswith("observation")}
            obs_sequence.append(obs_dict)

        # Generate actions
        actions = self._generate_actions(obs_sequence)

        # Unnormalize actions
        actions = self.unnormalize_outputs({"action": actions})["action"]

        return actions

    def _generate_actions(self, obs_sequence: list[dict[str, Tensor]]) -> Tensor:
        """Generate actions using diffusion sampling."""
        batch_size = obs_sequence[0]["observation.state"].shape[0]
        device = get_device_from_parameters(self)

        # Encode observations
        obs_encodings = []
        for obs_dict in obs_sequence:
            # Process images
            visual_features = None
            for key, value in obs_dict.items():
                if key.startswith("observation.image"):
                    if self.config.crop_shape:
                        # Apply center crop during inference
                        h, w = self.config.crop_shape
                        _, _, orig_h, orig_w = value.shape
                        top = (orig_h - h) // 2
                        left = (orig_w - w) // 2
                        value = value[:, :, top : top + h, left : left + w]

                    img_features = self.vision_encoder(value)
                    if visual_features is None:
                        visual_features = img_features
                    else:
                        visual_features = torch.cat([visual_features, img_features], dim=1)

            # Process state
            state = obs_dict.get("observation.state", torch.zeros(batch_size, 0, device=device))

            # Combine visual and state features
            if visual_features is not None:
                combined_features = torch.cat([visual_features, state], dim=1)
            else:
                combined_features = state

            obs_encoding = self.obs_encoder(combined_features)
            obs_encodings.append(obs_encoding)

        # Stack observations
        obs_encoded = torch.stack(obs_encodings, dim=1)  # [batch, n_obs_steps, hidden_dim]

        # Generate actions via diffusion
        noise_actions = torch.randn(batch_size, self.horizon, self.action_dim, device=device)

        # Set timesteps for inference
        self.noise_scheduler.set_timesteps(self.config.eval_diffusion_steps)
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)

        # Encode observations once
        enc_cache = self.noise_net.forward_enc(obs_encoded.reshape(-1, obs_encoded.shape[-1]))

        # Diffusion sampling loop
        for timestep in self.noise_scheduler.timesteps:
            batched_timestep = timestep.unsqueeze(0).repeat(batch_size).to(device)

            # Predict noise
            noise_pred = self.noise_net.forward_dec(noise_actions, batched_timestep, enc_cache)

            # Take diffusion step
            noise_actions = self.noise_scheduler.step(
                model_output=noise_pred, timestep=timestep, sample=noise_actions
            ).prev_sample

        return noise_actions

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Forward pass for training."""
        batch = self.normalize_inputs(batch)
        targets = self.normalize_targets(batch)

        batch_size = batch["observation.state"].shape[0]
        device = get_device_from_parameters(self)

        # Process observations for all time steps
        obs_encodings = []
        for i in range(self.n_obs_steps):
            obs_dict = {key: batch[key][:, i] for key in batch if key.startswith("observation")}

            # Process images
            visual_features = None
            for key, value in obs_dict.items():
                if key.startswith("observation.image"):
                    if self.config.crop_shape:
                        # Apply random crop during training
                        if self.config.crop_is_random and self.training:
                            h, w = self.config.crop_shape
                            _, _, orig_h, orig_w = value.shape
                            top = torch.randint(0, orig_h - h + 1, (1,)).item()
                            left = torch.randint(0, orig_w - w + 1, (1,)).item()
                            value = value[:, :, top : top + h, left : left + w]
                        else:
                            # Center crop
                            h, w = self.config.crop_shape
                            _, _, orig_h, orig_w = value.shape
                            top = (orig_h - h) // 2
                            left = (orig_w - w) // 2
                            value = value[:, :, top : top + h, left : left + w]

                    img_features = self.vision_encoder(value)
                    if visual_features is None:
                        visual_features = img_features
                    else:
                        visual_features = torch.cat([visual_features, img_features], dim=1)

            # Process state
            state = obs_dict.get("observation.state", torch.zeros(batch_size, 0, device=device))

            # Combine features
            if visual_features is not None:
                combined_features = torch.cat([visual_features, state], dim=1)
            else:
                combined_features = state

            obs_encoding = self.obs_encoder(combined_features)
            obs_encodings.append(obs_encoding)

        # Stack observations
        obs_encoded = torch.stack(obs_encodings, dim=1)

        # Get actions and add noise
        actions = targets["action"]  # [batch, horizon, action_dim]
        noise = torch.randn_like(actions)

        # Sample timesteps
        timesteps = torch.randint(0, self.config.train_diffusion_steps, (batch_size,), device=device).long()

        # Add noise to actions
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, timesteps)

        # Predict noise
        enc_cache, noise_pred = self.noise_net(
            noisy_actions, timesteps, obs_encoded.reshape(-1, obs_encoded.shape[-1])
        )

        # Compute loss
        loss = F.mse_loss(noise_pred, noise, reduction="none")

        # Apply masking if needed (for padded actions)
        if self.config.do_mask_loss_for_padding:
            # This would need to be implemented based on your specific masking logic
            pass

        return {
            "loss": loss.mean(),
            "action": self.unnormalize_outputs({"action": actions})["action"],
        }

    @property
    def expected_image_keys(self) -> list[str]:
        """Return expected image keys."""
        return [key for key in self.config.input_shapes.keys() if key.startswith("observation.image")]

    @property
    def device(self) -> torch.device:
        return get_device_from_parameters(self)
