"""Native DETRPose Transformer Decoder implementation.

Copyright (c) 2025 The DETRPose Authors. All Rights Reserved.
Modified for native integration into nitid in 2026.
"""

from __future__ import annotations

import copy
import math
from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DETRPoseDecoder"]


def weighting_function(
    reg_max: int, up: torch.Tensor, reg_scale: torch.Tensor, deploy: bool = False
) -> torch.Tensor:
    """Generates the non-uniform Weighting Function W(n) for keypoint offset regression."""
    if deploy:
        upper_bound1_v = float((abs(up[0]) * abs(reg_scale)).item())
        upper_bound2_v = float((abs(up[0]) * abs(reg_scale) * 2).item())
        step = (upper_bound1_v + 1) ** (2 / (reg_max - 2))
        left_values = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values_f: list[float] = (
            [-upper_bound2_v] + left_values + [0.0] + right_values + [upper_bound2_v]
        )
        return torch.tensor([values_f], dtype=up.dtype, device=up.device)
    else:
        upper_bound1 = abs(up[0]) * abs(reg_scale)
        upper_bound2 = abs(up[0]) * abs(reg_scale) * 2
        step = (upper_bound1 + 1) ** (2 / (reg_max - 2))
        left_values_t = [-((step) ** i) + 1 for i in range(reg_max // 2 - 1, 0, -1)]
        right_values_t = [(step) ** i - 1 for i in range(1, reg_max // 2)]
        values_t: list[torch.Tensor] = (
            [-upper_bound2]
            + left_values_t
            + [torch.zeros_like(up[0][None])]
            + right_values_t
            + [upper_bound2]
        )
        return torch.cat(values_t, 0)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    x_clamped = x.clamp(min=0, max=1)
    x1 = x_clamped.clamp(min=eps)
    x2 = (1 - x_clamped).clamp(min=eps)
    return torch.log(x1 / x2)


def distance2pose(
    points: torch.Tensor, distance: torch.Tensor, reg_scale: torch.Tensor | float
) -> torch.Tensor:
    """Decodes keypoint edge-distances into normalized coordinates."""
    reg_scale_val: float = (
        float(reg_scale)
        if isinstance(reg_scale, (int, float))
        else float(cast(torch.Tensor, reg_scale).item())
    )
    x1 = points[..., 0] + distance[..., 0] / reg_scale_val
    y1 = points[..., 1] + distance[..., 1] / reg_scale_val
    return torch.stack([x1, y1], -1)


class Gate(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(2 * d_model, 2 * d_model)
        bias = float(-math.log((1 - 0.5) / 0.5))
        nn.init.constant_(self.gate.bias, bias)
        nn.init.constant_(self.gate.weight, 0)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([x1, x2], dim=-1)
        gates = torch.sigmoid(self.gate(gate_input))
        gate1, gate2 = gates.chunk(2, dim=-1)
        return self.norm(gate1 * x1 + gate2 * x2)


class Integral(nn.Module):
    def __init__(self, reg_max: int = 32) -> None:
        super().__init__()
        self.reg_max = reg_max

    def forward(self, x: torch.Tensor, project: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_soft = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        weights = project.to(device=x.device, dtype=x_soft.dtype).view(1, -1)
        x_proj = (x_soft * weights).sum(dim=1).reshape(-1, 4)
        return x_proj.reshape(list(shape[:-1]) + [-1])


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, p) for n, p in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LQE(nn.Module):
    def __init__(self, topk: int, hidden_dim: int, num_layers: int, num_body_points: int) -> None:
        super().__init__()
        self.k = topk
        self.hidden_dim = hidden_dim
        self.reg_conf = MLP(num_body_points * (topk + 1), hidden_dim, 1, num_layers)
        last_layer = cast(nn.Linear, self.reg_conf.layers[-1])
        nn.init.constant_(last_layer.weight.data, 0)
        nn.init.constant_(last_layer.bias.data, 0)
        self.num_body_points = num_body_points

    def forward(
        self, scores: torch.Tensor, pred_poses: torch.Tensor, feat: torch.Tensor
    ) -> torch.Tensor:
        b, length = pred_poses.shape[:2]
        pred_poses = pred_poses.reshape(b, length, self.num_body_points, 2)
        grid = 2 * pred_poses - 1
        sampling_values = F.grid_sample(
            feat, grid, mode="bilinear", padding_mode="zeros", align_corners=False
        ).permute(0, 2, 3, 1)

        prob_topk = sampling_values.topk(self.k, dim=-1)[0]
        stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)
        quality_score = self.reg_conf(stat.reshape(b, length, -1))
        return scores + quality_score


def ms_deform_attn_core_pytorch(
    value: list[torch.Tensor],
    value_spatial_shapes: list[tuple[int, int]] | list[list[int]],
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    _, d_col, _ = value[0].shape
    n_bs, lq, m_head, l_lvl, p_pts, _ = sampling_locations.shape

    sampling_grids = 2 * sampling_locations - 1
    sampling_grids = sampling_grids.transpose(1, 2).flatten(0, 1)

    sampling_value_list = []
    for lid, (h_val, w_val) in enumerate(value_spatial_shapes):
        value_l = value[lid].unflatten(2, (h_val, w_val))
        sampling_grid_l = sampling_grids[:, :, lid]
        sampling_value_l = F.grid_sample(
            value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.transpose(1, 2).reshape(n_bs * m_head, 1, lq, l_lvl * p_pts)
    output = (
        (torch.concat(sampling_value_list, dim=-1) * attn_weights)
        .sum(-1)
        .view(n_bs, m_head * d_col, lq)
    )
    return output.transpose(1, 2)


class MSDeformAttn(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
        use_4d_normalizer: bool = False,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got {d_model} and {n_heads}")

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.use_4d_normalizer = use_4d_normalizer

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i % 4 + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        if self.n_points % 4 != 0:
            nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight.data, 0.0)
        nn.init.constant_(self.attention_weights.bias.data, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: list[torch.Tensor],
        input_spatial_shapes: list[tuple[int, int]] | list[list[int]],
    ) -> torch.Tensor:
        n_bs, len_q, _ = query.shape

        sampling_offsets = self.sampling_offsets(query).view(
            n_bs, len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            n_bs, len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            n_bs, len_q, self.n_heads, self.n_levels, self.n_points
        )

        ref_points = torch.transpose(reference_points, 2, 3).flatten(1, 2)

        if ref_points.shape[-1] == 2:
            offset_normalizer = torch.tensor(input_spatial_shapes, device=query.device)
            offset_normalizer = offset_normalizer.flip([1]).reshape(1, 1, 1, self.n_levels, 1, 2)
            sampling_locations = (
                ref_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer
            )
        elif ref_points.shape[-1] == 4:
            if self.use_4d_normalizer:
                shapes_tensor = torch.tensor(input_spatial_shapes, device=query.device)
                offset_normalizer = torch.stack([shapes_tensor[..., 1], shapes_tensor[..., 0]], -1)
                sampling_locations = (
                    ref_points[:, :, None, :, None, :2]
                    + sampling_offsets
                    / offset_normalizer[None, None, None, :, None, :]
                    * ref_points[:, :, None, :, None, 2:]
                    * 0.5
                )
            else:
                sampling_locations = (
                    ref_points[:, :, None, :, None, :2]
                    + sampling_offsets / self.n_points * ref_points[:, :, None, :, None, 2:] * 0.5
                )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, got {ref_points.shape[-1]}"
            )

        return ms_deform_attn_core_pytorch(
            value, input_spatial_shapes, sampling_locations, attention_weights
        )


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ) -> None:
        super().__init__()
        self.within_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.within_dropout = nn.Dropout(dropout)
        self.within_norm = nn.LayerNorm(d_model)

        self.across_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.across_dropout = nn.Dropout(dropout)
        self.across_norm = nn.LayerNorm(d_model)

        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)

        self.gateway = Gate(d_model)

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    @staticmethod
    def with_pos_embed(tensor: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
        if pos is not None:
            n_p = pos.shape[2]
            tensor = tensor.clone()
            tensor[:, :, -n_p:] = tensor[:, :, -n_p:] + pos
        return tensor

    def forward_ffn(self, tgt: torch.Tensor) -> torch.Tensor:
        tgt2 = self.linear2(self.dropout2(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return self.norm2(tgt.clamp(min=-65504, max=65504))

    def forward(
        self,
        tgt_pose: torch.Tensor,
        tgt_pose_query_pos: torch.Tensor | None,
        tgt_pose_reference_points: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        memory: list[torch.Tensor] | None = None,
        memory_spatial_shapes: list[tuple[int, int]] | list[list[int]] | None = None,
    ) -> torch.Tensor:
        bs, nq, num_kpt, d_model = tgt_pose.shape

        # Within-instance self-attention
        q_w = k_w = self.with_pos_embed(tgt_pose, tgt_pose_query_pos).flatten(0, 1)
        tgt2 = self.within_attn(q_w, k_w, tgt_pose.flatten(0, 1))[0].reshape(
            bs, nq, num_kpt, d_model
        )
        tgt_pose = tgt_pose + self.within_dropout(tgt2)
        tgt_pose = self.within_norm(tgt_pose)

        # Across-instance self-attention
        tgt_pose = tgt_pose.transpose(1, 2).flatten(0, 1)
        q_pose = k_pose = tgt_pose
        tgt2_pose = self.across_attn(q_pose, k_pose, tgt_pose, attn_mask=attn_mask)[0].reshape(
            bs * num_kpt, nq, d_model
        )
        tgt_pose = tgt_pose + self.across_dropout(tgt2_pose)
        tgt_pose = self.across_norm(tgt_pose).reshape(bs, num_kpt, nq, d_model).transpose(1, 2)

        # Deformable cross-attention
        tgt2_cross = self.cross_attn(
            self.with_pos_embed(tgt_pose, tgt_pose_query_pos).flatten(1, 2),
            tgt_pose_reference_points,
            memory,
            memory_spatial_shapes,
        ).reshape(bs, nq, num_kpt, d_model)

        tgt_pose = self.gateway(tgt_pose, self.dropout1(tgt2_cross))
        return self.forward_ffn(tgt_pose)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: DeformableTransformerDecoderLayer,
        num_layers: int,
        return_intermediate: bool = False,
        hidden_dim: int = 256,
        num_body_points: int = 17,
    ) -> None:
        super().__init__()
        self.layers = (
            nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
            if num_layers > 0
            else nn.ModuleList()
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_body_points = num_body_points
        self.return_intermediate = return_intermediate
        self.half_pose_ref_point_head = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.eval_idx = num_layers - 1

        dim_t = torch.arange(hidden_dim // 2, dtype=torch.float32)
        dim_t = 10000 ** (2 * (dim_t // 2) / (hidden_dim // 2))
        self.register_buffer("dim_t", dim_t)
        self.scale = 2 * math.pi

    def sine_embedding(self, pos_tensor: torch.Tensor) -> torch.Tensor:
        x_embed = pos_tensor[..., 0:1] * self.scale
        y_embed = pos_tensor[..., 1:2] * self.scale
        dim_t_tensor = cast(torch.Tensor, self.dim_t)
        pos_x = x_embed / dim_t_tensor
        pos_y = y_embed / dim_t_tensor
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)

        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=3)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[..., 2:3] * self.scale
            pos_w = w_embed / dim_t_tensor
            pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=4).flatten(3)
            h_embed = pos_tensor[..., 3:4] * self.scale
            pos_h = h_embed / dim_t_tensor
            pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=4).flatten(3)
            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=3)
        else:
            raise ValueError(f"Unknown pos_tensor shape(-1): {pos_tensor.size(-1)}")
        return pos

    def forward(
        self,
        tgt: torch.Tensor,
        memory: list[torch.Tensor],
        refpoints_sigmoid: torch.Tensor,
        pre_pose_head: nn.Module,
        pose_head: nn.ModuleList,
        class_head: nn.ModuleList,
        lqe_head: nn.ModuleList,
        feat_lqe: torch.Tensor,
        integral: nn.Module,
        up: torch.Tensor,
        reg_scale: torch.Tensor,
        reg_max: int,
        project: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        spatial_shapes: list[tuple[int, int]] | list[list[int]] | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        output = tgt
        refpoint_pose = refpoints_sigmoid
        output_pose_detach = pred_corners_undetach = 0

        dec_out_poses = []
        dec_out_logits = []
        dec_out_refs = []
        dec_out_pred_corners = []

        pre_poses = torch.tensor(0)
        pre_scores = torch.tensor(0)
        ref_pose_initial = torch.tensor(0)

        for layer_id, layer in enumerate(self.layers):
            refpoint_pose_input = refpoint_pose[:, :, None]
            refpoint_only_pose = refpoint_pose[:, :, 1:]
            pose_query_sine_embed = self.sine_embedding(refpoint_only_pose)
            pose_query_pos = self.half_pose_ref_point_head(pose_query_sine_embed)

            output = layer(
                tgt_pose=output,
                tgt_pose_query_pos=pose_query_pos,
                tgt_pose_reference_points=refpoint_pose_input,
                attn_mask=attn_mask,
                memory=memory,
                memory_spatial_shapes=spatial_shapes,
            )

            output_pose = output[:, :, 1:]
            output_instance = output[:, :, 0]

            if layer_id == 0:
                pre_poses = torch.sigmoid(
                    pre_pose_head(output_pose) + inverse_sigmoid(refpoint_only_pose)
                )
                pre_scores = class_head[0](output_instance)
                ref_pose_initial = pre_poses.detach()

            pred_corners = (
                pose_head[layer_id](output_pose + output_pose_detach) + pred_corners_undetach
            )
            refpoint_pose_without_center = distance2pose(
                ref_pose_initial, integral(pred_corners, project), reg_scale
            )

            refpoint_center_pose = torch.mean(refpoint_pose_without_center, dim=2, keepdim=True)
            refpoint_pose = torch.cat([refpoint_center_pose, refpoint_pose_without_center], dim=2)

            if self.training or layer_id == self.eval_idx:
                score = class_head[layer_id](output_instance)
                logit = lqe_head[layer_id](score, refpoint_pose_without_center, feat_lqe)
                dec_out_logits.append(logit)
                dec_out_poses.append(refpoint_pose_without_center)
                dec_out_pred_corners.append(pred_corners)
                dec_out_refs.append(ref_pose_initial)

                if not self.training:
                    break

            pred_corners_undetach = pred_corners
            if self.training:
                refpoint_pose = refpoint_pose.detach()
                output_pose_detach = output_pose.detach()
            else:
                refpoint_pose = refpoint_pose
                output_pose_detach = output_pose

        return (
            torch.stack(dec_out_poses),
            torch.stack(dec_out_logits),
            torch.stack(dec_out_pred_corners),
            torch.stack(dec_out_refs),
            pre_poses,
            pre_scores,
        )


class DETRPoseDecoder(nn.Module):
    """Native DETRPose Decoder component."""

    def __init__(
        self,
        hidden_dim: int = 256,
        nhead: int = 8,
        num_queries: int = 60,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        activation: str = "relu",
        num_feature_levels: int = 3,
        dec_n_points: int = 4,
        num_classes: int = 2,
        aux_loss: bool = True,
        num_body_points: int = 17,
        learnable_tgt_init: bool = True,
        feat_strides: tuple[int, ...] = (8, 16, 32),
        eval_spatial_size: tuple[int, int] | None = (640, 640),
        reg_max: int = 32,
        reg_scale: float = 4.0,
    ) -> None:
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.aux_loss = aux_loss
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_body_points = num_body_points
        self.learnable_tgt_init = learnable_tgt_init

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=hidden_dim,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=dec_n_points,
        )

        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            return_intermediate=False,
            hidden_dim=hidden_dim,
            num_body_points=num_body_points,
        )

        self.keypoint_embedding = nn.Embedding(num_body_points, hidden_dim)
        self.instance_embedding = nn.Embedding(1, hidden_dim)

        self.tgt_embed: nn.Embedding | None = None
        if learnable_tgt_init:
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)

        self.label_enc = nn.Embedding(80 + 1, hidden_dim)
        self.pose_enc = nn.Embedding(num_body_points, hidden_dim)

        self.enc_output = nn.Linear(hidden_dim, hidden_dim)
        self.enc_output_norm = nn.LayerNorm(hidden_dim)

        _class_embed = nn.Linear(hidden_dim, num_classes)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value

        _pre_point_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        _pre_last = cast(nn.Linear, _pre_point_embed.layers[-1])
        nn.init.constant_(_pre_last.weight.data, 0)
        nn.init.constant_(_pre_last.bias.data, 0)

        _point_embed = MLP(hidden_dim, hidden_dim, 2 * (reg_max + 1), 3)
        _point_last = cast(nn.Linear, _point_embed.layers[-1])
        nn.init.constant_(_point_last.weight.data, 0)
        nn.init.constant_(_point_last.bias.data, 0)

        _lqe_embed = LQE(4, 256, 2, num_body_points)

        self.class_embed = nn.ModuleList(
            [copy.deepcopy(_class_embed) for _ in range(num_decoder_layers)]
        )
        self.pose_embed = nn.ModuleList(
            [copy.deepcopy(_point_embed) for _ in range(num_decoder_layers)]
        )
        self.lqe_embed = nn.ModuleList(
            [copy.deepcopy(_lqe_embed) for _ in range(num_decoder_layers)]
        )
        self.pre_pose_embed = _pre_point_embed
        self.integral = Integral(reg_max)

        self.up = nn.Parameter(torch.tensor([0.5]), requires_grad=False)
        self.reg_max = reg_max
        self.reg_scale = nn.Parameter(torch.tensor([reg_scale]), requires_grad=False)
        self.deploy = False

        _keypoint_embed = MLP(hidden_dim, 2 * hidden_dim, 2 * num_body_points, 4)
        _kpt_last = cast(nn.Linear, _keypoint_embed.layers[-1])
        nn.init.constant_(_kpt_last.weight.data, 0)
        nn.init.constant_(_kpt_last.bias.data, 0)
        self.enc_pose_embed = _keypoint_embed
        self.enc_out_class_embed = _class_embed

        self.feat_strides = feat_strides
        self.eval_spatial_size = eval_spatial_size
        if self.eval_spatial_size is not None:
            anchors, valid_mask = self._generate_anchors()
            self.register_buffer("anchors", anchors)
            self.register_buffer("valid_mask", valid_mask)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def _get_encoder_input(
        self, feats: list[torch.Tensor]
    ) -> tuple[torch.Tensor, list[list[int]], list[int]]:
        feat_flatten = []
        spatial_shapes = []
        split_sizes = []

        for feat in feats:
            _, _, h_val, w_val = feat.shape
            feat_flatten.append(feat.flatten(2).permute(0, 2, 1))
            spatial_shapes.append([h_val, w_val])
            split_sizes.append(h_val * w_val)

        return torch.concat(feat_flatten, 1), spatial_shapes, split_sizes

    def _generate_anchors(
        self, spatial_shapes: list[list[int]] | None = None, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if spatial_shapes is None:
            spatial_shapes = []
            if self.eval_spatial_size is not None:
                eval_h, eval_w = self.eval_spatial_size
                for s in self.feat_strides:
                    spatial_shapes.append([int(eval_h / s), int(eval_w / s)])

        anchors = []
        for lvl, (h_val, w_val) in enumerate(spatial_shapes):
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, h_val - 1, h_val, dtype=torch.float32, device=device),
                torch.linspace(0, w_val - 1, w_val, dtype=torch.float32, device=device),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y], -1)
            grid = (grid.unsqueeze(0).expand(1, -1, -1, -1) + 0.5) / torch.tensor(
                [w_val, h_val], dtype=torch.float32, device=device
            )
            anchors.append(grid.view(1, -1, 2))
        anchors_cat = torch.cat(anchors, 1)
        valid_mask = ((anchors_cat > 0.01) & (anchors_cat < 0.99)).all(-1, keepdim=True)
        anchors_unsig = torch.log(anchors_cat / (1 - anchors_cat))
        return anchors_unsig, ~valid_mask

    def convert_to_deploy(self) -> None:
        self.project = weighting_function(self.reg_max, self.up, self.reg_scale, deploy=True)
        self.lqe_embed = nn.ModuleList(
            [nn.Identity()] * (self.num_decoder_layers - 1) + [self.lqe_embed[-1]]
        )
        self.deploy = True

    def forward(
        self,
        feats: list[torch.Tensor],
        targets: list | dict | None = None,
        samples: torch.Tensor | None = None,
        low_level_feat: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        memory, spatial_shapes, split_sizes = self._get_encoder_input(feats)

        if self.training:
            output_proposals, valid_mask = self._generate_anchors(spatial_shapes, memory.device)
            output_memory = memory.masked_fill(valid_mask, float(0))
            output_proposals = output_proposals.repeat(memory.size(0), 1, 1)
        else:
            anchors_buf = cast(torch.Tensor, self.anchors)
            valid_mask_buf = cast(torch.Tensor, self.valid_mask)
            if (
                anchors_buf.shape[1] != memory.shape[1]
                or valid_mask_buf.shape[1] != memory.shape[1]
            ):
                anchors_buf, valid_mask_buf = self._generate_anchors(spatial_shapes, memory.device)
            output_proposals = anchors_buf.repeat(memory.size(0), 1, 1)
            output_memory = memory.masked_fill(valid_mask_buf, float(0))

        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        topk = self.num_queries
        enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
        topk_idx = torch.topk(enc_outputs_class_unselected.max(-1)[0], topk, dim=1)[1]

        topk_memory = output_memory.gather(
            dim=1, index=topk_idx.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )
        topk_anchors = output_proposals.gather(dim=1, index=topk_idx.unsqueeze(-1).repeat(1, 1, 2))

        bs, nq = topk_memory.shape[:2]
        delta_unsig_keypoint = self.enc_pose_embed(topk_memory).reshape(
            bs, nq, self.num_body_points, 2
        )
        enc_outputs_pose_coord = torch.sigmoid(delta_unsig_keypoint + topk_anchors.unsqueeze(-2))
        enc_outputs_center_coord = torch.mean(enc_outputs_pose_coord, dim=2, keepdim=True)
        enc_outputs_pose_coord = torch.cat(
            [enc_outputs_center_coord, enc_outputs_pose_coord], dim=2
        )
        refpoint_pose_sigmoid = enc_outputs_pose_coord.detach()

        if self.learnable_tgt_init and self.tgt_embed is not None:
            topk_mem_detached = (
                self.tgt_embed.weight.unsqueeze(0).repeat([memory.size(0), 1, 1]).unsqueeze(-2)
            )
        else:
            topk_mem_detached = topk_memory.detach().unsqueeze(-2)

        tgt_pose = (
            self.keypoint_embedding.weight[None, None].repeat(1, topk, 1, 1).expand(bs, -1, -1, -1)
            + topk_mem_detached
        )
        tgt_global = (
            self.instance_embedding.weight[None, None].repeat(1, topk, 1, 1).expand(bs, -1, -1, -1)
        )
        tgt_pose = torch.cat([tgt_global, tgt_pose], dim=2)

        value = memory.unflatten(2, (self.nhead, -1))
        value_split = value.permute(0, 2, 3, 1).flatten(0, 1).split(split_sizes, dim=-1)

        attn_mask = None
        project: torch.Tensor
        if hasattr(self, "project"):
            project = cast(torch.Tensor, self.project)
        else:
            project = weighting_function(self.reg_max, self.up, self.reg_scale)

        out_poses, out_logits, out_corners, out_refs, out_pre_poses, out_pre_scores = self.decoder(
            tgt=tgt_pose,
            memory=value_split,
            refpoints_sigmoid=refpoint_pose_sigmoid,
            spatial_shapes=spatial_shapes,
            attn_mask=attn_mask,
            pre_pose_head=self.pre_pose_embed,
            pose_head=self.pose_embed,
            class_head=self.class_embed,
            lqe_head=self.lqe_embed,
            feat_lqe=feats[0],
            up=self.up,
            reg_max=self.reg_max,
            reg_scale=self.reg_scale,
            integral=self.integral,
            project=project,
        )

        if not self.deploy:
            out_poses_flat = out_poses.flatten(-2)
        else:
            out_poses_flat = out_poses

        return {"pred_logits": out_logits[-1], "pred_keypoints": out_poses_flat[-1]}
