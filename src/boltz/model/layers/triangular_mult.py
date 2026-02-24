import logging
import os
from typing import Optional

import torch
from torch import Tensor, nn

from boltz.model.layers import initialize as init
from boltz.model.layers.triangular_attention.utils import (
    is_fp16_enabled,
    permute_final_dims,
)

try:
    from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
except ImportError:
    triangle_multiplicative_update = None

LOGGER = logging.getLogger(__name__)


class InvalidTriangleMultiplicativeError(ValueError):
    """Raised when triangle_multiplicative mode is invalid."""

    def __init__(self, mode: str) -> None:
        message = (
            "triangle_multiplicative must be 'cuequivariance' or 'torch', "
            f"but got {mode}"
        )
        super().__init__(message)


class MissingCuequivarianceError(ImportError):
    """Raised when cuequivariance backend is requested but unavailable."""

    def __init__(self) -> None:
        message = (
            "cuequivariance_torch is not installed; "
            "set triangle_multiplicative='torch'"
        )
        super().__init__(message)


def kernel_triangular_mult(
    x: Tensor,
    direction: str,
    mask: Tensor,
    norm_in_weight: Tensor,
    norm_in_bias: Tensor,
    p_in_weight: Tensor,
    g_in_weight: Tensor,
    norm_out_weight: Tensor,
    norm_out_bias: Tensor,
    p_out_weight: Tensor,
    g_out_weight: Tensor,
    eps: float,
) -> Tensor:
    """Run cuequivariance triangle multiplicative update kernel."""
    if triangle_multiplicative_update is None:
        raise MissingCuequivarianceError

    return triangle_multiplicative_update(
        x,
        direction=direction,
        mask=mask,
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        g_in_weight=g_in_weight,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        g_out_weight=g_out_weight,
        eps=eps,
    )


class _TriangleMultiplication(nn.Module):
    def __init__(self, dim: int, outgoing: bool) -> None:
        super().__init__()
        self.dim = dim
        self.outgoing = outgoing
        self._debug_logged = False

        self.norm_in = nn.LayerNorm(dim, eps=1e-5)
        self.p_in = nn.Linear(dim, 2 * dim, bias=False)
        self.g_in = nn.Linear(dim, 2 * dim, bias=False)

        self.norm_out = nn.LayerNorm(dim)
        self.p_out = nn.Linear(dim, dim, bias=False)
        self.g_out = nn.Linear(dim, dim, bias=False)

        init.bias_init_one_(self.norm_in.weight)
        init.bias_init_zero_(self.norm_in.bias)
        init.lecun_normal_init_(self.p_in.weight)
        init.gating_init_(self.g_in.weight)

        init.bias_init_one_(self.norm_out.weight)
        init.bias_init_zero_(self.norm_out.bias)
        init.final_init_(self.p_out.weight)
        init.gating_init_(self.g_out.weight)

    def _combine_projections(
        self,
        a: Tensor,
        b: Tensor,
        _inplace_chunk_size: Optional[int] = None,
    ) -> Tensor:
        if self.outgoing:
            a = permute_final_dims(a, [2, 0, 1])
            b = permute_final_dims(b, [2, 1, 0])
        else:
            a = permute_final_dims(a, [2, 1, 0])
            b = permute_final_dims(b, [2, 0, 1])

        if _inplace_chunk_size is not None:
            for i in range(0, a.shape[-3], _inplace_chunk_size):
                a_chunk = a[..., i : i + _inplace_chunk_size, :, :]
                b_chunk = b[..., i : i + _inplace_chunk_size, :, :]
                a[..., i : i + _inplace_chunk_size, :, :] = torch.matmul(
                    a_chunk,
                    b_chunk,
                )
            p = a
        else:
            p = torch.matmul(a, b)

        return permute_final_dims(p, [1, 2, 0])

    def _project_input(
        self,
        x: Tensor,
        triangle_mult_gate_nchunks: int,
    ) -> Tensor:
        if triangle_mult_gate_nchunks <= 1:
            return self.p_in(x) * self.g_in(x).sigmoid()

        chunk_sizes = torch.linspace(
            0,
            x.shape[2],
            steps=triangle_mult_gate_nchunks + 1,
            device=x.device,
        ).long()
        proj = torch.empty(
            (*x.shape[:-1], x.shape[-1] * 2),
            device=x.device,
            dtype=x.dtype,
        )
        for i in range(triangle_mult_gate_nchunks):
            start = chunk_sizes[i].item()
            end = chunk_sizes[i + 1].item()
            proj[:, :, start:end, :] = self.p_in(x[:, :, start:end, :]) * self.g_in(
                x[:, :, start:end, :]
            ).sigmoid()

        return proj

    def _forward_torch(
        self,
        x: Tensor,
        mask: Tensor,
        triangle_mult_gate_nchunks: int = 1,
        _inplace_chunk_size: Optional[int] = None,
    ) -> Tensor:
        x = self.norm_in(x)
        x_in = x
        x = self._project_input(x, triangle_mult_gate_nchunks)
        x *= mask.unsqueeze(-1)

        a, b = torch.chunk(x, 2, dim=-1)

        a_std = a.std()
        b_std = b.std()
        if is_fp16_enabled() and a_std != 0.0 and b_std != 0.0:
            a = a / a_std
            b = b / b_std

        if is_fp16_enabled():
            with torch.amp.autocast("cuda", enabled=False):
                x = self._combine_projections(
                    a.float(), b.float(), _inplace_chunk_size=_inplace_chunk_size
                )
        else:
            x = self._combine_projections(a, b, _inplace_chunk_size=_inplace_chunk_size)

        x = self.norm_out(x)
        x = self.p_out(x)
        x *= self.g_out(x_in).sigmoid()
        return x

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        triangle_mult_gate_nchunks: int = 1,
        triangle_multiplicative: str = "torch",
        _inplace_chunk_size: Optional[int] = None,
        _input_inplace_safe: bool = False,
        _add_with_inplace: bool = False,
    ) -> Tensor:
        if (
            not self._debug_logged
            and os.getenv("BOLTZ_DEBUG_TRI_MULT", "0") in {"1", "true", "TRUE"}
        ):
            direction = "outgoing" if self.outgoing else "incoming"
            LOGGER.warning(
                "[triangular_mult] direction=%s backend=%s shape=%s dtype=%s "
                "gate_nchunks=%s cuequivariance_available=%s",
                direction,
                triangle_multiplicative,
                tuple(x.shape),
                x.dtype,
                triangle_mult_gate_nchunks,
                triangle_multiplicative_update is not None,
            )
            self._debug_logged = True

        if mask is None:
            mask = x.new_ones(x.shape[:-1])

        if triangle_multiplicative == "cuequivariance":
            x_in = x.clone() if (_input_inplace_safe and _add_with_inplace) else None
            x = kernel_triangular_mult(
                x,
                direction="outgoing" if self.outgoing else "incoming",
                mask=mask,
                norm_in_weight=self.norm_in.weight,
                norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight,
                g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight,
                norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight,
                g_out_weight=self.g_out.weight,
                eps=1e-5,
            )
            if x_in is not None:
                return x + x_in
            return x

        if triangle_multiplicative != "torch":
            raise InvalidTriangleMultiplicativeError(triangle_multiplicative)

        out = self._forward_torch(
            x,
            mask,
            triangle_mult_gate_nchunks=triangle_mult_gate_nchunks,
            _inplace_chunk_size=_inplace_chunk_size,
        )
        if _input_inplace_safe and _add_with_inplace:
            out = out + x
        return out


class TriangleMultiplicationOutgoing(_TriangleMultiplication):
    """TriangleMultiplicationOutgoing."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__(dim=dim, outgoing=True)


class TriangleMultiplicationIncoming(_TriangleMultiplication):
    """TriangleMultiplicationIncoming."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__(dim=dim, outgoing=False)
