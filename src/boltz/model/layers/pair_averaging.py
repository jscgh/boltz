import torch
from torch import Tensor, nn

import boltz.model.layers.initialize as init


class PairWeightedAveraging(nn.Module):
    """Pair weighted averaging layer."""

    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_h: int,
        num_heads: int,
        inf: float = 1e6,
    ) -> None:
        """Initialize the pair weighted averaging layer.

        Parameters
        ----------
        c_m: int
            The dimension of the input sequence.
        c_z: int
            The dimension of the input pairwise tensor.
        c_h: int
            The dimension of the hidden.
        num_heads: int
            The number of heads.
        inf: float
            The value to use for masking, default 1e6.

        """
        super().__init__()
        self.c_m = c_m
        self.c_z = c_z
        self.c_h = c_h
        self.num_heads = num_heads
        self.inf = inf

        self.norm_m = nn.LayerNorm(c_m)
        self.norm_z = nn.LayerNorm(c_z)

        self.proj_m = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_g = nn.Linear(c_m, c_h * num_heads, bias=False)
        self.proj_z = nn.Linear(c_z, num_heads, bias=False)
        self.proj_o = nn.Linear(c_h * num_heads, c_m, bias=False)
        init.final_init_(self.proj_o.weight)

    def forward(
        self,
        m: Tensor,
        z: Tensor,
        mask: Tensor,
        chunk_heads: bool = False,
        chunk_size_msa: int = None,
        chunk_size_pair: int = None,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        m : torch.Tensor
            The input sequence tensor (B, S, N, D)
        z : torch.Tensor
            The input pairwise tensor (B, N, N, D)
        mask : torch.Tensor
            The pairwise mask tensor (B, N, N)

        Returns
        -------
        torch.Tensor
            The output sequence tensor (B, S, N, D)

        """
        # Compute layer norms
        m = self.norm_m(m)
        z = self.norm_z(z)

        if chunk_heads and not self.training:
            # Compute heads sequentially
            msa_chunk_size = (
                m.shape[1]
                if chunk_size_msa is None or chunk_size_msa <= 0
                else chunk_size_msa
            )
            pair_chunk_size = (
                z.shape[1]
                if chunk_size_pair is None or chunk_size_pair <= 0
                else chunk_size_pair
            )
            o_out = torch.zeros_like(m)

            for head_idx in range(self.num_heads):
                sliced_weight_proj_m = self.proj_m.weight[
                    head_idx * self.c_h : (head_idx + 1) * self.c_h, :
                ]

                sliced_weight_proj_z = self.proj_z.weight[head_idx : (head_idx + 1), :]

                sliced_weight_proj_g = self.proj_g.weight[
                    head_idx * self.c_h : (head_idx + 1) * self.c_h, :
                ]
                sliced_weight_proj_o = self.proj_o.weight[
                    :, head_idx * self.c_h : (head_idx + 1) * self.c_h
                ]

                # Compute output in MSA chunks to reduce peak memory
                for msa_start in range(0, m.shape[1], msa_chunk_size):
                    msa_end = min(msa_start + msa_chunk_size, m.shape[1])
                    m_chunk = m[:, msa_start:msa_end]

                    v: Tensor = m_chunk @ sliced_weight_proj_m.T
                    v = v.reshape(*v.shape[:3], 1, self.c_h)
                    v = v.permute(0, 3, 1, 2, 4)

                    g: Tensor = m_chunk @ sliced_weight_proj_g.T
                    g = g.sigmoid()

                    for pair_start in range(0, z.shape[1], pair_chunk_size):
                        pair_end = min(pair_start + pair_chunk_size, z.shape[1])
                        z_chunk = z[:, pair_start:pair_end]
                        mask_chunk = mask[:, pair_start:pair_end]

                        b_chunk: Tensor = z_chunk @ sliced_weight_proj_z.T
                        b_chunk = b_chunk.permute(0, 3, 1, 2)
                        b_chunk += (1 - mask_chunk[:, None]) * -self.inf
                        w_chunk = torch.softmax(b_chunk, dim=-1)

                        o_chunk = torch.einsum("bhij,bhsjd->bhsid", w_chunk, v)
                        o_chunk = o_chunk.permute(0, 2, 3, 1, 4)
                        o_chunk = o_chunk.reshape(*o_chunk.shape[:3], self.c_h)
                        o_chunk *= g[:, :, pair_start:pair_end]

                        o_out[:, msa_start:msa_end, pair_start:pair_end] += (
                            o_chunk @ sliced_weight_proj_o.T
                        )

                        del z_chunk, mask_chunk, b_chunk, w_chunk, o_chunk

                    del m_chunk, v, g

                del sliced_weight_proj_o
                del sliced_weight_proj_g
                del sliced_weight_proj_m
                del sliced_weight_proj_z
            return o_out
        else:
            # Project input tensors
            v: Tensor = self.proj_m(m)
            v = v.reshape(*v.shape[:3], self.num_heads, self.c_h)
            v = v.permute(0, 3, 1, 2, 4)

            # Compute weights
            b: Tensor = self.proj_z(z)
            b = b.permute(0, 3, 1, 2)
            b = b + (1 - mask[:, None]) * -self.inf
            w = torch.softmax(b, dim=-1)

            # Compute gating
            g: Tensor = self.proj_g(m)
            g = g.sigmoid()

            # Compute output
            o = torch.einsum("bhij,bhsjd->bhsid", w, v)
            o = o.permute(0, 2, 3, 1, 4)
            o = o.reshape(*o.shape[:3], self.num_heads * self.c_h)
            o = self.proj_o(g * o)
            return o
