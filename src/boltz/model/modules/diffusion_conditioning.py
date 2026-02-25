from __future__ import annotations

import torch
from torch import nn
from torch.nn import Module

from boltz.model.modules.encodersv2 import (
    AtomEncoder,
    PairwiseConditioning,
)


class DiffusionConditioning(Module):
    def __init__(
        self,
        token_s: int,
        token_z: int,
        atom_s: int,
        atom_z: int,
        atoms_per_window_queries: int = 32,
        atoms_per_window_keys: int = 128,
        atom_encoder_depth: int = 3,
        atom_encoder_heads: int = 4,
        token_transformer_depth: int = 24,
        token_transformer_heads: int = 8,
        atom_decoder_depth: int = 3,
        atom_decoder_heads: int = 4,
        atom_feature_dim: int = 128,
        conditioning_transition_layers: int = 2,
        use_no_atom_char: bool = False,
        use_atom_backbone_feat: bool = False,
        use_residue_feats_atoms: bool = False,
    ) -> None:
        super().__init__()

        self.pairwise_conditioner = PairwiseConditioning(
            token_z=token_z,
            dim_token_rel_pos_feats=token_z,
            num_transitions=conditioning_transition_layers,
        )

        self.atom_encoder = AtomEncoder(
            atom_s=atom_s,
            atom_z=atom_z,
            token_s=token_s,
            token_z=token_z,
            atoms_per_window_queries=atoms_per_window_queries,
            atoms_per_window_keys=atoms_per_window_keys,
            atom_feature_dim=atom_feature_dim,
            structure_prediction=True,
            use_no_atom_char=use_no_atom_char,
            use_atom_backbone_feat=use_atom_backbone_feat,
            use_residue_feats_atoms=use_residue_feats_atoms,
        )

        self.atom_enc_proj_z = nn.ModuleList()
        for _ in range(atom_encoder_depth):
            self.atom_enc_proj_z.append(
                nn.Sequential(
                    nn.LayerNorm(atom_z),
                    nn.Linear(atom_z, atom_encoder_heads, bias=False),
                )
            )

        self.atom_dec_proj_z = nn.ModuleList()
        for _ in range(atom_decoder_depth):
            self.atom_dec_proj_z.append(
                nn.Sequential(
                    nn.LayerNorm(atom_z),
                    nn.Linear(atom_z, atom_decoder_heads, bias=False),
                )
            )

        self.token_trans_proj_z = nn.ModuleList()
        for _ in range(token_transformer_depth):
            self.token_trans_proj_z.append(
                nn.Sequential(
                    nn.LayerNorm(token_z),
                    nn.Linear(token_z, token_transformer_heads, bias=False),
                )
            )

    def forward(
        self,
        s_trunk,  # Float['b n ts']
        z_trunk,  # Float['b n n tz']
        relative_position_encoding,  # Float['b n n tz']
        feats,
    ):
        z = self.pairwise_conditioner(
            z_trunk,
            relative_position_encoding,
        )

        q, c, p, to_keys = self.atom_encoder(
            feats=feats,
            s_trunk=s_trunk,  # Float['b n ts'],
            z=z,  # Float['b n n tz'],
        )

        atom_enc_chunks = [layer(p) for layer in self.atom_enc_proj_z]
        atom_enc_bias = torch.cat(atom_enc_chunks, dim=-1)
        del atom_enc_chunks

        atom_dec_chunks = [layer(p) for layer in self.atom_dec_proj_z]
        atom_dec_bias = torch.cat(atom_dec_chunks, dim=-1)
        del atom_dec_chunks

        first_token_bias = self.token_trans_proj_z[0](z)
        token_bias_dim = first_token_bias.shape[-1]
        token_trans_bias = z.new_empty(
            (*first_token_bias.shape[:-1], token_bias_dim * len(self.token_trans_proj_z))
        )
        token_trans_bias[..., :token_bias_dim] = first_token_bias
        del first_token_bias
        for i, layer in enumerate(self.token_trans_proj_z[1:], start=1):
            start = i * token_bias_dim
            end = start + token_bias_dim
            token_trans_bias[..., start:end] = layer(z)
        del z

        return q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias
