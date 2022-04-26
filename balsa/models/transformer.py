# Copyright 2022 The Balsa Authors.
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

import numpy as np
import torch
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self, plan_vocab_size, parent_pos_vocab_size, d_model,
                 num_heads, d_ff, num_layers, d_query_feat, plan_pad_idx,
                 parent_pos_pad_idx, use_pos_embs):
        super(Transformer, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout=0),
            num_layers,
        )
        self.d_model = d_model
        self.use_pos_embs = use_pos_embs

        self.embeds = nn.Embedding(plan_vocab_size,
                                   d_model,
                                   padding_idx=plan_pad_idx)
        self.plan_pad_idx = plan_pad_idx

        if use_pos_embs:
            # self.pos_embeds =nn.Embedding(100, #parent_pos_vocab_size,
            self.pos_embeds = nn.Embedding(parent_pos_vocab_size,
                                           d_model,
                                           padding_idx=parent_pos_pad_idx)

        self.mlp = nn.Sequential(*[
            nn.Linear(d_query_feat + d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        ])

        self.reset_weights()

    def reset_weights(self):

        def init(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
            if type(m) == nn.Embedding:
                nn.init.normal_(m.weight, std=0.02)

        self.apply(init)

    def forward(self, query_feats, src, parent_pos):
        """Forward pass.

        Args:
          query_feats: query features, Tensor of float, sized [batch size, num
            relations].
          src: Tensor of int64, sized [batch size, num sequence length].  This
            represents the input plans.
          parent_pos: Tensor of int64, sized [batch size, num sequence length].

        Returns:
          Predicted costs: Tensor of float, sized [batch size, 1].
        """
        # Ensures no info is gathered from PAD tokens.
        # src_key_padding_mask: (N, S)
        # [src/tgt/memory]_key_padding_mask should be a ByteTensor where True
        # values are positions that should be masked with float(‘-inf’) and
        # False values will be unchanged. This mask ensures that no information
        # will be taken from position i if it is masked, and has a separate mask
        # for each sequence in a batch.
        src_key_padding_mask = (src == self.plan_pad_idx)
        # src_key_padding_mask = None

        if self.use_pos_embs:
            src = self.embeds(src) + self.pos_embeds(parent_pos)
        else:
            src = self.embeds(src)

        # src: (S, N, E)
        src = src.transpose(0, 1)

        # [S, BS, E]
        output = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask)

        # [BS, S, E]
        output = output.transpose(0, 1)

        # [BS, E]
        root = output[:, 0, :]
        # [BS, NumRels + E]
        # print(query_feats.dtype, root.dtype)
        out = torch.cat((query_feats, root), dim=1)
        return self.mlp(out)


class TransformerV2(nn.Module):
    """V2. Process query features via Transformer as well."""

    def __init__(self,
                 plan_vocab_size,
                 parent_pos_vocab_size,
                 d_model,
                 num_heads,
                 d_ff,
                 num_layers,
                 d_query_feat,
                 plan_pad_idx,
                 parent_pos_pad_idx,
                 use_pos_embs,
                 dropout=0,
                 cross_entropy=False,
                 max_label_bins=None):
        super(TransformerV2, self).__init__()

        self.d_query_mlp = d_model
        self.d_model = d_model
        self.use_pos_embs = use_pos_embs

        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model + self.d_query_mlp,
                                    num_heads,
                                    d_ff,
                                    dropout=dropout),
            num_layers,
        )

        self.query_mlp = nn.Sequential(*[
            nn.Linear(d_query_feat, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, self.d_query_mlp),
        ])

        output_dim = max_label_bins if cross_entropy else 1
        self.out_mlp = nn.Sequential(*[
            nn.Linear(d_model + self.d_query_mlp, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        ])

        self.embeds = nn.Embedding(plan_vocab_size,
                                   d_model,
                                   padding_idx=plan_pad_idx)
        self.plan_pad_idx = plan_pad_idx
        if use_pos_embs:
            self.pos_embeds = nn.Embedding(
                parent_pos_vocab_size,
                d_model + self.d_query_mlp,
                # d_model,
                padding_idx=parent_pos_pad_idx)

        self.reset_weights()

    def reset_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                if 'embeds.weight' in name:
                    embedding_dim = p.shape[-1]
                    nn.init.normal_(p, mean=0, std=embedding_dim**-0.5)
                    # Last row is pad_idx.
                    nn.init.zeros_(p[-1])
                else:
                    # Weights.
                    nn.init.normal_(p, std=0.02)
                # nn.init.normal_(p, std=0.02)
            elif 'bias' in name:
                # Layer norm bias; linear bias, etc.
                nn.init.zeros_(p)
            else:
                # Layer norm weight.
                #assert 'norm' in name and 'weight' in name, name
                nn.init.ones_(p)

    def forward(self, query_feats, src, parent_pos):
        """Forward pass.

        Args:
          query_feats: query features, Tensor of float, sized [batch size, num
            relations].
          src: Tensor of int64, sized [batch size, num sequence length].  This
            represents the input plans.
          parent_pos: Tensor of int64, sized [batch size, num sequence length].

        Returns:
          Predicted costs: Tensor of float, sized [batch size, 1].
        """

        # [batch size, 1, query_mlp dim]
        query_embs = self.query_mlp(query_feats.unsqueeze(1))
        # [batch size, seq length, query_mlp dim]
        query_embs = query_embs.repeat(1, src.shape[1], 1)

        # [batch size, seq length, d_model]
        plan_embs = self.embeds(src)

        # [batch size, seq length, d_model + query_embs.shape[-1]]
        # TODO: does it make sense to add pos embed to query features?
        if self.use_pos_embs:
            plan_and_query_embs = torch.cat((plan_embs, query_embs), -1)
            plan_and_query_embs = plan_and_query_embs + self.pos_embeds(
                parent_pos)
            # plan_and_query_embs = torch.cat(
            #     (plan_embs + self.pos_embeds(parent_pos), query_embs), -1)
        else:
            plan_and_query_embs = torch.cat((plan_embs, query_embs), -1)

        # Send the concat'd embs through Transformer.

        # src: (S, N, E)
        plan_and_query_embs = plan_and_query_embs.transpose(0, 1)

        # Ensures no info is gathered from PAD tokens.
        # src_key_padding_mask: (N, S)
        # [src/tgt/memory]_key_padding_mask should be a ByteTensor where True
        # values are positions that should be masked with float(‘-inf’) and
        # False values will be unchanged. This mask ensures that no information
        # will be taken from position i if it is masked, and has a separate mask
        # for each sequence in a batch.
        src_key_padding_mask = (src == self.plan_pad_idx)

        # [S, BS, E]
        output = self.transformer_encoder(
            plan_and_query_embs, src_key_padding_mask=src_key_padding_mask)

        # [BS, S, E]
        output = output.transpose(0, 1)
        # root = output.max(dim=1)[0]
        root = output[:, 0, :]  # FIXME: pool?
        return self.out_mlp(root)


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb
