import torch
from torch import nn
from torch.nn import functional as F

#from .models import register_backbone
from blocks import (get_sinusoid_encoding, TransformerBlock, MaskedConv1D,
                     ConvBlock, LayerNorm)



class STAdapter(nn.Module):
    def __init__(self, n_embd, bottleneck_dim=512, kernel_size=3):
        super(STAdapter, self).__init__()
        if bottleneck_dim <= 0:
            raise ValueError("ST-Adapter bottleneck_dim must be positive.")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("ST-Adapter kernel_size must be a positive odd integer.")
        self.down = nn.Conv1d(n_embd, bottleneck_dim, 1)
        self.temporal = nn.Conv1d(
            bottleneck_dim, bottleneck_dim, kernel_size,
            padding=kernel_size // 2, groups=bottleneck_dim
        )
        self.norm = LayerNorm(bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Conv1d(bottleneck_dim, n_embd, 1)

    def forward(self, x, mask):
        residual = x
        out = self.down(x)
        out = self.temporal(out)
        out = self.act(self.norm(out))
        out = self.up(out)
        return (residual + out) * mask.to(out.dtype)

#@register_backbone("convTransformer")
class ConvTransformerBackbone(nn.Module):
    """
        A backbone that combines convolutions with transformers
    """
    def __init__(
        self,
        topk,
        n_in,                  # input feature dimension
        n_embd,                # embedding dimension (after convolution)
        n_head,                # number of head for self-attention in transformers
        n_embd_ks,             # conv kernel size of the embedding network
        max_len,               # max sequence length
        arch = (2, 2, 5),      # (#convs, #stem transformers, #branch transformers)
        mha_win_size = [-1]*6, # size of local window for mha
        scale_factor = 2,      # dowsampling rate for the branch
        with_ln = False,       # if to attach layernorm after conv
        attn_pdrop = 0.0,      # dropout rate for the attention map
        proj_pdrop = 0.0,      # dropout rate for the projection / MLP
        path_pdrop = 0.0,      # droput rate for drop path
        use_abs_pe = False,    # use absolute position embedding
        use_rel_pe = False,    # use relative position embedding
        args = None
    ):
        super().__init__()
        if isinstance(mha_win_size, int):
            self.mha_win_size = [mha_win_size]*(1 + arch[-1])
        else:
            assert len(mha_win_size) == (1 + arch[-1])
            self.mha_win_size = mha_win_size
        assert len(arch) == 3
        #assert len(mha_win_size) == (1 + arch[2])
        self.topk = topk
        self.original_n_in = n_in
        self.n_in = n_in
        self.arch = arch
        #self.mha_win_size = mha_win_size
        self.max_len = max_len
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.use_abs_pe = use_abs_pe
        self.use_rel_pe = use_rel_pe
        self.args = args
        self.ins_prompt = args.ins_prompt
        self.target_prompt = args.target_prompt
        self.verb_prompt = args.verb_prompt
        self.task_prompt = args.task_prompt
        self.vpt_prompt = getattr(args, 'vpt_prompt', False)
        self.vpt_prompt_len = int(getattr(args, 'vpt_prompt_len', 4))
        self.vpt_prompt_layers = [int(layer) for layer in getattr(args, 'vpt_prompt_layers', [4])]
        self.st_adapter = getattr(args, 'st_adapter', False)
        self.st_adapter_dim = int(getattr(args, 'st_adapter_dim', n_embd))
        self.st_adapter_kernel_size = int(getattr(args, 'st_adapter_kernel_size', 3))
        if self.st_adapter and self.st_adapter_dim <= 0:
            raise ValueError("--st_adapter_dim must be positive when --st_adapter is enabled.")
        if self.vpt_prompt:
            if self.vpt_prompt_len <= 0:
                raise ValueError('--vpt_prompt_len must be positive when --vpt_prompt is enabled.')
            invalid_layers = [layer for layer in self.vpt_prompt_layers if layer < 0 or layer >= arch[1]]
            if invalid_layers:
                raise ValueError('--vpt_prompt_layers must refer to stem layers in [0, {}], got {}'.format(arch[1] - 1, invalid_layers))
        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            if self.ins_prompt != -1:
                self.proj_prompt = nn.ModuleList([
                    MaskedConv1D(c0, c1, 1) for c0, c1 in zip(self.topk*3 * n_in, n_embd)
                ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None
            self.proj_prompt = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())
        
        if self.ins_prompt != -1:
            self.embd_prompt = nn.ModuleList()
            self.embd_norm_prompt = nn.ModuleList()
            for idx in range(arch[0]):
                n_in = n_embd if idx > 0 else self.topk*3*self.original_n_in # 13*768 or 1024
                self.embd_prompt.append(MaskedConv1D(n_in, n_embd, n_embd_ks, stride=1, padding=n_embd_ks//2, bias=(not with_ln)))
                self.embd_norm_prompt.append(LayerNorm(n_embd))

        # if self.target_prompt != -1:
        #     self.embd_target = nn.ModuleList()
        #     self.embd_norm_target = nn.ModuleList()
        #     for idx in range(arch[0]):
        #         n_in = n_embd if idx > 0 else 2*768
        #         self.embd_target.append(MaskedConv1D(n_in, n_embd, n_embd_ks, stride=1, padding=n_embd_ks//2, bias=(not with_ln)))
        #         self.embd_norm_target.append(LayerNorm(n_embd))
        
        # if self.verb_prompt != -1:
        #     self.embd_verb = nn.ModuleList()
        #     self.embd_norm_verb = nn.ModuleList()
        #     for idx in range(arch[0]):
        #         n_in = n_embd if idx > 0 else 2*768
        #         self.embd_verb.append(MaskedConv1D(n_in, n_embd, n_embd_ks, stride=1, padding=n_embd_ks//2, bias=(not with_ln)))
        #         self.embd_norm_verb.append(LayerNorm(n_embd))

        self.vpt_prompts = nn.ParameterDict()
        if self.vpt_prompt:
            for layer in self.vpt_prompt_layers:
                prompt = nn.Parameter(torch.empty(1, n_embd, self.vpt_prompt_len))
                nn.init.trunc_normal_(prompt, mean=0.0, std=0.02)
                self.vpt_prompts[str(layer)] = prompt

        # position embedding (1, C, T), rescaled by 1/sqrt(n_embd)
        if self.use_abs_pe:
            pos_embd = get_sinusoid_encoding(self.max_len, n_embd) / (n_embd**0.5)
            self.register_buffer("pos_embd", pos_embd, persistent=False)

        # stem network using (vanilla) transformer
        self.stem = nn.ModuleList()
        self.stem_st_adapters = nn.ModuleList()
        for idx in range(arch[1]):
            if idx == self.ins_prompt:
                ins_prompt = True
            else:
                ins_prompt = False
            
            if idx == self.target_prompt:
                target_prompt = True
            else:
                target_prompt = False

            if idx == self.verb_prompt:
                verb_prompt = True
            else:
                verb_prompt = False
            
            # if idx == 4:
            #     verb_prompt = True
            # else:
            #     verb_prompt = False
            if idx == self.task_prompt:
                task_prompt = True
            else:
                task_prompt = False

            self.stem.append(
                TransformerBlock(
                    args,
                    max_len,
                    n_embd, n_head,
                    n_ds_strides=(1, 1),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[0],
                    use_rel_pe=self.use_rel_pe,
                    ins_prompt=ins_prompt,
                    target_prompt=target_prompt,
                    verb_prompt=verb_prompt,
                    task_prompt=task_prompt,
                    residual=False
                )
            )
            if self.st_adapter:
                self.stem_st_adapters.append(STAdapter(n_embd, self.st_adapter_dim, self.st_adapter_kernel_size))

        # main branch using transformer with pooling
        self.branch = nn.ModuleList()
        self.branch_st_adapters = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(
                TransformerBlock(
                    args,
                    max_len,
                    n_embd, n_head,
                    n_ds_strides=(self.scale_factor, self.scale_factor),
                    attn_pdrop=attn_pdrop,
                    proj_pdrop=proj_pdrop,
                    path_pdrop=path_pdrop,
                    mha_win_size=self.mha_win_size[1 + idx],
                    use_rel_pe=self.use_rel_pe,
                    residual=False
                )
            )
            if self.st_adapter:
                self.branch_st_adapters.append(STAdapter(n_embd, self.st_adapter_dim, self.st_adapter_kernel_size))

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear/nn.Conv1d bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)
        # if isinstance(module, (nn.Linear, nn.Conv1d)):
        #     torch.nn.init.xavier_normal_(module.weight)
        #     if module.bias is not None:
        #         torch.nn.init.constant_(module.bias, 0.)
        # if isinstance(module, nn.LayerNorm):
        #     torch.nn.init.constant_(module.bias, 0.)
        #     torch.nn.init.constant_(module.weight, 1.0)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        if self.ins_prompt != -1:
            orig_x = x[0]
            x_prompt = x[1]
        else:
            orig_x = x
        B, C, T = orig_x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            orig_x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, orig_x.split(self.n_in, dim=1))
                ], dim=1
            )
            x_prompt = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj_prompt, x_prompt.split(self.topk*3 * self.original_n_in, dim=1))
                ], dim=1
            )

        # embedding network
        for idx in range(len(self.embd)):
            orig_x, mask = self.embd[idx](orig_x, mask)
            if self.ins_prompt != -1:
                x_prompt, mask = self.embd_prompt[idx](x_prompt, mask)
            # if self.target_prompt != -1:
            #     x_prompt, mask = self.embd_target[idx](x_prompt, mask)
            # if self.verb_prompt != -1:
            #     x_prompt, mask = self.embd_verb[idx](x_prompt, mask)
            orig_x = self.relu(self.embd_norm[idx](orig_x))
            if self.ins_prompt != -1:
                x_prompt = self.relu(self.embd_norm_prompt[idx](x_prompt))
            # if self.target_prompt != -1:
            #     x_prompt = self.relu(self.embd_norm_target[idx](x_prompt))
            # if self.verb_prompt != -1:
            #     x_prompt = self.relu(self.embd_norm_verb[idx](x_prompt))

        # # training: using fixed length position embeddings
        # if self.use_abs_pe and self.training:
        #     assert T <= self.max_len, "Reached max length."
        #     pe = self.pos_embd
        #     # add pe to x
        #     x = x + pe[:, :, :T] * mask.to(x.dtype)

        # # inference: re-interpolate position embeddings for over-length sequences
        # if self.use_abs_pe and (not self.training):
        #     if T >= self.max_len:
        #         pe = F.interpolate(
        #             self.pos_embd, T, mode='linear', align_corners=False)
        #     else:
        #         pe = self.pos_embd
        #     # add pe to x
        #     x = x + pe[:, :, :T] * mask.to(x.dtype)

        # stem transformer
        for idx in range(len(self.stem)):
            if self.vpt_prompt and str(idx) in self.vpt_prompts:
                prompt = self.vpt_prompts[str(idx)].expand(orig_x.size(0), -1, -1)
                prompt_mask = torch.ones(orig_x.size(0), 1, self.vpt_prompt_len, dtype=mask.dtype, device=mask.device)
                orig_x = torch.cat([prompt, orig_x], dim=2)
                mask = torch.cat([prompt_mask, mask], dim=2)

                pad_len = 0
                if self.mha_win_size[0] > 1:
                    align = self.mha_win_size[0] - 1
                    pad_len = (align - (orig_x.size(2) % align)) % align
                if pad_len > 0:
                    pad_tokens = orig_x.new_zeros(orig_x.size(0), orig_x.size(1), pad_len)
                    pad_mask = torch.zeros(orig_x.size(0), 1, pad_len, dtype=mask.dtype, device=mask.device)
                    orig_x = torch.cat([orig_x, pad_tokens], dim=2)
                    mask = torch.cat([mask, pad_mask], dim=2)

                orig_x, mask = self.stem[idx](orig_x, mask)
                end_idx = orig_x.size(2) - pad_len if pad_len > 0 else orig_x.size(2)
                orig_x = orig_x[:, :, self.vpt_prompt_len:end_idx]
                mask = mask[:, :, self.vpt_prompt_len:end_idx]
            elif idx == self.ins_prompt:
                orig_x, mask = self.stem[idx]([orig_x, x_prompt], mask)
            elif idx == self.target_prompt:
                orig_x, mask = self.stem[idx]([orig_x, x_prompt], mask)
            elif idx == self.verb_prompt:
                orig_x, mask = self.stem[idx]([orig_x, x_prompt], mask)
            elif idx == self.task_prompt:
                orig_x, mask, task_prompts_cos = self.stem[idx](orig_x, mask)
            else:
                orig_x, mask = self.stem[idx](orig_x, mask)
            if self.st_adapter:
                orig_x = self.stem_st_adapters[idx](orig_x, mask)

        # prep for outputs
        out_feats = (orig_x, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            orig_x, mask = self.branch[idx](orig_x, mask)
            if self.st_adapter:
                orig_x = self.branch_st_adapters[idx](orig_x, mask)
            out_feats += (orig_x, )
            out_masks += (mask, )
        if self.task_prompt!= -1:
            return out_feats, out_masks, task_prompts_cos
        else:
            return out_feats, out_masks


#@register_backbone("conv")
class ConvBackbone(nn.Module):
    """
        A backbone that with only conv
    """
    def __init__(
        self,
        n_in,               # input feature dimension
        n_embd,             # embedding dimension (after convolution)
        n_embd_ks,          # conv kernel size of the embedding network
        arch = (2, 2, 5),   # (#convs, #stem convs, #branch convs)
        scale_factor = 2,   # dowsampling rate for the branch
        with_ln=False,      # if to use layernorm
    ):
        super().__init__()
        assert len(arch) == 3
        self.n_in = n_in
        self.arch = arch
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        # feature projection
        self.n_in = n_in
        if isinstance(n_in, (list, tuple)):
            assert isinstance(n_embd, (list, tuple)) and len(n_in) == len(n_embd)
            self.proj = nn.ModuleList([
                MaskedConv1D(c0, c1, 1) for c0, c1 in zip(n_in, n_embd)
            ])
            n_in = n_embd = sum(n_embd)
        else:
            self.proj = None

        # embedding network using convs
        self.embd = nn.ModuleList()
        self.embd_norm = nn.ModuleList()
        for idx in range(arch[0]):
            n_in = n_embd if idx > 0 else n_in
            self.embd.append(
                MaskedConv1D(
                    n_in, n_embd, n_embd_ks,
                    stride=1, padding=n_embd_ks//2, bias=(not with_ln)
                )
            )
            if with_ln:
                self.embd_norm.append(LayerNorm(n_embd))
            else:
                self.embd_norm.append(nn.Identity())

        # stem network using convs
        self.stem = nn.ModuleList()
        for idx in range(arch[1]):
            self.stem.append(ConvBlock(n_embd, 3, 1))

        # main branch using convs with pooling
        self.branch = nn.ModuleList()
        for idx in range(arch[2]):
            self.branch.append(ConvBlock(n_embd, 3, self.scale_factor))

        # init weights
        self.apply(self.__init_weights__)

    def __init_weights__(self, module):
        # set nn.Linear bias term to 0
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.)

    def forward(self, x, mask):
        # x: batch size, feature channel, sequence length,
        # mask: batch size, 1, sequence length (bool)
        B, C, T = x.size()

        # feature projection
        if isinstance(self.n_in, (list, tuple)):
            x = torch.cat(
                [proj(s, mask)[0] \
                    for proj, s in zip(self.proj, x.split(self.n_in, dim=1))
                ], dim=1
            )

        # embedding network
        for idx in range(len(self.embd)):
            x, mask = self.embd[idx](x, mask)
            x = self.relu(self.embd_norm[idx](x))

        # stem conv
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # prep for outputs
        out_feats = (x, )
        out_masks = (mask, )

        # main branch with downsampling
        for idx in range(len(self.branch)):
            x, mask = self.branch[idx](x, mask)
            out_feats += (x, )
            out_masks += (mask, )

        return out_feats, out_masks