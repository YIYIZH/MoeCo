#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from torch import nn

import torch.nn.functional as F
import copy
import random
import math
import numpy as np
import network_trans
from blocks import MaskedConv1D, Scale, LayerNorm, TransformerBlock
import clip

class FPNIdentity(nn.Module):
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = #levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True,     # if to apply layer norm at the end
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # check feat dims
            assert self.in_channels[i] == self.out_channel
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        # print(len(inputs))
        # print(len(self.in_channels))
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # apply norms, fpn_masks will remain the same with 1x1 convs
        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(len(self.fpn_norms)):
            x = self.fpn_norms[i](inputs[i + self.start_level])
            fpn_feats += (x, )
            new_fpn_masks += (fpn_masks[i + self.start_level], )

        return fpn_feats, new_fpn_masks
    
class TaskFeatureBranch(nn.Module):
    def __init__(self, args, num_f_maps):
        super(TaskFeatureBranch, self).__init__()
        self.block = TransformerBlock(
            args,
            max_len=4032,
            n_embd=num_f_maps,
            n_head=args.head_num,
            n_ds_strides=(1, 1),
            attn_pdrop=0.0,
            proj_pdrop=0.0,
            path_pdrop=0.1,
            mha_win_size=-1,
            use_rel_pe=False,
            residual=False
        )

    def forward(self, x, mask):
        return self.block(x, mask)


class RandomMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2

        self.height, self.width = input_size

        self.num_patches = self.height * self.width
        self.num_mask = int(mask_ratio * self.num_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.num_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.num_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask  # [196]


class VideoNas(nn.Module):

    def __init__(self, args, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, num_i=6, num_v=10,
                 num_t=15):
        super(VideoNas, self).__init__()
        # self.PG = Prediction_Generation(args, num_layers_PG, num_f_maps, dim, num_classes)
        self.PG = BaseCausalTCN(num_layers_PG, num_f_maps, dim, num_classes)

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.conv_out_i = nn.Conv1d(num_f_maps, num_i, 1)
        self.conv_out_v = nn.Conv1d(num_f_maps, num_v, 1)
        self.conv_out_t = nn.Conv1d(num_f_maps, num_t, 1)
        self.args = args
        self.Rs = nn.ModuleList(
            [copy.deepcopy(Refinement(args, num_layers_R, num_f_maps, num_classes, num_classes, self.conv_out)) for s in
             range(num_R)])
        self.use_fpn = args.fpn
        self.use_output = args.output
        self.use_feature = args.feature
        self.use_trans = args.trans
        # self.prototpye=[]
        if args.fpn:
            self.fpn = FPN(num_f_maps)

    def forward(self, x, ismask):
        out_list = []
        out_list_i = []
        out_list_v = []
        out_list_t = []
        f_list = []
        x = x.permute(0, 2, 1)
        if self.args.mask and ismask:
            num_patches = x.flatten().shape[0]
            num_mask = int(num_patches * 0.75)
            mask = torch.concat((torch.zeros(num_patches - num_mask), torch.ones(num_mask)))
            mask = mask[torch.randperm(mask.nelement())]
            mask = mask.view(x.shape)

            f, out1 = self.PG(x, mask)
        else:
            f, out1 = self.PG(x)

        f_list.append(f)
        if not self.use_fpn:
            out_list.append(out1)

        for R in self.Rs:
            f, out1 = R(f)
            f_list.append(f)
        if self.use_fpn:
            f_list = self.fpn(f_list)
            for f in f_list:
                out_list.append(self.conv_out(f))
                out_list_i.append(self.conv_out_i(f))
                out_list_v.append(self.conv_out_v(f))
                out_list_t.append(self.conv_out_t(f))
        return out_list, out_list_i, out_list_v, out_list_t, f_list, f_list

class ClipAlign(nn.Module):
    def __init__(self, feature_dim):
        super(ClipAlign, self).__init__()
        self.feature_dim = feature_dim
        self.fc1 = nn.Linear(feature_dim, feature_dim // 2)  # Reduce dimensionality
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(feature_dim // 2, feature_dim)  # Project back to original dimensionality
        self.layernorm = nn.LayerNorm(feature_dim)  # Normalize output

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.layernorm(x)  # Optional, helps stabilize training
        return x

class FeatureFusionModule(nn.Module):
    def __init__(self, f1_dim, f2_dim, alpha=0.5, hidden_dim=512):
        super(FeatureFusionModule, self).__init__()
        # Feature enhancement layers
        self.fc_f1 = nn.Linear(f1_dim, hidden_dim)
        self.fc_f2 = nn.Linear(f2_dim, hidden_dim)
        
        # Learnable weight for fusion
        self.alpha = nn.Parameter(torch.tensor(alpha))  # Initialize alpha to favor f1
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, f1_dim)
        
    def forward(self, f1, f2):
        # Feature enhancement
        f1_enhanced = F.relu(self.fc_f1(f1))
        f2_enhanced = F.relu(self.fc_f2(f2))
        
        # Weighted fusion
        alpha = torch.sigmoid(self.alpha)  # Ensure alpha is in [0, 1]
        fused_feature = alpha * f1_enhanced + (1 - alpha) * f2_enhanced
        
        # Output layer
        output = self.fc_out(fused_feature)
        
        # Residual connection to reinforce f1's dominance
        output = output + f1
        
        return output, f2_enhanced
    
class CrossAttentionFusionModule(nn.Module):
    def __init__(self, f1_dim, f2_dim, alpha=0.5, hidden_dim=512, num_heads=4):
        super(CrossAttentionFusionModule, self).__init__()
        # Feature enhancement layers (using 1D Convolution)
        self.conv_f1 = nn.Conv1d(in_channels=f1_dim, out_channels=hidden_dim, kernel_size=1)
        self.conv_f2 = nn.Conv1d(in_channels=f2_dim, out_channels=hidden_dim, kernel_size=1)
        
        # Layer Normalization for feature enhancement
        self.norm_f1 = nn.LayerNorm(hidden_dim)
        self.norm_f2 = nn.LayerNorm(hidden_dim)
        
        # Multi-head Cross-Attention
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)  # Query from f1
        self.key = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)    # Key from f2
        self.value = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1)  # Value from f2
        
        # Layer Normalization for attention
        self.norm_attention = nn.LayerNorm(hidden_dim)
        
        # Output layer (using 1D Convolution)
        self.conv_out = nn.Conv1d(in_channels=hidden_dim, out_channels=f1_dim, kernel_size=1)
        
        # Layer Normalization for output
        self.norm_out = nn.LayerNorm(f1_dim)

        self.dropout_out = nn.Dropout(0.7)
        self.alpha = nn.Parameter(torch.tensor(alpha))
        
    def forward(self, f1, f2):
        # f1: [batch_size, seq_len, f1_dim]
        # f2: [batch_size, seq_len, f2_dim]
        batch_size, seq_len, _ = f1.shape
        
        # Transpose for 1D convolution: [batch_size, feature_dim, seq_len]
        f1 = f1.transpose(1, 2)  # [batch_size, f1_dim, seq_len]
        f2 = f2.transpose(1, 2)  # [batch_size, f2_dim, seq_len]
        
        # Feature enhancement (1D Convolution)
        f1_enhanced = self.conv_f1(f1)  # [batch_size, hidden_dim, seq_len]
        f2_enhanced = self.conv_f2(f2)  # [batch_size, hidden_dim, seq_len]
        
        # Transpose back for LayerNorm: [batch_size, seq_len, hidden_dim]
        f1_enhanced = f1_enhanced.transpose(1, 2)
        f2_enhanced = f2_enhanced.transpose(1, 2)
        
        # Layer Normalization and ReLU
        f1_enhanced = F.relu(self.norm_f1(f1_enhanced))  # [batch_size, seq_len, hidden_dim]
        f2_enhanced = F.relu(self.norm_f2(f2_enhanced))  # [batch_size, seq_len, hidden_dim]
        
        # Transpose for attention: [batch_size, hidden_dim, seq_len]
        f1_enhanced = f1_enhanced.transpose(1, 2)
        f2_enhanced = f2_enhanced.transpose(1, 2)
        
        # Multi-head Cross-Attention
        query = self.query(f1_enhanced)  # Query from f1, Shape: [batch_size, hidden_dim, seq_len]
        key = self.key(f2_enhanced)      # Key from f2, Shape: [batch_size, hidden_dim, seq_len]
        value = self.value(f2_enhanced)  # Value from f2, Shape: [batch_size, hidden_dim, seq_len]
        
        # Reshape for multi-head attention: [batch_size, num_heads, seq_len, head_dim]
        query = query.view(batch_size, self.num_heads, self.head_dim, seq_len).transpose(-2, -1)
        key = key.view(batch_size, self.num_heads, self.head_dim, seq_len).transpose(-2, -1)
        value = value.view(batch_size, self.num_heads, self.head_dim, seq_len).transpose(-2, -1)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, seq_len, seq_len]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]
        
        # Weighted sum of values
        attended_features = torch.matmul(attention_weights, value)  # [batch_size, num_heads, seq_len, head_dim]
        attended_features = attended_features.transpose(-2, -1).contiguous().view(batch_size, self.head_dim * self.num_heads, seq_len)  # [batch_size, hidden_dim, seq_len]
        
        # Transpose for LayerNorm: [batch_size, seq_len, hidden_dim]
        attended_features = attended_features.transpose(1, 2)
        
        # Layer Normalization and residual connection
        attended_features = self.norm_attention(attended_features)  # [batch_size, seq_len, hidden_dim]
        attended_features = attended_features.transpose(1, 2)  # [batch_size, hidden_dim, seq_len]
        
        alpha = torch.sigmoid(self.alpha) 
        # Residual connection: Add attended features to f1_enhanced
        fused_feature = alpha * f1_enhanced + (1-alpha) * attended_features  # [batch_size, hidden_dim, seq_len]
        
        # Output layer (1D Convolution)
        output = self.conv_out(fused_feature)  # [batch_size, f1_dim, seq_len]
        
        # Transpose back to original shape: [batch_size, seq_len, f1_dim]
        output = output.transpose(1, 2)
        
        # Layer Normalization and final residual connection
        output = self.dropout_out(self.norm_out(output))  # [batch_size, seq_len, f1_dim]
        f1 = f1.transpose(1, 2)  # [batch_size, seq_len, f1_dim]

        
        output1 = output + f1
        
        return output1, output
    
class SelfAttentionFusionModule(nn.Module):
    def __init__(self, f1_dim, f2_dim, alpha=0.5, hidden_dim=512):
        super(SelfAttentionFusionModule, self).__init__()
        # Feature enhancement layers
        self.fc_f1 = nn.Linear(f1_dim, hidden_dim)
        self.fc_f2 = nn.Linear(f2_dim, hidden_dim)
        
        # Self-Attention layers
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, f1_dim)
        self.hidden_dim = hidden_dim

        self.alpha = nn.Parameter(torch.tensor(alpha))
        
    def forward(self, f1, f2):
        # Feature enhancement
        f1_enhanced = F.relu(self.fc_f1(f1))
        f2_enhanced = F.relu(self.fc_f2(f2))
        
        # Concatenate features for Self-Attention
        features = torch.cat([f1_enhanced.unsqueeze(1), f2_enhanced.unsqueeze(1)], dim=1)  # Shape: (batch_size, 2, hidden_dim)
        
        # Self-Attention
        query = self.query(features)  # Shape: (batch_size, 2, hidden_dim)
        key = self.key(features)      # Shape: (batch_size, 2, hidden_dim)
        value = self.value(features)  # Shape: (batch_size, 2, hidden_dim)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.hidden_dim ** 0.5)  # Shape: (batch_size, 2, 2)
        attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: (batch_size, 2, 2)
        
        # Weighted sum of values
        attended_features = torch.matmul(attention_weights, value)  # Shape: (batch_size, 2, hidden_dim)
        
        # Separate attended features
        attended_f1 = attended_features[:, 0, :]  # Shape: (batch_size, hidden_dim)
        attended_f2 = attended_features[:, 1, :]  # Shape: (batch_size, hidden_dim)
        
        alpha = torch.sigmoid(self.alpha) 
        # Weighted fusion
        fused_feature = alpha * attended_f1 + (1-alpha) * attended_f2  # Sum of attended features
        
        # Output layer
        output = self.fc_out(fused_feature)
        
        # Residual connection to reinforce f1's dominance
        output = output + f1
        
        return output, f2_enhanced

class PromptDropout(nn.Module):
    def __init__(self, feature_dim, prompt_dim=512, dropout_rate=0.5):
        super(PromptDropout, self).__init__()
        self.feature_dim = feature_dim
        self.prompt_dim = prompt_dim
        self.dropout_rate = dropout_rate
        
        # MLP to generate mask and prompt
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, prompt_dim),
            nn.ReLU(),
            nn.Linear(prompt_dim, feature_dim * 2)  # Output mask and prompt
        )
        
        # Layer Normalization
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, x):
        # x: [batch_size, feature_dim, seq_len]
        batch_size, feature_dim, seq_len = x.shape
        
        # Transpose for MLP: [batch_size, seq_len, feature_dim]
        x_transposed = x.transpose(1, 2)
        
        # Generate mask and prompt
        mask_prompt = self.mlp(x_transposed)  # [batch_size, seq_len, feature_dim * 2]
        mask, prompt = torch.split(mask_prompt, [feature_dim, feature_dim], dim=-1)
        
        # Apply dropout-like mask
        mask = torch.sigmoid(mask)  # Normalize to [0, 1]
        mask = (mask > self.dropout_rate).float()  # Binary mask
        masked_x = x_transposed * mask  # Apply mask
        
        # Add prompt
        prompted_x = masked_x + prompt  # [batch_size, seq_len, feature_dim]
        #prompted_x = masked_x 
        
        # Layer Normalization
        prompted_x = self.norm(prompted_x)
        
        # Transpose back: [batch_size, feature_dim, seq_len]
        prompted_x = prompted_x.transpose(1, 2)
        
        return prompted_x

class VideoTrans_clip(nn.Module):

    def __init__(self, args, num_f_maps, dim, num_classes=100, num_i=6, num_v=10,
                 num_t=15):
        super(VideoTrans_clip, self).__init__()
        self.max_seq_len=4032
        # self.PG = Prediction_Generation(args, num_layers_PG, num_f_maps, dim, num_classes)
        #self.PG = BaseCausalTCN(num_layers_PG, num_f_maps, dim, num_classes)
        self.PG = network_trans.ConvTransformerBackbone(
                    topk=args.topk,
                    n_in=dim, # 768
                    n_embd=num_f_maps, # 512
                    n_head=args.head_num,
                    n_embd_ks=3,
                    scale_factor=args.scale_factor,
                    max_len=4032,
                    mha_win_size=args.winsize,
                    arch = args.arch,
                    with_ln=True,
                    attn_pdrop=0.0,
                    proj_pdrop=0.0,
                    path_pdrop=0.1,
                    use_abs_pe=False,
                    use_rel_pe=False)
        #print(self.PG)
        self.conv_clip = nn.Conv1d(dim, dim, 1)
        self.classifier = nn.Conv1d(args.input_dim, num_classes,1)
        if args.fusion=='self':
            self.fuse = SelfAttentionFusionModule(dim, dim, args.beta)
        elif args.fusion=='cross':
            self.fuse = CrossAttentionFusionModule(dim, dim, args.beta)
        elif args.fusion=='para':
            self.fuse = FeatureFusionModule(dim, dim, args.beta)
        #self.conv_clip = ClipAlign(dim)
        self.channel_dropout = nn.Dropout2d()
        #self.prompt_drop = PromptDropout(dim)

        # self.conv_out_i = nn.Conv1d(num_f_maps, num_i, 1)
        # self.conv_out_v = nn.Conv1d(num_f_maps, num_v, 1)
        # self.conv_out_t = nn.Conv1d(num_f_maps, num_t, 1)
        kernel_size = 3
        self.conv_out = MaskedConv1D(
                num_f_maps, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )
        self.conv_out_i = MaskedConv1D(
                num_f_maps, num_i, kernel_size,
                stride=1, padding=kernel_size//2
            )
        self.conv_out_v = MaskedConv1D(
                num_f_maps, num_v, kernel_size,
                stride=1, padding=kernel_size//2
            )
        self.conv_out_t = MaskedConv1D(
                num_f_maps, num_t, kernel_size,
                stride=1, padding=kernel_size//2
            )
        self.args = args
        
        if args.fpn=='p1':
            self.fpn = Action_FPN(num_f_maps,level=args.arch[2]+1)
        else:
            self.fpn = FPNIdentity(in_channels=[num_f_maps] * (args.arch[2]+1),
            out_channel=num_f_maps,
            scale_factor=args.scale_factor,
            start_level=0,
            with_ln=True)

        if args.text_feat:
            model, preprocess = clip.load("ViT-B/32")
            model = model.cuda()
            text_i = clip.tokenize(['Please pay attention to the instruments in the surgical images.']).cuda()
            text_v = clip.tokenize(['Please pay attention to the action or movement in the surgical images.']).cuda()
            text_t = clip.tokenize(['Please pay attention to the organs and tissues in the surgical images.']).cuda()
            with torch.no_grad():
                self.text_features_i = model.encode_text(text_i)
                self.text_features_v = model.encode_text(text_v)
                self.text_features_t = model.encode_text(text_t)
     

    def forward(self, f, ismask):
        out_list = []
        out_list_i = []
        out_list_v = []
        out_list_t = []
        x = f[0]
        clip_i = f[1]
        clip_t = f[2]

        if self.args.fusion != 'none':
            x, clip_i = self.fuse(x, clip_i)
        x = x.permute(0, 2, 1)
        
        clip_t = clip_t.permute(0, 2, 1)
        clip_i = clip_i.permute(0, 2, 1)
        #clip_i = self.conv_clip(clip_i)
        cls = self.classifier(clip_i)
        #clip_i = torch.nn.functional.normalize(clip_i, p=2, dim=1)

        # clip_i = clip_i.unsqueeze(3)  # of shape (bs, c, l, 1)
        # clip_i = self.channel_dropout(clip_i)
        # clip_i = clip_i.squeeze(3)
        # clip_t = clip_t.unsqueeze(3)  # of shape (bs, c, l, 1)
        # clip_t = self.channel_dropout(clip_t)
        # clip_t = clip_t.squeeze(3)

        x = x.unsqueeze(3)  # of shape (bs, c, l, 1)
        x = self.channel_dropout(x)
        x = x.squeeze(3)
        #x = self.prompt_drop(x)


        #x = x + clip_i.detach()

        # if self.args.mask and ismask:
        #     num_patches = x.flatten().shape[0]
        #     num_mask = int(num_patches * 0.75)
        #     mask = torch.concat((torch.zeros(num_patches - num_mask), torch.ones(num_mask)))
        #     mask = mask[torch.randperm(mask.nelement())]
        #     mask = mask.view(x.shape)

        #     f, out1 = self.PG(x, mask)
        # else:
        #     f, out1 = self.PG(x)
        batched_inputs, batched_masks = self.preprocessing(x)

        # forward the network (backbone -> neck -> heads)
        f, masks = self.PG(batched_inputs, batched_masks)
        #print(len(f), len(masks))
        fpn_feats, fpn_masks = self.fpn(f, masks)
        #print(len(fpn_feats))
        #valid_mask = torch.cat(fpn_masks, dim=1)

        if self.args.text_feat:
            for f, m in zip(fpn_feats, fpn_masks):
                #print(f.shape,fm.shape)
                out_list.append(self.conv_out(f, m))
                text_features_i_expanded = self.text_features_i.unsqueeze(-1).expand(-1, -1, f.shape[-1])
                text_features_v_expanded = self.text_features_v.unsqueeze(-1).expand(-1, -1, f.shape[-1])
                text_features_t_expanded = self.text_features_t.unsqueeze(-1).expand(-1, -1, f.shape[-1])

                out_list_i.append(self.conv_out_i(f + text_features_i_expanded, m))
                out_list_v.append(self.conv_out_v(f+text_features_v_expanded, m))
                out_list_t.append(self.conv_out_t(f+text_features_t_expanded, m))
        else:
            for f, m in zip(fpn_feats, fpn_masks):
                #print(f.shape,fm.shape)
                out_list.append(self.conv_out(f, m))
                out_list_i.append(self.conv_out_i(f, m))
                out_list_v.append(self.conv_out_v(f, m))
                out_list_t.append(self.conv_out_t(f, m))

        
        return out_list, out_list_i, out_list_v, out_list_t, clip_i, clip_t, cls
    
    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        #print(video_list)
        feats = [x for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.cuda()
        batched_masks = batched_masks.unsqueeze(1).cuda()

        return batched_inputs, batched_masks
    
class VideoTrans(nn.Module):

    def __init__(self, args, num_f_maps, dim, num_classes=100, num_i=6, num_v=10,
                 num_t=15):
        super(VideoTrans, self).__init__()
        if getattr(args, 'traditional_task_branches', False) and args.task_prompt != -1:
            raise ValueError('--traditional_task_branches requires --task_prompt -1; do not enable CTA and traditional branches together.')
        if getattr(args, 'vpt_prompt', False):
            if getattr(args, 'traditional_task_branches', False):
                raise ValueError('--vpt_prompt and --traditional_task_branches are separate comparison baselines; enable only one.')
            if args.ins_prompt != -1 or args.target_prompt != -1 or args.verb_prompt != -1 or args.task_prompt != -1:
                raise ValueError('--vpt_prompt is a pure prompt-tuning baseline and requires all other prompt arguments to be -1.')
        if getattr(args, 'st_adapter', False):
            if getattr(args, 'traditional_task_branches', False) or getattr(args, 'vpt_prompt', False):
                raise ValueError('--st_adapter, --vpt_prompt, and --traditional_task_branches are separate comparison baselines; enable only one.')
            if args.ins_prompt != -1 or args.target_prompt != -1 or args.verb_prompt != -1 or args.task_prompt != -1:
                raise ValueError('--st_adapter requires all prompt arguments to be -1 for an isolated ST-Adapter comparison.')
        self.max_seq_len=4032
        # self.PG = Prediction_Generation(args, num_layers_PG, num_f_maps, dim, num_classes)
        #self.PG = BaseCausalTCN(num_layers_PG, num_f_maps, dim, num_classes)
        self.PG = network_trans.ConvTransformerBackbone(
                    topk=args.topk,
                    n_in=dim, # 768
                    n_embd=num_f_maps, # 512
                    n_head=args.head_num,
                    n_embd_ks=3,
                    scale_factor=args.scale_factor,
                    max_len=4032,
                    mha_win_size=args.winsize,
                    arch = args.arch,
                    with_ln=True,
                    attn_pdrop=0.0,
                    proj_pdrop=0.0,
                    path_pdrop=0.1,
                    use_abs_pe=False,
                    use_rel_pe=False,
                    args = args)
        print(self.PG)
        self.channel_dropout = nn.Dropout2d()
        # self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        # self.conv_out_i = nn.Conv1d(num_f_maps, num_i, 1)
        # self.conv_out_v = nn.Conv1d(num_f_maps, num_v, 1)
        # self.conv_out_t = nn.Conv1d(num_f_maps, num_t, 1)
        kernel_size = 3
        self.conv_out = MaskedConv1D(
                num_f_maps, num_classes, kernel_size,
                stride=1, padding=kernel_size//2
            )
        if args.seperate:
            self.conv_out_i = MaskedConv1D(num_classes, num_i, kernel_size,
                    stride=1, padding=kernel_size//2)
            self.conv_out_v = MaskedConv1D(num_classes, num_v, kernel_size,
                    stride=1, padding=kernel_size//2)
            self.conv_out_t = MaskedConv1D(num_classes, num_t, kernel_size,
                    stride=1, padding=kernel_size//2)
        else:
            self.conv_out_i = MaskedConv1D(
                    num_f_maps, num_i, kernel_size,
                    stride=1, padding=kernel_size//2
                )
            self.conv_out_v = MaskedConv1D(
                    num_f_maps, num_v, kernel_size,
                    stride=1, padding=kernel_size//2
                )
            self.conv_out_t = MaskedConv1D(
                    num_f_maps, num_t, kernel_size,
                    stride=1, padding=kernel_size//2
                )
        self.args = args
        self.use_traditional_task_branches = getattr(args, 'traditional_task_branches', False)
        self.traditional_task_branches = nn.ModuleList([
            TaskFeatureBranch(args, num_f_maps) for _ in range(args.task_num)
        ]) if self.use_traditional_task_branches else nn.ModuleList()
        
        if args.fpn =='p1':
            self.fpn = Action_FPN(num_f_maps,level=args.arch[2]+1)
        else:
            self.fpn = FPNIdentity(in_channels=[num_f_maps] * (args.arch[2]+1),
            out_channel=num_f_maps,
            scale_factor=args.scale_factor,
            start_level=0,
            with_ln=True)

        # if args.text_feat:
        #     model, preprocess = clip.load("ViT-B/32")
        #     model = model.cuda()
        #     text_i = clip.tokenize(['Please pay attention to the instruments in the surgical images.']).cuda()
        #     text_v = clip.tokenize(['Please pay attention to the action or movement in the surgical images.']).cuda()
        #     text_t = clip.tokenize(['Please pay attention to the organs and tissues in the surgical images.']).cuda()
        #     with torch.no_grad():
        #         self.text_features_i = model.encode_text(text_i)
        #         self.text_features_v = model.encode_text(text_v)
        #         self.text_features_t = model.encode_text(text_t)
     

    def forward(self, x, ismask):
        out_list = []
        out_list_i = []
        out_list_v = []
        out_list_t = []
        if self.args.ins_prompt != -1:
            orig_x = x[0]
            #concat the dim 2 and 3 of x[1]
            tip_feat = x[1].reshape(1, -1, x[1].shape[2] * x[1].shape[3])
            shaft_feat = x[2].reshape(1, -1, x[2].shape[2] * x[2].shape[3])
            wrist_feat = x[3].reshape(1, -1, x[3].shape[2] * x[3].shape[3])
            #verb_feat = x[4].reshape(1, -1, x[4].shape[2] * x[4].shape[3])
            #target_feat = x[5].reshape(1, -1, x[5].shape[2] * x[5].shape[3])

            # concat tip_feat, shaft_feat, wrist_feat   
            #x_prompt = torch.cat([tip_feat, shaft_feat, wrist_feat, verb_feat, target_feat], dim=2)
            x_prompt = torch.cat([tip_feat, shaft_feat, wrist_feat], dim=2)

            orig_x = orig_x.permute(0, 2, 1)
            orig_x = orig_x.unsqueeze(3)
            orig_x = self.channel_dropout(orig_x)
            orig_x = orig_x.squeeze(3)
            
            batched_inputs, batched_masks = self.preprocessing(orig_x)
            x_prompt = x_prompt.permute(0, 2, 1)
            batched_inputs_prompt, batched_masks_prompt = self.preprocessing(x_prompt)

            if self.args.task_prompt != -1:
                f, masks, task_prompts_cos = self.PG([batched_inputs, batched_inputs_prompt], batched_masks)
            else:
                f, masks = self.PG([batched_inputs, batched_inputs_prompt], batched_masks)

        # if self.args.verb_prompt != -1:
        #     orig_x = x[0]
        #     #concat the dim 2 and 3 of x[1]
        #     target_feat = x[1].reshape(1, -1, x[1].shape[2] * x[1].shape[3])

        #     # concat tip_feat, shaft_feat, wrist_feat   
        #     x_prompt = target_feat

        #     orig_x = orig_x.permute(0, 2, 1)
        #     orig_x = orig_x.unsqueeze(3)
        #     orig_x = self.channel_dropout(orig_x)
        #     orig_x = orig_x.squeeze(3)
            
        #     batched_inputs, batched_masks = self.preprocessing(orig_x)
        #     x_prompt = x_prompt.permute(0, 2, 1)
        #     batched_inputs_prompt, batched_masks_prompt = self.preprocessing(x_prompt)

        #     f, masks = self.PG([batched_inputs, batched_inputs_prompt], batched_masks)

        else:
            x = x.permute(0, 2, 1)

            x = x.unsqueeze(3)  # of shape (bs, c, l, 1)
            x = self.channel_dropout(x)
            x = x.squeeze(3)

            # if self.args.mask and ismask:
            #     num_patches = x.flatten().shape[0]
            #     num_mask = int(num_patches * 0.75)
            #     mask = torch.concat((torch.zeros(num_patches - num_mask), torch.ones(num_mask)))
            #     mask = mask[torch.randperm(mask.nelement())]
            #     mask = mask.view(x.shape)

            #     f, out1 = self.PG(x, mask)
            # else:
            #     f, out1 = self.PG(x)
            batched_inputs, batched_masks = self.preprocessing(x)

            # forward the network (backbone -> neck -> heads)
            if self.args.task_prompt != -1:
                f, masks, task_prompts_cos = self.PG(batched_inputs, batched_masks)
            else:
                f, masks = self.PG(batched_inputs, batched_masks)

        #print(len(f), len(masks))
        #fpn_feats, fpn_masks = self.fpn(f, masks)
        fpn_feats, fpn_masks= self.fpn(f, masks)
        #print(len(fpn_feats))
        #valid_mask = torch.cat(fpn_masks, dim=1)

        if self.args.text_feat:
            for f, m in zip(fpn_feats, fpn_masks):
                #print(f.shape,fm.shape)
                out_list.append(self.conv_out(f, m))
                text_features_i_expanded = self.text_features_i.unsqueeze(-1).expand(-1, -1, f.shape[-1])
                text_features_v_expanded = self.text_features_v.unsqueeze(-1).expand(-1, -1, f.shape[-1])
                text_features_t_expanded = self.text_features_t.unsqueeze(-1).expand(-1, -1, f.shape[-1])

                out_list_i.append(self.conv_out_i(f + text_features_i_expanded, m))
                out_list_v.append(self.conv_out_v(f+text_features_v_expanded, m))
                out_list_t.append(self.conv_out_t(f+text_features_t_expanded, m))
        elif self.args.task_prompt != -1:
            for f, m in zip(fpn_feats, fpn_masks):
                out_list_i.append(self.conv_out_i(f[0].unsqueeze(0), m[0].unsqueeze(0)))
                out_list_v.append(self.conv_out_v(f[1].unsqueeze(0), m[1].unsqueeze(0)))
                out_list_t.append(self.conv_out_t(f[2].unsqueeze(0), m[2].unsqueeze(0)))
                # normalize the task_prompts_cos
                # weights = task_prompts_cos[0,:,3]
                # weights = weights / weights.sum()
                # weight_i, weight_v, weight_t = weights[0], weights[1], weights[2]


                # average the features in dim 0
                # f_fuse = (f[0] + f[1] + f[2])/3
                # f_fuse = f_fuse.unsqueeze(0)
                fuse = 0.1
                f_fuse = fuse*f[0] + fuse*f[1] + fuse*f[2] + fuse*f[3] + f[4]
                #f_fuse = weight_i*f[0] + weight_v*f[1] + weight_t*f[2] + f[4]

                out_list.append(self.conv_out(f_fuse.unsqueeze(0), m[4].unsqueeze(0)))
        elif self.use_traditional_task_branches:
            branch_fpn_feats = tuple()
            branch_fpn_masks = tuple()
            fuse = 0.1
            for f, m in zip(fpn_feats, fpn_masks):
                task_feats = [branch(f, m)[0] for branch in self.traditional_task_branches]
                task_masks = [m for _ in task_feats]

                out_list_i.append(self.conv_out_i(task_feats[0], m))
                out_list_v.append(self.conv_out_v(task_feats[1], m))
                out_list_t.append(self.conv_out_t(task_feats[2], m))

                f_fuse = fuse * (task_feats[0] + task_feats[1] + task_feats[2] + task_feats[3]) + f
                out_list.append(self.conv_out(f_fuse, m))

                branch_fpn_feats += (torch.cat(task_feats + [f], dim=0), )
                branch_fpn_masks += (torch.cat(task_masks + [m], dim=0), )
            fpn_feats, fpn_masks = branch_fpn_feats, branch_fpn_masks
        else:
            for f, m in zip(fpn_feats, fpn_masks):
                #print(f.shape,fm.shape)
                if self.args.seperate:
                    ivt = self.conv_out(f, m)
                    out_list.append(self.conv_out(f, m))
                    out_list_i.append(self.conv_out_i(ivt[0], m))
                    out_list_v.append(self.conv_out_v(ivt[0], m))
                    out_list_t.append(self.conv_out_t(ivt[0], m))
                else:
                    out_list.append(self.conv_out(f, m))
                    out_list_i.append(self.conv_out_i(f, m))
                    out_list_v.append(self.conv_out_v(f, m))
                    out_list_t.append(self.conv_out_t(f, m))

        return out_list, out_list_i, out_list_v, out_list_t, fpn_feats, fpn_masks, 0
    
    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        #print(video_list)
        feats = [x for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(
                feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.cuda()
        batched_masks = batched_masks.unsqueeze(1).cuda()

        return batched_inputs, batched_masks

class FPN(nn.Module):
    def __init__(self, num_f_maps):
        super(FPN, self).__init__()
        self.latlayer1 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, W = y.size()
        return F.interpolate(x, size=W, mode='linear') + y

    def forward(self, out_list):
        p4 = out_list[3]
        c3 = out_list[2]
        c2 = out_list[1]
        c1 = out_list[0]
        p3 = self._upsample_add(p4, self.latlayer1(c3))
        p2 = self._upsample_add(p3, self.latlayer1(c2))
        p1 = self._upsample_add(p2, self.latlayer1(c1))
        return [p1, p2, p3, p4]

class Action_FPN(nn.Module):
    def __init__(self, num_f_maps, level):
        super(Action_FPN, self).__init__()
        # self.latlayer1 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
        # self.latlayer2 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)

        # self.latlayer3 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
        # self.latlayer4 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
        # self.latlayer5 = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
        self.fpn_layer = nn.ModuleList()
        self.level = level
        for i in range(self.level-1):
            fpn = nn.Conv1d(num_f_maps, num_f_maps, kernel_size=1, stride=1, padding=0)
            self.fpn_layer.append(fpn)

        self.fpn_norms = nn.ModuleList()
        for i in range(self.level):
            fpn_norm = LayerNorm(num_f_maps)
            self.fpn_norms.append(fpn_norm)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, W = y.size()
        return F.interpolate(x, size=W, mode='linear',align_corners=True) + y

    def forward(self, out_list, fpn_masks):
        out_list = list(out_list)
        for i in range(len(self.fpn_layer)-1, -1 ,-1):
            out_list[i] = self._upsample_add(out_list[i+1], self.fpn_layer[i](out_list[i]))
        #out_list = tuple(out_list)

        fpn_feats = tuple()
        new_fpn_masks = tuple()
        for i in range(len(self.fpn_norms)):
            x = self.fpn_norms[i](out_list[i])
            fpn_feats += (x, )
            new_fpn_masks += (fpn_masks[i], )

        return fpn_feats, new_fpn_masks

        # p4 = out_list[3]
        # c3 = out_list[2]
        # c2 = out_list[1]
        # c1 = out_list[0]
        # p3 = self._upsample_add(p4, self.latlayer1(c3))
        # p2 = self._upsample_add(p3, self.latlayer2(c2))
        # p1 = self._upsample_add(p2, self.latlayer3(c1))
        # return [p1, p2, p3, p4]
        

class BaseCausalTCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        print(num_layers)
        super(BaseCausalTCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.channel_dropout = nn.Dropout2d()
        # self.downsample = nn.Linear(num_f_maps,num_f_maps, kernel_size=3, stride=2,dilation=3)
        # self.center = torch.nn.Parameter(torch.zeros(1, 64, num_classes), requires_grad=False)
        self.num_classes = num_classes

    def forward(self, x, labels=None, mask=None, test=False):
        # x = x.permute(0,2,1) # (bs,l,c) -> (bs, c, l)

        if mask is not None:
            # print(x.size(),mask.size())
            x = x * mask

        x = x.unsqueeze(3)  # of shape (bs, c, l, 1)
        x = self.channel_dropout(x)
        x = x.squeeze(3)

        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)

        x = self.conv_out(out)  # (bs, c, l)

        return out, x


class Refinement(nn.Module):
    def __init__(self, args, num_layers, num_f_maps, dim, num_classes, conv_out):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        # self.conv_out = conv_out
        self.max_pool_1x1 = nn.AvgPool1d(kernel_size=7, stride=3)
        self.use_output = args.output
        self.hier = args.hier

    def forward(self, x):
        if self.use_output:
            out = self.conv_1x1(x)
        else:
            out = x
        for layer in self.layers:
            out = layer(out)
            # print(out.max(), out.min())
        if self.hier:
            f = self.max_pool_1x1(out)
        else:
            f = out
        out = self.conv_out(f)

        return f, out


class DilatedResidualCausalLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, padding=None):
        super(DilatedResidualCausalLayer, self).__init__()
        if padding == None:

            self.padding = 2 * dilation
        else:
            self.padding = padding
        # causal: add padding to the front of the input
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=0, dilation=dilation)  #
        # self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.pad(x, [self.padding, 0], 'constant', 0)
        # print(self.padding)  # add padding to the front of input
        out = F.relu(self.conv_dilated(out))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out)


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)

        return x + out
