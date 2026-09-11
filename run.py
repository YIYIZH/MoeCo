#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import libraries
from ipaddress import v4_int_to_packed
from itertools import count
from math import log
import os
# Disable proxy for wandb
# os.environ['http_proxy'] = ''
# os.environ['https_proxy'] = ''
# os.environ['HTTP_PROXY'] = ''
# os.environ['HTTPS_PROXY'] = ''
# os.environ['no_proxy'] = '*'
# os.environ['NO_PROXY'] = '*'
import sys
import time
from turtle import pos
import torch
import random
#from LibMTL.LibMTL import loss
import network, network_trans
import argparse
import platform
import ivtmetrics  # You must "pip install ivtmetrics" to use
import dataloader
import numpy as np
from torch import cosine_similarity, eq, nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import pickle
import matplotlib.pyplot as plt
import wandb
import torch.distributed as dist
from loss.eqlv2 import EQLv2
from loss.blloss import EQL
np.seterr(invalid='ignore')
# %% @args parsing
# %% @args parsing
parser = argparse.ArgumentParser()
# model

parser.add_argument('--version', type=str, default='', help='Model version control (for keeping several versions)')
parser.add_argument('--version1', type=str, default='', help='Model version control (for keeping several versions)')
parser.add_argument('--hr_output', action='store_true',
                    help='Whether to use higher resolution output (32x56) or not (8x14). Default: False')
parser.add_argument('--latest', action='store_true', help='to test')
parser.add_argument('--use_ln', action='store_true',
                    help='Whether to use layer norm or batch norm in AddNorm() function. Default: False')
parser.add_argument('--decoder_layer', type=int, default=8, help='Number of MHMA layers ')
# job
parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')
parser.add_argument('-t', '--train', action='store_true', help='to train.')
parser.add_argument('-e', '--test', action='store_true', help='to test')
parser.add_argument('--val_interval', type=int, default=1,
                    help='(for hp tuning). Epoch interval to evaluate on validation data. set -1 for only after final epoch, or a number higher than the total epochs to not validate.')
# data
parser.add_argument('--data_dir', type=str, default='/mnt/ssd/data/CholecT45/',
                    help='path to dataset?')
# parser.add_argument('--data_dir', type=str, default='/SDIM/shuangchun/Data/Video/Surgical/CholecT45/CholecT45',
#                     help='path to dataset?')
# parser.add_argument('--data_dir', type=str, default='/home/shuangchun/Data/Video/CholecT45/CholecT45',
#                     help='path to dataset?')
parser.add_argument('--dataset_variant', type=str, default='cholect45-crossval',
                    choices=['cholect50', 'cholect45', 'chdataloaderolect50-challenge', 'cholect50-crossval',
                             'cholect45-crossval', 'cholect45-challenge'], help='Variant of the dataset to use')
parser.add_argument('-k', '--kfold', type=int, default=1,
                    help='The test split in k-fold cross-validation')
parser.add_argument('--train_div', type=int, default=1, help='path to pretrain_weight?')
parser.add_argument('--image_width', type=int, default=448, help='Image width ')
parser.add_argument('--image_height', type=int, default=256, help='Image height ')
parser.add_argument('--image_channel', type=int, default=3, help='Image channels ')
parser.add_argument('--num_tool_classes', type=int, default=6, help='Number of tool categories')
parser.add_argument('--num_verb_classes', type=int, default=10, help='Number of verb categories')
parser.add_argument('--num_target_classes', type=int, default=15, help='Number of target categories')
parser.add_argument('--num_triplet_classes', type=int, default=100, help='Number of triplet categories')
parser.add_argument('--augmentation_list', type=str, nargs='*',
                    default=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
                    help='List augumentation styles (see dataloader.py for list of supported styles).')
parser.add_argument('--pin_memory', action='store_true',
                    help='Enable DataLoader pin_memory. Disabled by default to avoid CUDA invalid-argument errors in the pin-memory thread.')
# hp
parser.add_argument('-b', '--batch', type=int, default=32, help='The size of sample training batch')
parser.add_argument('--epochs', type=int, default=1000, help='How many training epochs?')
parser.add_argument('-w', '--warmups', type=int, nargs='+', default=[9, 18, 58],
                    help='List warmup epochs for tool, verb-target, triplet respectively')
parser.add_argument('-l', '--initial_learning_rates', type=float, nargs='+', default=[0.01, 0.01, 0.01],
                    help='List learning rates for tool, verb-target, triplet respectively')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 regularization weight decay constant')
parser.add_argument('--decay_steps', type=int, default=10, help='Step to exponentially decay')
parser.add_argument('--decay_rate', type=float, default=0.99, help='Learning rates weight decay rate')
parser.add_argument('--momentum', type=float, default=0.95, help="Optimizer's momentum")
parser.add_argument('--power', type=float, default=0.1, help='Learning rates weight decay power')
# weights
parser.add_argument('--pretrain_dir', type=str, default='', help='path to pretrain_weight?')
parser.add_argument('--test_ckpt', type=str, default=None, help='path to model weight for testing')
parser.add_argument('--loss_type', type=str, default='focal', help='path to pretrain_weight?')
parser.add_argument('--pos_w', type=str, default='pre_w', help='path to pretrain_weight?')
####ms-tcn2
parser.add_argument('--num_layers_PG', default="11", type=int)
parser.add_argument('--num_layers_R', default="10", type=int)
parser.add_argument('--num_R', default="3", type=int)


parser.add_argument('--mask', action='store_true')
parser.add_argument('--output', default=False, type=bool)
parser.add_argument('--feature', default=False, type=bool)
parser.add_argument('--trans', default=False, type=bool)
parser.add_argument('--prototype', default=False, type=bool)
parser.add_argument('--last', default=False, type=bool)
parser.add_argument('--first', default=False, type=bool)
parser.add_argument('--hier', default=False, type=bool)

##Transformer
parser.add_argument('-sf','--scale_factor', type=int, default=1)
parser.add_argument('--winsize', type=int, default=19)
parser.add_argument('--head_num', type=int, default=4)
parser.add_argument('--embed_num', type=int, default=512)
parser.add_argument('--input_dim', type=int, default=512)
parser.add_argument('--block_num', type=int, default=1)
parser.add_argument('--positional_encoding_type', default="learned", type=str, help="fixed or learned")
parser.add_argument('--arch', type=int, nargs='+', default=[6,5,5])
##actionformer
parser.add_argument('--fpn', type=str, default='p1')
parser.add_argument('--alpha', default=0.5, type=float)
parser.add_argument('--random', default=0.7, type=float)
parser.add_argument('--reverse', action='store_true')
parser.add_argument('--step', action='store_true')
parser.add_argument('--transit', default=1, type=float)
parser.add_argument('--focal' , action='store_true')
parser.add_argument('--cb', action='store_true')
parser.add_argument('--eql', action='store_true')
parser.add_argument('--eqlv2', action='store_true')
parser.add_argument('--gamma', default=0, type=float)
parser.add_argument('--beta', default=0.5, type=float)
parser.add_argument('--seperate', action='store_true')
#clip
parser.add_argument('--model', type=str, default='actionformer', choices=['mstcn','actionformer','clip'], help='Model name?')
parser.add_argument('--clip_image', action='store_true')
parser.add_argument('--clip_text', action='store_true') # add at the beginning of the model
parser.add_argument('--text_feat', action='store_true') # add at the end of the model
parser.add_argument('--clip_i_feature', type=str, default='TERL/0_5fold_TCN_black/clip_features_ViT-L-14_feats_32.pkl')
parser.add_argument('--clip_t_feature', type=str, default='TERL/0_5fold_TCN_black/clip_features_ViT-L-14_feats_32_text.pkl')
parser.add_argument('--fuse', type=str, default='add')
parser.add_argument('--clip_loss', type=str, default='cos')
parser.add_argument('--norm', action='store_true')
parser.add_argument('--fusion', type=str, default='para')

#prompt
parser.add_argument('--ins_prompt', type=int, default=-1)
parser.add_argument('--verb_prompt', type=int, default=-1)
parser.add_argument('--target_prompt', type=int, default=-1)
parser.add_argument('--task_prompt', type=int, default=-1)
parser.add_argument('--task_num', type=int, default=4)
parser.add_argument('--ins_prompt_source', type=str, default='gmm', choices=['gmm', 'random', 'random_gmm', 'gt_attribute'],
                    help='Source of instrument prompt features for KD-MoE: GMM-activated prototypes, randomly initialized features, randomly activated GMM prototypes, or GT instrument attribute oracle prompts.')
parser.add_argument('--random_ins_prompt_std', type=float, default=1.0,
                    help='Standard deviation for random instrument prompt features when --ins_prompt_source random.')
parser.add_argument('--random_ins_prompt_no_normalize', action='store_true',
                    help='Do not L2-normalize random instrument prompt features.')
parser.add_argument('--traditional_task_branches', action='store_true',
                    help='Use four independent task-specific feature branches after the shared ActionFormer/FPN backbone. Requires --task_prompt -1.')
parser.add_argument('--vpt_prompt', action='store_true',
                    help='Use VPT-style task-agnostic learnable prompt tokens in the ActionFormer stem. Mutually exclusive with CTA, prompt-MoE, and traditional branches.')
parser.add_argument('--vpt_prompt_len', type=int, default=4,
                    help='Number of VPT prompt tokens. Default 4 matches CTA task prompt count.')
parser.add_argument('--vpt_prompt_layers', type=int, nargs='+', default=[4],
                    help='Stem layer indices where VPT prompts are inserted, locally attended with temporary non-learnable padding, and removed after that layer.')
parser.add_argument('--st_adapter', action='store_true',
                    help='Use ST-Adapter style residual adapters after each ActionFormer transformer block. Mutually exclusive with CTA, VPT, prompt-MoE, and traditional branches.')
parser.add_argument('--st_adapter_dim', type=int, default=512,
                    help='Bottleneck width for ST-Adapter. Default 512 makes this comparison slightly larger than CTA in parameter count.')
parser.add_argument('--st_adapter_kernel_size', type=int, default=3,
                    help='Temporal depthwise convolution kernel size used inside ST-Adapter.')
parser.add_argument('--cgl_split_source', type=str, default='full', choices=['full', 'train'],
                    help='Class-frequency source for CGL head/medium/tail grouping.')
parser.add_argument('--cgl_split_mode', type=str, default='absolute', choices=['absolute', 'percentile', 'ratio'],
                    help='CGL grouping mode: absolute count thresholds, top/bottom percentages, or class-frequency ratios.')
parser.add_argument('--cgl_head_threshold', type=float, default=10000,
                    help='Head-class count threshold for --cgl_split_mode absolute.')
parser.add_argument('--cgl_tail_threshold', type=float, default=1000,
                    help='Tail-class count threshold for --cgl_split_mode absolute.')
parser.add_argument('--cgl_head_percent', type=float, default=0.10,
                    help='Top percentage used as head classes for --cgl_split_mode percentile.')
parser.add_argument('--cgl_tail_percent', type=float, default=0.10,
                    help='Bottom percentage used as tail classes for --cgl_split_mode percentile.')
parser.add_argument('--cgl_head_ratio', type=float, default=0.07,
                    help='Head-class relative-frequency threshold for --cgl_split_mode ratio.')
parser.add_argument('--cgl_tail_ratio', type=float, default=0.007,
                    help='Tail-class relative-frequency threshold for --cgl_split_mode ratio.')
parser.add_argument('--cgl_full_stats_path', type=str, default='',
                    help='Optional path to full-dataset class statistics JSON. Defaults to 0_5fold_TCN_black/all_data.json.')
parser.add_argument('--topk', type=int, default=3)


# FLOPs profiling
parser.add_argument('--flops_debug', action='store_true',
                    help='Print fvcore FLOPs breakdown, unsupported ops, and selected-module stats.')
parser.add_argument('--flops_profile_train', action='store_true',
                    help='Profile the training/mask path by calling model(..., True). Default profiles runtime/eval path.')
parser.add_argument('--flops_fixed_seq_len', type=int, default=-1,
                    help='If >0, crop/pad temporal inputs to this fixed length for fair ablation comparison.')
parser.add_argument('--flops_module_keywords', type=str, nargs='*',
                    default=['kd', 'moe', 'expert', 'gaussian', 'prompt'],
                    help='Keywords used to identify KD-MoE/prompt modules in fvcore by_module() output.')
parser.add_argument('--flops_baseline_override', type=float, default=-1.0,
                    help='Optional baseline FLOPs value to add to selected module FLOPs. Use raw FLOPs, e.g. 5.4e7.')
parser.add_argument('--flops_baseline_per_frame_override', type=float, default=-1.0,
                    help='Optional baseline FLOPs/frame to add to selected module FLOPs/frame. Alternative to --flops_baseline_override.')
parser.add_argument('--flops_report_file', type=str, default='',
                    help='Optional path to save detailed FLOPs profiling report as a text file.')
parser.add_argument('--profile_inference_fps', action='store_true',
                    help='Measure model-only inference FPS during test/validation loops.')
parser.add_argument('--fps_warmup_videos', type=int, default=0,
                    help='Number of initial videos per test/validation loop excluded from FPS timing.')

# device
parser.add_argument('--gpu', type=str, default="0,1,2",
                    help='The gpu device to use. To use multiple gpu put all the device ids comma-separated, e.g: "0,1,2" ')
FLAGS, unparsed = parser.parse_known_args()

random.seed(FLAGS.seed)
np.random.seed(FLAGS.seed)
torch.manual_seed(FLAGS.seed)
torch.cuda.manual_seed_all(FLAGS.seed)
# %% @params definitions
alpha = FLAGS.alpha
is_train = FLAGS.train
is_test = FLAGS.test
dataset_variant = FLAGS.dataset_variant
data_dir = FLAGS.data_dir
kfold = FLAGS.kfold if "crossval" in dataset_variant else 0
version = FLAGS.version
hr_output = FLAGS.hr_output
use_ln = FLAGS.use_ln
batch_size = FLAGS.batch
pretrain_dir = FLAGS.pretrain_dir
test_ckpt = FLAGS.test_ckpt
weight_decay = FLAGS.weight_decay
learning_rates = FLAGS.initial_learning_rates
warmups = FLAGS.warmups
decay_steps = FLAGS.decay_steps
decay_rate = FLAGS.decay_rate
power = FLAGS.power
momentum = FLAGS.momentum
epochs = FLAGS.epochs
gpu = FLAGS.gpu
image_height = FLAGS.image_height
image_width = FLAGS.image_width
image_channel = FLAGS.image_channel
num_triplet = FLAGS.num_triplet_classes
num_tool = FLAGS.num_tool_classes
num_verb = FLAGS.num_verb_classes
num_target = FLAGS.num_target_classes
val_interval = FLAGS.epochs - 1 if FLAGS.val_interval == -1 else FLAGS.val_interval
set_chlg_eval = True  # To observe challenge evaluation protocol
gpu = ",".join(str(FLAGS.gpu).split(","))
decodelayer = FLAGS.decoder_layer
addnorm = "layer" if use_ln else "batch"
modelsize = "high" if hr_output else "low"
FLAGS.multigpu = len(gpu) > 1  # not yet implemented !
mheaders = ["", "l", "cholect", "k"]
margs = [FLAGS.model, decodelayer, dataset_variant, kfold]
wheaders = ["norm", "res"]
wargs = [addnorm, modelsize]
modelname = "_".join(["{}{}".format(x, y) for x, y in zip(mheaders, margs) if len(str(y))]) + "_" + \
            "_".join(["{}{}".format(x, y) for x, y in zip(wargs, wheaders) if len(str(x))])
model_dir = "./__checkpoint__/run_{}".format(version)
if not os.path.exists(model_dir): os.makedirs(model_dir)
resume_ckpt = None
ckpt_path = os.path.join(model_dir, '{}.pth'.format(modelname))
ckpt_path_epoch = os.path.join(model_dir, '{}'.format(modelname))
logfile = os.path.join(model_dir, '{}.log'.format(modelname))
data_augmentations = FLAGS.augmentation_list
iterable_augmentations = []
if FLAGS.latest:
    FLAGS.version1 = FLAGS.version1 + '_latest'
print("Configuring network ...")

# with open('/home/student/Documents/data/CholecT45/dict/maps.txt', 'r') as f:
#     lines = f.readlines()[1:]
#     ivt_maps = [list(map(int, line.strip().split(','))) for line in lines]

# class_num = [100, 6, 10, 15]
# ivt_maps = np.stack(ivt_maps)

ivt_head = [17, 60, 19]
ivt_medium = [58, 7, 20, 12, 94, 61, 96, 82, 59, 57, 29, 79, 16]  # Medium frequency IVT indices
ivt_tail = [78, 69, 1, 18, 68, 95, 99, 63, 14, 27, 88, 4, 22, 92, 36, 28, 62, 98, 21, 30, 51, 10, 13, 52, 64, 37, 23, 97, 44, 6, 66, 34, 90, 33, 87, 39, 76, 71, 84, 93, 40, 0, 53, 26, 3, 32, 45, 24, 9, 31, 25, 73, 35, 81, 11, 75, 15, 48, 83, 77, 43, 2, 91, 86, 89, 5, 72, 46, 56, 67, 70, 65, 49, 80, 74, 47, 85, 42, 50, 8, 38, 41, 54, 55]
i_head =[0,2]
i_medium = [3,4]
i_tail = [1,5]
v_head = [1,2]
v_medium = [0,9]
v_tail = [3,4,5,6,7,8]
t_head = [0]
t_medium = [1,2,3,6,8,10,13,14]
t_tail = [4,5,7,9,11,12]

def mean_ap_by_indices(ap, indices):
    vals = [ap[idx] for idx in indices if idx < len(ap)]
    return np.nanmean(vals) if vals else np.nan

def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def infer_num_frames_from_labels(y4):
    """Infer the number of temporal predictions represented by one dataloader item."""
    if torch.is_tensor(y4):
        if y4.dim() >= 2:
            return int(y4.shape[1])
        return int(y4.shape[0])
    return 0

def count_model_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable
    return trainable, total, frozen


def _normalize_flops_tensor(x):
    """Normalize one dataloader tensor while preserving temporal layout.

    The temporal models in this code commonly receive [B, T, C]. Some dataset
    items are [T, C] before DataLoader batching, so we add the batch dimension
    only in that case.
    """
    if not torch.is_tensor(x):
        raise TypeError(f"FLOPs input must be a tensor, but got {type(x)}")
    if x.dim() == 2:
        x = x.unsqueeze(0)
    return x.contiguous()


def prepare_flops_input(img):
    """Prepare profiling inputs without changing the model's input structure.

    Important: when prompt/KD-MoE is enabled, the training code forwards the
    whole img list into the model. Selecting only img[0] profiles a different
    path and can underestimate FLOPs.
    """
    if isinstance(img, list):
        input_kind = 'list'
        inputs = img
    elif isinstance(img, tuple):
        input_kind = 'tuple'
        inputs = list(img)
    else:
        input_kind = 'single'
        inputs = [img]
    return tuple(_normalize_flops_tensor(x) for x in inputs), input_kind


def _move_flops_inputs_to_model_device(inputs, model):
    device = next(model.parameters()).device
    return tuple(x.to(device, non_blocking=True) for x in inputs)


def _flatten_tensors(obj):
    """Flatten nested model outputs and keep only tensors for fvcore tracing."""
    if torch.is_tensor(obj):
        return [obj]
    if isinstance(obj, dict):
        tensors = []
        for key in sorted(obj.keys()):
            tensors.extend(_flatten_tensors(obj[key]))
        return tensors
    if isinstance(obj, (list, tuple)):
        tensors = []
        for item in obj:
            tensors.extend(_flatten_tensors(item))
        return tensors
    return []


class _FlopCountWrapper(nn.Module):
    """Wrap model.forward so fvcore sees tensor-only inputs/outputs."""
    def __init__(self, model, input_kind='single', ismask=False):
        super().__init__()
        self.model = model
        self.input_kind = input_kind
        self.ismask = ismask

    def forward(self, *xs):
        if self.input_kind == 'list':
            model_input = list(xs)
        elif self.input_kind == 'tuple':
            model_input = tuple(xs)
        else:
            model_input = xs[0]

        outputs = self.model(model_input, self.ismask)
        tensors = _flatten_tensors(outputs)
        if len(tensors) == 0:
            raise RuntimeError("Model forward produced no tensor outputs for FLOPs tracing.")
        # Returning a tuple avoids invalid additions between different heads and
        # still forces fvcore to trace the full forward graph.
        return tensors[0] if len(tensors) == 1 else tuple(tensors)


def _infer_profile_seq_len(inputs):
    """Infer temporal length from the first profiling tensor."""
    x = inputs[0]
    input_dim = getattr(FLAGS, 'input_dim', None)
    if x.dim() == 3:
        # Usually [B, T, C]. If the last dim is not feature dim, assume [B, C, T].
        tdim = 1 if (input_dim is None or x.shape[-1] == input_dim) else 2
        return int(x.shape[tdim])
    if x.dim() == 2:
        return int(x.shape[0])
    return int(x.shape[0])


def _temporal_dim(x):
    input_dim = getattr(FLAGS, 'input_dim', None)
    if x.dim() == 3:
        return 1 if (input_dim is None or x.shape[-1] == input_dim) else 2
    if x.dim() == 2:
        return 0
    return None


def _resize_one_flops_tensor(x, seq_len):
    """Crop or right-pad one profiling tensor to a fixed temporal length."""
    if seq_len is None or seq_len <= 0:
        return x
    tdim = _temporal_dim(x)
    if tdim is None:
        return x
    cur_len = x.shape[tdim]
    if cur_len == seq_len:
        return x.contiguous()
    if cur_len > seq_len:
        slices = [slice(None)] * x.dim()
        slices[tdim] = slice(0, seq_len)
        return x[tuple(slices)].contiguous()
    pad_shape = list(x.shape)
    pad_shape[tdim] = seq_len - cur_len
    pad = x.new_zeros(pad_shape)
    return torch.cat([x, pad], dim=tdim).contiguous()


def _resize_flops_inputs(inputs, seq_len):
    return tuple(_resize_one_flops_tensor(x, seq_len) for x in inputs)


def _match_module_keywords(module_name, keywords):
    name = module_name.lower()
    return any(k.lower() in name for k in keywords if k)


def _is_descendant_module(child, parent):
    return child.startswith(parent + '.')


def _summarize_selected_module_flops(by_module, keywords, total_flops, seq_len):
    """Sum FLOPs of modules whose names match user-provided keywords.

    fvcore by_module() is hierarchical, so summing every matching parent and
    child module can double count. We therefore sum only the top-most matching
    modules: if ``moe`` matches, ``moe.expert1`` is shown for debugging but is
    not added again.
    """
    all_matched = []
    for name, flops in by_module.items():
        if name == '':
            continue
        if _match_module_keywords(name, keywords):
            all_matched.append((name, int(flops)))

    # Non-overlapping top-most modules for the actual selected FLOPs sum.
    selected_roots = []
    for name, flops in sorted(all_matched, key=lambda kv: kv[0].count('.')):
        if any(name == parent or _is_descendant_module(name, parent) for parent, _ in selected_roots):
            continue
        selected_roots.append((name, flops))

    all_matched = sorted(all_matched, key=lambda kv: kv[1], reverse=True)
    selected_roots = sorted(selected_roots, key=lambda kv: kv[1], reverse=True)
    selected_total = int(sum(v for _, v in selected_roots))
    selected_per_frame = selected_total / seq_len if selected_total is not None and seq_len > 0 else None
    ratio = selected_total / total_flops if total_flops not in (None, 0) else None
    return {
        'keywords': list(keywords),
        'matched_modules': all_matched,
        'selected_root_modules': selected_roots,
        'selected_flops': selected_total,
        'selected_flops_per_frame': selected_per_frame,
        'selected_ratio': ratio,
    }


def _format_flops_report(profile_detail, max_modules=30):
    lines = []
    if not profile_detail:
        return ''
    lines.append('FLOPs detail | input_kind: {} | seq_len: {} | profile_train: {}'.format(
        profile_detail.get('input_kind'), profile_detail.get('seq_len'), profile_detail.get('profile_train')))
    lines.append('FLOPs detail | by_operator: {}'.format(profile_detail.get('by_operator')))
    lines.append('FLOPs detail | unsupported_ops: {}'.format(profile_detail.get('unsupported_ops')))
    lines.append('FLOPs detail | uncalled_modules_count: {}'.format(len(profile_detail.get('uncalled_modules', []))))
    lines.append('FLOPs detail | first_uncalled_modules: {}'.format(profile_detail.get('uncalled_modules', [])[:80]))
    top_modules = profile_detail.get('top_modules', [])[:max_modules]
    lines.append('FLOPs detail | top_modules: {}'.format(top_modules))
    selected = profile_detail.get('selected_modules', {})
    if selected:
        lines.append('FLOPs detail | selected_keywords: {}'.format(selected.get('keywords')))
        lines.append('FLOPs detail | selected_module_flops: {}'.format(format_large_number(selected.get('selected_flops'), 'FLOPs')))
        lines.append('FLOPs detail | selected_module_flops/frame: {}'.format(format_large_number(selected.get('selected_flops_per_frame'), 'FLOPs')))
        ratio = selected.get('selected_ratio')
        if ratio is not None:
            lines.append('FLOPs detail | selected_module_ratio: {:.2f}%'.format(ratio * 100))
        lines.append('FLOPs detail | selected_root_modules: {}'.format(selected.get('selected_root_modules', [])))
        lines.append('FLOPs detail | matched_selected_modules_top{}: {}'.format(
            max_modules, selected.get('matched_modules', [])[:max_modules]))
    return '\n'.join(lines)


def estimate_model_flops(model, img, profile_train=False, fixed_seq_len=-1,
                         debug=False, module_keywords=None,
                         baseline_override=-1.0, baseline_per_frame_override=-1.0, report_file=''):
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        print("fvcore not installed; skipping FLOPs estimation.")
        return None, None, None, {}

    module_keywords = module_keywords or []
    flops_inputs, input_kind = prepare_flops_input(img)
    flops_inputs = _resize_flops_inputs(flops_inputs, fixed_seq_len)
    flops_inputs = _move_flops_inputs_to_model_device(flops_inputs, model)
    input_seq_len = _infer_profile_seq_len(flops_inputs)

    was_training = model.training
    if profile_train:
        model.train()
        profile_ismask = True
    else:
        model.eval()
        profile_ismask = False

    detail = {
        'input_kind': input_kind,
        'seq_len': input_seq_len,
        'profile_train': profile_train,
        'by_operator': {},
        'unsupported_ops': {},
        'uncalled_modules': [],
        'top_modules': [],
        'selected_modules': {},
        'baseline_plus_selected_flops': None,
        'baseline_plus_selected_flops_per_frame': None,
    }

    try:
        with torch.no_grad():
            wrapper = _FlopCountWrapper(model, input_kind=input_kind, ismask=profile_ismask)
            flops_analyzer = FlopCountAnalysis(wrapper, flops_inputs)
            flops_analyzer.unsupported_ops_warnings(debug)
            flops_analyzer.uncalled_modules_warnings(debug)
            total_flops = int(flops_analyzer.total())

            by_operator = dict(flops_analyzer.by_operator())
            unsupported_ops = dict(flops_analyzer.unsupported_ops())
            uncalled_modules = sorted(list(flops_analyzer.uncalled_modules()))
            by_module = dict(flops_analyzer.by_module())
            top_modules = sorted(by_module.items(), key=lambda kv: kv[1], reverse=True)[:50]
            selected = _summarize_selected_module_flops(
                by_module, module_keywords, total_flops, input_seq_len
            ) if module_keywords else {}

            detail.update({
                'by_operator': by_operator,
                'unsupported_ops': unsupported_ops,
                'uncalled_modules': uncalled_modules,
                'top_modules': top_modules,
                'selected_modules': selected,
            })

            # Optional accounting for a parallel KD-MoE branch: if you already
            # measured the baseline FLOPs in a separate run, this reports
            # baseline + selected KD-MoE/prompt branch FLOPs in the same unit.
            if selected:
                if baseline_override is not None and baseline_override > 0:
                    baseline_plus = float(baseline_override) + float(selected.get('selected_flops', 0))
                    detail['baseline_plus_selected_flops'] = baseline_plus
                    detail['baseline_plus_selected_flops_per_frame'] = baseline_plus / input_seq_len if input_seq_len > 0 else None
                elif baseline_per_frame_override is not None and baseline_per_frame_override > 0:
                    selected_pf = float(selected.get('selected_flops_per_frame') or 0)
                    detail['baseline_plus_selected_flops_per_frame'] = float(baseline_per_frame_override) + selected_pf
                    detail['baseline_plus_selected_flops'] = detail['baseline_plus_selected_flops_per_frame'] * input_seq_len if input_seq_len > 0 else None

            if debug:
                print(_format_flops_report(detail))
            if report_file:
                with open(report_file, 'w') as f:
                    f.write(_format_flops_report(detail))
                    f.write('\n')
    except Exception as exc:
        print(f"FLOPs estimation failed: {exc}")
        total_flops = None
    finally:
        model.train(was_training)

    flops_per_frame = total_flops / input_seq_len if total_flops is not None and input_seq_len > 0 else None
    return total_flops, flops_per_frame, input_seq_len, detail


def format_large_number(value, unit):
    if value is None:
        return "N/A"
    if value >= 1e12:
        return f"{value / 1e12:.3f} T{unit}"
    if value >= 1e9:
        return f"{value / 1e9:.3f} G{unit}"
    if value >= 1e6:
        return f"{value / 1e6:.3f} M{unit}"
    if value >= 1e3:
        return f"{value / 1e3:.3f} K{unit}"
    return f"{value:.0f} {unit}"

# %% class weight balancing
def get_weight_balancing(case='cholect50'):
    # 50:   cholecT50, data splits as used in rendezvous paper
    # 50ch: cholecT50, data splits as used in CholecTriplet challenge
    # 45cv: cholecT45, official data splits (cross-val)
    # 50cv: cholecT50, official data splits (cross-val)
    switcher = {
        'cholect50': {
            'tool': [0.08084519, 0.81435289, 0.10459284, 2.55976864, 1.630372490, 1.29528455],
            'verb': [0.31956735, 0.07252306, 0.08111481, 0.81137309, 1.302895320, 2.12264151, 1.54109589, 8.86363636,
                     12.13692946, 0.40462028],
            'target': [0.06246232, 1.00000000, 0.34266478, 0.84750219, 14.80102041, 8.73795181, 1.52845100, 5.74455446,
                       0.285756500, 12.72368421, 0.6250808, 3.85771277, 6.95683453, 0.84923888, 0.40130032]
        },
        'cholect50-challenge': {
            'tool': [0.08495163, 0.88782288, 0.11259564, 2.61948830, 1.784866470, 1.144624170],
            'verb': [0.39862805, 0.06981640, 0.08332925, 0.81876204, 1.415868390, 2.269359150, 1.28428410, 7.35822511,
                     18.67857143, 0.45704490],
            'target': [0.07333818, 0.87139287, 0.42853950, 1.00000000, 17.67281106, 13.94545455, 1.44880997, 6.04889590,
                       0.326188650, 16.82017544, 0.63577586, 6.79964539, 6.19547658, 0.96284208, 0.51559559]
        },
        'cholect45-crossval': {
            1: {
                'tool': [0.08165644, 0.91226868, 0.10674758, 2.85418156, 1.60554885, 1.10640067],
                'verb': [0.37870137, 0.06836869, 0.07931255, 0.84780024, 1.21880342, 2.52836879, 1.30765704, 6.88888889,
                         17.07784431, 0.45241117],
                'target': [0.07149629, 1.0, 0.41013597, 0.90458015, 13.06299213, 12.06545455, 1.5213205, 5.04255319,
                           0.35808332, 45.45205479, 0.67493897, 7.04458599, 9.14049587, 0.97330595, 0.52633249]
            },
            2: {
                'tool': [0.0854156, 0.89535362, 0.10995253, 2.74936869, 1.78264429, 1.13234529],
                'verb': [0.36346863, 0.06771776, 0.07893261, 0.82842725, 1.33892161, 2.13049748, 1.26120359, 5.72674419,
                         19.7, 0.43189126],
                'target': [0.07530655, 0.97961957, 0.4325135, 0.99393438, 15.5387931, 14.5951417, 1.53862569,
                           6.01836394, 0.35184462, 15.81140351, 0.709506, 5.79581994, 8.08295964, 1.0, 0.52689272]
            },
            3: {
                "tool": [0.0915228, 0.89714969, 0.12057004, 2.72128174, 1.94092281, 1.12948557],
                "verb": [0.43636862, 0.07558554, 0.0891017, 0.81820519, 1.53645582, 2.31924198, 1.28565657, 6.49387755,
                         18.28735632, 0.48676763],
                "target": [0.06841828, 0.90980736, 0.38826607, 1.0, 14.3640553, 12.9875, 1.25939394, 5.38341969,
                           0.29060227, 13.67105263, 0.59168565, 6.58985201, 5.72977941, 0.86824513, 0.47682423]

            },
            4: {
                'tool': [0.08222218, 0.85414117, 0.10948695, 2.50868784, 1.63235867, 1.20593318],
                'verb': [0.41154261, 0.0692142, 0.08427214, 0.79895288, 1.33625219, 2.2624166, 1.35343681, 7.63,
                         17.84795322, 0.43970609],
                'target': [0.07536126, 0.85398445, 0.4085784, 0.95464422, 15.90497738, 18.5978836, 1.55875831,
                           5.52672956, 0.33700863, 15.41666667, 0.74755423, 5.4921875, 6.11304348, 1.0, 0.50641118],
            },
            5: {
                'tool': [0.0804654, 0.92271157, 0.10489631, 2.52302243, 1.60074906, 1.09141982],
                'verb': [0.50710436, 0.06590258, 0.07981184, 0.81538866, 1.29267277, 2.20525568, 1.29699248, 7.32311321,
                         25.45081967, 0.46733895],
                'target': [0.07119395, 0.87450495, 0.43043372, 0.86465981, 14.01984127, 23.7114094, 1.47577277,
                           5.81085526, 0.32129865, 22.79354839, 0.63304067, 6.92745098, 5.88833333, 1.0, 0.53175798]
            }
        },
        'cholect50-crossval': {
            1: {
                'tool': [0.0828851, 0.8876, 0.10830995, 2.93907285, 1.63884786, 1.14499484],
                'verb': [0.29628942, 0.07366916, 0.08267971, 0.83155428, 1.25402434, 2.38358209, 1.34938741, 7.56872038,
                         12.98373984, 0.41502079],
                'target': [0.06551745, 1.0, 0.36345711, 0.82434783, 13.06299213, 8.61818182, 1.4017744, 4.62116992,
                           0.32822238, 45.45205479, 0.67343211, 4.13200498, 8.23325062, 0.88527215, 0.43113306],

            },
            2: {
                'tool': [0.08586283, 0.87716737, 0.11068887, 2.84210526, 1.81016949, 1.16283571],
                'verb': [0.30072757, 0.07275414, 0.08350168, 0.80694143, 1.39209979, 2.22754491, 1.31448763, 6.38931298,
                         13.89211618, 0.39397505],
                'target': [0.07056703, 1.0, 0.39451115, 0.91977006, 15.86206897, 9.68421053, 1.44483706, 5.44378698,
                           0.31858714, 16.14035088, 0.7238395, 4.20571429, 7.98264642, 0.91360477, 0.43304307],
            },
            3: {
                'tool': [0.09225068, 0.87856006, 0.12195811, 2.82669323, 1.97710987, 1.1603972],
                'verb': [0.34285159, 0.08049804, 0.0928239, 0.80685714, 1.56125608, 2.23984772, 1.31471136, 7.08835341,
                         12.17241379, 0.43180428],
                'target': [0.06919395, 1.0, 0.37532866, 0.9830703, 15.78801843, 8.99212598, 1.27597765, 5.36990596,
                           0.29177312, 15.02631579, 0.64935557, 5.08308605, 5.86643836, 0.86580743, 0.41908257],
            },
            4: {
                'tool': [0.08247885, 0.83095539, 0.11050268, 2.58193042, 1.64497676, 1.25538881],
                'verb': [0.31890981, 0.07380354, 0.08804592, 0.79094077, 1.35928144, 2.17017208, 1.42947103, 8.34558824,
                         13.19767442, 0.40666428],
                'target': [0.07777646, 0.95894072, 0.41993829, 0.95592153, 17.85972851, 12.49050633, 1.65701092,
                           5.74526929, 0.33763901, 17.31140351, 0.83747083, 3.95490982, 6.57833333, 1.0, 0.47139615],
            },
            5: {
                'tool': [0.07891691, 0.89878025, 0.10267677, 2.53805556, 1.60636428, 1.12691169],
                'verb': [0.36420961, 0.06825313, 0.08060635, 0.80956984, 1.30757221, 2.09375, 1.33625848, 7.9009434,
                         14.1350211, 0.41429631],
                'target': [0.07300329, 0.97128713, 0.42084942, 0.8829883, 15.57142857, 19.42574257, 1.56521739,
                           5.86547085, 0.32732733, 25.31612903, 0.70171674, 4.55220418, 6.13125, 1.0, 0.48528321],
            }
        }
    }
    return switcher.get(case)


def assign_gpu(gpu=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'


def fusion(predicted_list, labels):
    all_out_list = []
    resize_out_list = []
    labels_list = []
    all_out = 0
    len_layer = len(predicted_list)
    weight_list = [1.0 / len_layer for i in range(0, len_layer)]
    for out, w in zip(predicted_list, weight_list):
        resize_out = F.interpolate(out, size=labels.size(0), mode='nearest')
        resize_out_list.append(resize_out)
        if out.size(2) == labels.size(0):
            resize_label = labels
            labels_list.append(resize_label.squeeze().long())
        else:
            resize_label = F.interpolate(labels.float().transpose(0, 1).unsqueeze(0), size=out.size(2), mode='nearest')

            labels_list.append(resize_label.squeeze().transpose(0, 1).long())

        all_out_list.append(out)

    # sss
    return all_out, all_out_list, labels_list

def fairgrad(loss, alpha):
    return (loss ** (1-alpha)) / (1-alpha)
    

def train_loop(dataloader, model, activation, loss_fn_ivt, optimizers, scheduler,
               epoch, writer, wb, alpha,final_eval=False):
    epoch_start = time.time()
    video_time_total = 0.0
    num_videos = 0
    for batch, (img, (y1, y2, y3, y4, y4_soft), weights) in enumerate(dataloader):
        # print(y4.shape)
        if batch > len(dataloader) / FLAGS.train_div:
            break
        sync_cuda()
        video_start = time.time()
        if FLAGS.model == 'clip':
            for i in range(len(img)):
                img[i] = img[i].cuda()
            y1, y2, y3, y4 = y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()
        elif FLAGS.ins_prompt != -1 or FLAGS.target_prompt != -1 or FLAGS.verb_prompt != -1:
            for i in range(len(img)):
                img[i] = img[i].cuda()
            y1, y2, y3, y4 = y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()
        else:
            img, y1, y2, y3, y4, y4_soft = img.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda(), y4_soft.cuda()
        model.train()
        #predicted_list, predicted_list_i, predicted_list_v, predicted_list_t, clip_i, clip_t, cls = model(img, True)
        predicted_list, predicted_list_i, predicted_list_v, predicted_list_t, fpn_feats, fpn_masks, cls = model(img, True)

        # calculate loss
        loss_ivt, loss_i, loss_v, loss_t = [0, 0, 0, 0]
        if FLAGS.model == 'clip':
            cls_loss = loss_cls(cls[0].transpose(1,0), y4[0].float())
            if FLAGS.clip_loss == 'cos':
                loss_clip = cos_loss(clip_i, clip_t)
            elif FLAGS.clip_loss == 'kl':
                loss_clip = kl_loss(clip_i, clip_t)
            elif FLAGS.clip_loss == 'contrastive':
                loss_clip = contrastive_loss(clip_i, clip_t)
        
        if FLAGS.fpn == 'p1':
            logit_ivt = []
            for logits, mask in [predicted_list[0]]:
                l = mask.sum().item()
                logit_ivt.append(logits[:,:,:l])
            logit_i = []
            for logits, mask in [predicted_list_i[0]]:
                l = mask.sum().item()
                logit_i.append(logits[:,:,:l])
            logit_v = []
            for logits, mask in [predicted_list_v[0]]:
                l = mask.sum().item()
                logit_v.append(logits[:,:,:l])
            logit_t = []
            for logits, mask in [predicted_list_t[0]]:
                l = mask.sum().item()
                logit_t.append(logits[:,:,:l])
            feature_fpn = []
            for feat, mask in zip(fpn_feats[0], fpn_masks[0]):
                l = mask.sum().item()
                feature_fpn.append(feat[:,:l])
        else:
            logit_ivt = []
            for logits, mask in predicted_list:
                l = mask.sum().item()
                logit_ivt.append(logits[:,:,:l])
            logit_i = []
            for logits, mask in predicted_list_i:
                l = mask.sum().item()
                logit_i.append(logits[:,:,:l])
            logit_v = []
            for logits, mask in predicted_list_v:
                l = mask.sum().item()
                logit_v.append(logits[:,:,:l])
            logit_t = []
            for logits, mask in predicted_list_t:
                l = mask.sum().item()
                logit_t.append(logits[:,:,:l])


        #ss = y4[0].shape[0]
        # logit_ivt = [p[:,:,:ss] for p in predicted_list]
        # logit_i = [p[:,:,:ss] for p in predicted_list_i]
        # logit_v = [p[:,:,:ss] for p in predicted_list_v]
        # logit_t = [p[:,:,:ss] for p in predicted_list_t]
        #print(y4[0].shape, logit_ivt[0].shape, logit_ivt[1].shape)
        #print(img.shape)

        _, resize_list_ivt, labels_list_ivt = fusion(logit_ivt, y4[0])
        for pd, la, feat in zip(resize_list_ivt, labels_list_ivt, feature_fpn):
            if FLAGS.focal:
                loss_ivt = loss_ivt + loss_fn_ivt(feat, pd[0].transpose(0, 1), la.float(), wb=wb, epoch=epoch)
            elif FLAGS.eql:
                loss_ivt = loss_ivt + loss_fn_ivt(feat, pd[0].transpose(0, 1), la.float(), wb, epoch)
            elif FLAGS.eqlv2:
                loss_ivt = loss_ivt + loss_fn_ivt(feat, pd[0].transpose(0, 1), la.float(), wb=wb, epoch=epoch)
            elif FLAGS.cb:
                loss_ivt = loss_ivt + loss_fn_ivt(feat, pd[0].transpose(0, 1), la.float(), train_weights)
            else:
                loss_ivt = loss_ivt + loss_fn_ivt(feat, pd[0].transpose(0, 1), la.float())
                #loss_ivt = loss_ivt.mean(dim=1) ## mean loss across classes for each frame
                #loss_ivt = (loss_ivt * weights.cuda()).mean() #Apply weights on each frame and compute the final loss

        _, resize_list, labels_list = fusion(logit_i, y1[0])
        for pd, la, pd_ivt in zip(resize_list, labels_list, resize_list_ivt):
            loss_i = loss_i + loss_fn_i(pd[0].transpose(0, 1), la.float())

        _, resize_list, labels_list = fusion(logit_v, y2[0])
        for pd, la, pd_ivt in zip(resize_list, labels_list, resize_list_ivt):
            loss_v = loss_v + loss_fn_v(pd[0].transpose(0, 1), la.float())

        _, resize_list, labels_list = fusion(logit_t, y3[0])
        for pd, la, pd_ivt in zip(resize_list, labels_list, resize_list_ivt):
            loss_t = loss_t + loss_fn_t(pd[0].transpose(0, 1), la.float())

        if FLAGS.loss_type == 'i':
            loss = loss_i
        elif FLAGS.loss_type == 'v':
            loss = loss_v
        elif FLAGS.loss_type == 't':
            loss = loss_t
        elif FLAGS.loss_type == 'ivt':
            loss = fairgrad(loss_ivt, alpha)
        elif FLAGS.loss_type == 'single':
            loss = (loss_i + loss_v + loss_t) / 3
        elif FLAGS.loss_type == 'fairgrad':
            loss_i = fairgrad(loss_i, alpha)
            loss_v = fairgrad(loss_v, alpha)
            loss_t = fairgrad(loss_t, alpha)
            loss_ivt = fairgrad(loss_ivt, alpha)
            # print(loss_i.item(), '--i---', loss_v.item(), '---v--', loss_t.item(), '---t--', loss_ivt.item(), '--ivt---',
            #       epoch)
            loss = FLAGS.beta * (loss_i + loss_v + loss_t) + loss_ivt
        else:
            loss = FLAGS.beta * (loss_i + loss_v + loss_t) + loss_ivt
            # print(loss_i.item(), '--i---', loss_v.item(), '---v--', loss_t.item(), '---t--', loss_ivt.item(), '--ivt---',
            #       epoch)


        total_step = (epoch) * len(dataloader) + batch + 1
        info_loss = {
            'loss_i': loss_i.item(),
            'loss_v': loss_v.item(),
            'loss_t': loss_t.item(),
            'loss_ivt': loss_ivt.item(),
            'loss': loss.item()
        }
        writer.add_scalars('train/loss', info_loss, total_step)

        if FLAGS.model == 'clip':
            wb.log({'train_loss_i': loss_i.item(),
            'train_loss_v': loss_v.item(),
            'train_loss_t': loss_t.item(),
            'train_loss_ivt': loss_ivt.item(),
            'train_loss': loss.item(),
            'clip_loss': loss_clip.item(),
            'cls_loss': cls_loss.item()},
            step=epoch)
        else:
            wb.log({'train_loss_i': loss_i.item(),
            'train_loss_v': loss_v.item(),
            'train_loss_t': loss_t.item(),
            'train_loss_ivt': loss_ivt.item(),
            'train_loss': loss.item()},
            step=epoch)

        for param in model.parameters():
            param.grad = None

        loss.backward()
        if FLAGS.model == 'clip':
            print('using clip loss')
            loss_clip = cls_loss + loss_clip
            loss_clip.backward()

        info_lr = {
            'lr_ivt': optimizers[0].state_dict()['param_groups'][0]['lr'],
        }
        writer.add_scalars('train/lr', info_lr, total_step)
        for opt in optimizers:
            opt.step()

        sync_cuda()
        video_time_total += time.time() - video_start
        num_videos += 1

    for sch in scheduler:
        sch.step()

    avg_video_time = video_time_total / num_videos if num_videos > 0 else 0.0
    epoch_time = time.time() - epoch_start
    print(
        "Epoch {} train profile | videos: {} | avg {:.3f}s/video | epoch {:.2f}s".format(
            epoch, num_videos, avg_video_time, epoch_time),
        file=open(logfile, 'a+'))
    writer.add_scalar('train/sec_per_video', avg_video_time, epoch)
    writer.add_scalar('train/epoch_time_sec', epoch_time, epoch)
    wb.log({
        'train_sec_per_video': avg_video_time,
        'train_epoch_time_sec': epoch_time,
        'train_videos_per_epoch': num_videos,
    }, step=epoch)
    return avg_video_time, epoch_time, num_videos


def test_loop(dataloader, model, activation, writer, wb, final_eval=False, mode='val'):
    global allstep, all_val_step, inference_fps_totals
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    mAP.reset()
    # final_eval = True
    # if final_eval and not set_chlg_eval:
    mAPv.reset()
    mAPt.reset()
    mAPi.reset()
    fps_model_time_total = 0.0
    fps_frames_total = 0
    fps_videos_total = 0
    fps_warmup_videos = max(0, FLAGS.fps_warmup_videos)

    with torch.no_grad():
        for batch, (img, (y1, y2, y3, y4, y4_soft), paths) in enumerate(dataloader):
            if FLAGS.model == 'clip':
                for i in range(len(img)):
                    img[i] = img[i].cuda()
                y1, y2, y3, y4 = y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()
            elif FLAGS.ins_prompt != -1 or FLAGS.target_prompt != -1 or FLAGS.verb_prompt != -1:
                for i in range(len(img)):
                    img[i] = img[i].cuda()
                y1, y2, y3, y4 = y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()
            else:
                img, y1, y2, y3, y4, y4_soft = img.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda(), y4_soft.cuda()

            model.eval()
            profile_fps_this_video = FLAGS.profile_inference_fps and batch >= fps_warmup_videos
            fps_num_frames = infer_num_frames_from_labels(y4)
            if profile_fps_this_video:
                sync_cuda()
                fps_model_start = time.time()
            predicted_list, predicted_list_i, predicted_list_v, predicted_list_t, clip_i, clip_t, cls = model(img,
                                                                                                                  False)
            if profile_fps_this_video:
                sync_cuda()
                fps_model_time_total += time.time() - fps_model_start
                fps_frames_total += fps_num_frames
                fps_videos_total += 1

            # calculate loss
            loss_ivt, loss_i, loss_v, loss_t = [0, 0, 0, 0]
            if FLAGS.model == 'clip':
                cls_loss = loss_cls(cls[0].transpose(1,0), y4[0].float())
                if FLAGS.clip_loss == 'cos':
                    loss_clip = cos_loss(clip_i, clip_t)
                elif FLAGS.clip_loss == 'kl':
                    loss_clip = kl_loss(clip_i, clip_t)
                elif FLAGS.clip_loss == 'contrastive':
                    loss_clip = contrastive_loss(clip_i, clip_t)
                wb.log({'test_clip_ls': loss_clip.item()},step=epoch)
                wb.log({'test_cls_ls': cls_loss.item()},step=epoch)

            # logit_ivt = [p for p in predicted_list]
            # logit_i = [p for p in predicted_list_i]
            # logit_v = [p for p in predicted_list_v]
            # logit_t = [p for p in predicted_list_t]
            # ss = y4[0].shape[0]
            # logit_ivt = [p[:,:,:ss] for p in predicted_list]
            # logit_i = [p[:,:,:ss] for p in predicted_list_i]
            # logit_v = [p[:,:,:ss] for p in predicted_list_v]
            # logit_t = [p[:,:,:ss] for p in predicted_list_t]
            #print(ss)

            if FLAGS.fpn == 'p1':
                logit_ivt = []
                for logits, mask in [predicted_list[0]]:
                    l = mask.sum().item()
                    logit_ivt.append(logits[:,:,:l])
                logit_i = []
                for logits, mask in [predicted_list_i[0]]:
                    l = mask.sum().item()
                    logit_i.append(logits[:,:,:l])
                logit_v = []
                for logits, mask in [predicted_list_v[0]]:
                    l = mask.sum().item()
                    logit_v.append(logits[:,:,:l])
                logit_t = []
                for logits, mask in [predicted_list_t[0]]:
                    l = mask.sum().item()
                    logit_t.append(logits[:,:,:l])
            else:
                logit_ivt = []
                for logits, mask in predicted_list:
                    l = mask.sum().item()
                    logit_ivt.append(logits[:,:,:l])
                logit_i = []
                for logits, mask in predicted_list_i:
                    l = mask.sum().item()
                    logit_i.append(logits[:,:,:l])
                logit_v = []
                for logits, mask in predicted_list_v:
                    l = mask.sum().item()
                    logit_v.append(logits[:,:,:l])
                logit_t = []
                for logits, mask in predicted_list_t:
                    l = mask.sum().item()
                    logit_t.append(logits[:,:,:l])

            mAP.update(y4[0].float().detach().cpu(),
                       activation(logit_ivt[0][0].transpose(0, 1)).detach().cpu())  # Log metrics
            mAPi.update(y1[0].float().detach().cpu(),
                        activation(logit_i[0][0].transpose(0, 1)).detach().cpu())  # Log metrics
            mAPv.update(y2[0].float().detach().cpu(),
                        activation(logit_v[0][0].transpose(0, 1)).detach().cpu())  # Log metrics
            mAPt.update(y3[0].float().detach().cpu(),
                        activation(logit_t[0][0].transpose(0, 1)).detach().cpu())  # Log metrics

    mAP.video_end()
    mAPv.video_end()
    mAPt.video_end()
    mAPi.video_end()

    if FLAGS.profile_inference_fps:
        inference_fps = fps_frames_total / fps_model_time_total if fps_model_time_total > 0 else 0.0
        sec_per_video = fps_model_time_total / fps_videos_total if fps_videos_total > 0 else 0.0
        fps_msg = (
            "{} inference profile | videos: {} | frames: {} | warmup videos: {} | "
            "model time: {:.4f}s | FPS: {:.3f} | sec/video: {:.4f}".format(
                mode, fps_videos_total, fps_frames_total, fps_warmup_videos,
                fps_model_time_total, inference_fps, sec_per_video)
        )
        stats = inference_fps_totals.setdefault(mode, {'time': 0.0, 'frames': 0, 'videos': 0})
        stats['time'] += fps_model_time_total
        stats['frames'] += fps_frames_total
        stats['videos'] += fps_videos_total
        cumulative_fps = stats['frames'] / stats['time'] if stats['time'] > 0 else 0.0
        cumulative_sec_per_video = stats['time'] / stats['videos'] if stats['videos'] > 0 else 0.0
        cumulative_msg = (
            "{} cumulative inference profile | videos: {} | frames: {} | "
            "model time: {:.4f}s | FPS: {:.3f} | sec/video: {:.4f}".format(
                mode, stats['videos'], stats['frames'], stats['time'],
                cumulative_fps, cumulative_sec_per_video)
        )
        print(fps_msg)
        print(cumulative_msg)
        print(fps_msg, file=open(logfile, 'a+'))
        print(cumulative_msg, file=open(logfile, 'a+'))
        writer.add_scalar('{}/inference_fps'.format(mode), inference_fps, all_val_step)
        writer.add_scalar('{}/inference_sec_per_video'.format(mode), sec_per_video, all_val_step)
        writer.add_scalar('{}/cumulative_inference_fps'.format(mode), cumulative_fps, all_val_step)
        writer.add_scalar('{}/cumulative_inference_sec_per_video'.format(mode), cumulative_sec_per_video, all_val_step)
        wb.log({
            '{}_inference_fps'.format(mode): inference_fps,
            '{}_inference_sec_per_video'.format(mode): sec_per_video,
            '{}_inference_frames'.format(mode): fps_frames_total,
            '{}_inference_videos'.format(mode): fps_videos_total,
            '{}_cumulative_inference_fps'.format(mode): cumulative_fps,
            '{}_cumulative_inference_sec_per_video'.format(mode): cumulative_sec_per_video,
            '{}_cumulative_inference_frames'.format(mode): stats['frames'],
            '{}_cumulative_inference_videos'.format(mode): stats['videos'],
        }, step=globals().get('epoch', all_val_step))


def weight_mgt(score, epoch):
    # hyperparameter selection based on validation set
    global benchmark
    torch.save(model.state_dict(), ckpt_path_epoch + '_latest.pth')
    if score > benchmark.item():
        torch.save(model.state_dict(), ckpt_path)
        benchmark = score
        print(f'>>> Saving checkpoint for epoch {epoch + 1} at {ckpt_path}, time {time.ctime()} ',
              file=open(logfile, 'a+'))
        return "increased"
    else:
        return "decreased"


# %% assign device and set debugger options
assign_gpu(gpu=gpu)
np.seterr(divide='ignore', invalid='ignore')
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)

# Or constant weights from average of the random sampling of the dataset: we found this to produce better result.

tool_weight = [0.93487068, 0.94234964, 0.93487068, 1.18448115, 1.02368339, 0.97974447]
verb_weight = [0.60002400, 0.60002400, 0.60002400, 0.61682467, 0.67082683, 0.80163207, 0.70562823, 2.11208448,
               2.69230769, 0.60062402]
target_weight = [0.49752894, 0.52041527, 0.49752894, 0.51394739, 2.71899565, 1.75577963, 0.58509403, 1.25228034,
                 0.49752894, 2.42993134, 0.49802647, 0.87266576, 1.36074165, 0.50150917, 0.49802647]


def load_model(model, dir):
    print('lode_dir', dir)
    pretrained_dict = torch.load(dir)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model.state_dict().update(pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

# import network_res

# model_base = network_res.VideoNas(basename='resnet18').cuda()
# # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
# pytorch_total_params = sum(p.numel() for p in model_base.parameters())
# pytorch_train_params = sum(p.numel() for p in model_base.parameters() if p.requires_grad)
# if FLAGS.pretrain_dir:
#     model_base = load_model(model_base, FLAGS.pretrain_dir)
# print('BackBone: Total params: %.2fM' % (sum(p.numel() for p in model_base.parameters()) / 1000000.0))
# for params in model_base.parameters():
#     params.requires_grad = False
# model_base.eval()
# %% model
num_stages = 3  # refinement stages
num_layers = 12  # layers of prediction tcn e
num_f_maps = 512
dim = FLAGS.input_dim  # input dim
num_classes = 100
# print(args.num_classes)
num_layers_PG = FLAGS.num_layers_PG
num_layers_R = FLAGS.num_layers_R
num_R = FLAGS.num_R
if FLAGS.model == 'mstcn':
    model = network.VideoNas(FLAGS, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes).cuda()
elif FLAGS.model == 'actionformer':
    model = network.VideoTrans(FLAGS, num_f_maps, dim).cuda()
elif FLAGS.model == 'clip':
    model = network.VideoTrans_clip(FLAGS, num_f_maps, dim).cuda()

pytorch_train_params, pytorch_total_params, pytorch_frozen_params = count_model_params(model)
print('Model params | trainable: {:.2f}M | total: {:.2f}M | frozen: {:.2f}M'.format(
    pytorch_train_params / 1e6, pytorch_total_params / 1e6, pytorch_frozen_params / 1e6))
model_flops = None
model_flops_per_frame = None
model_flops_seq_len = None
model_flops_detail = {}
kdmoe_flops = None
kdmoe_flops_per_frame = None
baseline_plus_kdmoe_flops = None
baseline_plus_kdmoe_flops_per_frame = None
# %% performance tracker for hp tuning
benchmark = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
print("Model built ...")
allstep = 0
all_val_step = 0
inference_fps_totals = {}

# %% optimizer and lr scheduler
wp_lr = [lr / power for lr in learning_rates]
# warmups[2] = int(epochs * 0.25)
optimizer_ivt = torch.optim.SGD(model.parameters(), lr=wp_lr[2], weight_decay=weight_decay)
scheduler_ivta = torch.optim.lr_scheduler.LinearLR(optimizer_ivt, start_factor=power, total_iters=warmups[2])
scheduler_ivtb = torch.optim.lr_scheduler.ExponentialLR(optimizer_ivt, gamma=decay_rate)
scheduler_ivt = torch.optim.lr_scheduler.SequentialLR(optimizer_ivt, schedulers=[scheduler_ivta, scheduler_ivtb],
                                                      milestones=[warmups[2] + 1])

lr_schedulers = [scheduler_ivt]
optimizers = [optimizer_ivt]

print("Model's weight loaded ...")
# (32, 3, 256, 448)
# (32, 6)
# (32, 10)
# (32, 15)
# (32, 100)
# %% data loading : variant and split selection (Note: original paper used different augumentation per epoch)

dataset = dataloader.CholecT50(
    args=FLAGS,
    dataset_dir=data_dir,
    dataset_variant=dataset_variant,
    test_fold=kfold,
    augmentation_list=data_augmentations,
    model=None
)

# build dataset
train_dataset, val_dataset, test_dataset = dataset.build()
train_weights, test_weights = dataset.cls_weights()

# %% Loss
activation = nn.Sigmoid()
loss_fn_i = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(tool_weight).cuda())
loss_fn_v = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(verb_weight).cuda())
loss_fn_t = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(target_weight).cuda())

if FLAGS.focal is True:
    loss_fn_ivt = MultiLabelFocalLoss(alpha=alpha, gamma=FLAGS.gamma)
    print('using focal loss')
elif FLAGS.eql is True:
    loss_fn_ivt = EQL(
        train_weights,
        cgl_split_source=FLAGS.cgl_split_source,
        cgl_split_mode=FLAGS.cgl_split_mode,
        cgl_head_threshold=FLAGS.cgl_head_threshold,
        cgl_tail_threshold=FLAGS.cgl_tail_threshold,
        cgl_head_percent=FLAGS.cgl_head_percent,
        cgl_tail_percent=FLAGS.cgl_tail_percent,
        cgl_head_ratio=FLAGS.cgl_head_ratio,
        cgl_tail_ratio=FLAGS.cgl_tail_ratio,
        cgl_full_stats_path=FLAGS.cgl_full_stats_path or None,
    )
    ivt_head = loss_fn_ivt.ivt_head
    ivt_medium = loss_fn_ivt.ivt_medium
    ivt_tail = loss_fn_ivt.ivt_tail
    print('using EQL loss')
elif FLAGS.eqlv2 is True:
    loss_fn_ivt = EQLv2()
    print('using EQL2 loss')
elif FLAGS.cb is True:
    loss_fn_ivt = ClassBalancedBCELoss()
    print('using class balanced loss')
else:
    loss_fn_ivt = nn.BCEWithLogitsLoss()
    print('using bce loss')
loss_cls = nn.BCEWithLogitsLoss()

# %% evaluation metrics
mAP = ivtmetrics.Recognition(100)
mAP.reset_global()
mAPi = ivtmetrics.Recognition(6)
mAPv = ivtmetrics.Recognition(10)
mAPt = ivtmetrics.Recognition(15)
mAPi.reset_global()
mAPv.reset_global()
mAPt.reset_global()
print("Metrics built ...")

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, prefetch_factor=5 * 1,
                              num_workers=1, pin_memory=FLAGS.pin_memory, persistent_workers=True, drop_last=False)
val_dataloaders = []
for video_dataset in val_dataset:
    val_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False,
                                prefetch_factor=5 * 1,
                                num_workers=1, pin_memory=FLAGS.pin_memory, persistent_workers=True, drop_last=False)
    val_dataloaders.append(val_dataloader)
# test data set is built per video, so load differently
test_dataloaders = []
for video_dataset in test_dataset:
    test_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False,
                                 prefetch_factor=5 * 1,
                                 num_workers=1, pin_memory=FLAGS.pin_memory, persistent_workers=True, drop_last=False)
    test_dataloaders.append(test_dataloader)
#test_train_dataloaders = []
# for video_dataset in test_train_dataset:
#     test_train_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False,
#                                        prefetch_factor=5 * 1,
#                                        num_workers=1, pin_memory=FLAGS.pin_memory, persistent_workers=True, drop_last=False)
#     test_train_dataloaders.append(test_train_dataloader)
print("Dataset loaded ...")

try:
    _profile_batch = next(iter(train_dataloader))
    _profile_img = _profile_batch[0]
    model_flops, model_flops_per_frame, model_flops_seq_len, model_flops_detail = estimate_model_flops(
        model,
        _profile_img,
        profile_train=FLAGS.flops_profile_train,
        fixed_seq_len=FLAGS.flops_fixed_seq_len,
        debug=FLAGS.flops_debug,
        module_keywords=FLAGS.flops_module_keywords,
        baseline_override=FLAGS.flops_baseline_override,
        baseline_per_frame_override=FLAGS.flops_baseline_per_frame_override,
        report_file=FLAGS.flops_report_file,
    )
    selected_detail = model_flops_detail.get('selected_modules', {}) if model_flops_detail else {}
    kdmoe_flops = selected_detail.get('selected_flops')
    kdmoe_flops_per_frame = selected_detail.get('selected_flops_per_frame')
    baseline_plus_kdmoe_flops = model_flops_detail.get('baseline_plus_selected_flops') if model_flops_detail else None
    baseline_plus_kdmoe_flops_per_frame = model_flops_detail.get('baseline_plus_selected_flops_per_frame') if model_flops_detail else None
except StopIteration:
    print("Train dataloader is empty; skipping FLOPs estimation.")

profile_msg = (
    "Model profile | trainable params: {} | total params: {} | frozen params: {} | "
    "FLOPs@seq{}: {} | FLOPs/frame: {} | selected KD-MoE/prompt FLOPs: {} | selected/frame: {}".format(
        format_large_number(pytorch_train_params, "params"),
        format_large_number(pytorch_total_params, "params"),
        format_large_number(pytorch_frozen_params, "params"),
        model_flops_seq_len if model_flops_seq_len is not None else "N/A",
        format_large_number(model_flops, "FLOPs"),
        format_large_number(model_flops_per_frame, "FLOPs"),
        format_large_number(kdmoe_flops, "FLOPs"),
        format_large_number(kdmoe_flops_per_frame, "FLOPs"),
    )
)
print(profile_msg)
print(profile_msg, file=open(logfile, 'a+'))

if baseline_plus_kdmoe_flops is not None:
    baseline_plus_msg = "Model profile | baseline_override + selected KD-MoE/prompt FLOPs@seq{}: {} | per frame: {}".format(
        model_flops_seq_len if model_flops_seq_len is not None else "N/A",
        format_large_number(baseline_plus_kdmoe_flops, "FLOPs"),
        format_large_number(baseline_plus_kdmoe_flops_per_frame, "FLOPs"),
    )
    print(baseline_plus_msg)
    print(baseline_plus_msg, file=open(logfile, 'a+'))

if FLAGS.flops_debug and model_flops_detail:
    detail_msg = _format_flops_report(model_flops_detail)
    print(detail_msg, file=open(logfile, 'a+'))

# %% log config
header1 = "** Run: {} | Framework: PyTorch | Method: {} | Version: {} | Data: CholecT50 | Batch: {} **".format(
    os.path.basename(__file__), modelname, version, batch_size)
header2 = "** Time: {} | Start: {}-epoch  {}-steps | Init CKPT: {} | Save CKPT: {} **".format(time.ctime(), 0, 0,
                                                                                              resume_ckpt, ckpt_path)
header3 = "** LR Config: Init: {} | Peak: {} | Warmup Epoch: {} | Rise: {} | Decay {} | train params {} | all params {} **".format(
    [float(f"{op.state_dict()['param_groups'][0]['lr']:.6f}") for op in optimizers], [float(f'{v:.6f}') for v in wp_lr],
    warmups, power,
    decay_rate, pytorch_train_params, pytorch_total_params)
maxlen = max(len(header1), len(header2), len(header3))
# maxlen = max(len(header1), len(header2))
header1 = "{}{}{}".format('*' * ((maxlen - len(header1)) // 2 + 1), header1, '*' * ((maxlen - len(header1)) // 2 + 1))
header2 = "{}{}{}".format('*' * ((maxlen - len(header2)) // 2 + 1), header2, '*' * ((maxlen - len(header2)) // 2 + 1))
header3 = "{}{}{}".format('*' * ((maxlen - len(header3)) // 2 + 1), header3, '*' * ((maxlen - len(header3)) // 2 + 1))
maxlen = max(len(header1), len(header2), len(header3))
# maxlen = max(len(header1), len(header2))
writer = SummaryWriter(model_dir)

print("\n\n\n{}\n{}\n{}\n{}\n{}".format("*" * maxlen, header1, header2, header3, "*" * maxlen),
      file=open(logfile, 'a+'))
print("Experiment started ...\n   logging outputs to: ", logfile)

wandb.init(
        # set the wandb project where this run will be logged
        project="terl",

        # track hyperparameters and run metadata
        config={
        #"learning_rate": 0.02,
        "architecture": 'stage2_actionformer',
        "dataset": "cholect45",
        #"epochs": params.epochs,
        #"lr": params.lr,
        "kfold": 1,
        "bs": 1,
        "trainable_params": pytorch_train_params,
        "total_params": pytorch_total_params,
        "frozen_params": pytorch_frozen_params,
        "model_flops": model_flops,
        "model_flops_per_frame": model_flops_per_frame,
        "model_flops_seq_len": model_flops_seq_len,
        "kdmoe_selected_flops": kdmoe_flops,
        "kdmoe_selected_flops_per_frame": kdmoe_flops_per_frame,
        "baseline_plus_kdmoe_flops": baseline_plus_kdmoe_flops,
        "baseline_plus_kdmoe_flops_per_frame": baseline_plus_kdmoe_flops_per_frame,
        #"name": params.arch+"_"+str(params.lr)+"_"+str(params.bs)
        }
    )
# %% run
if is_train:
    for epoch in range(0, epochs):
        try:
            # Train
            print("Traning | lr: {} | epoch {}".format([op.state_dict()['param_groups'][0]['lr'] for op in optimizers],
                                                       epoch), end=" | ",
                  file=open(logfile, 'a+'))
            train_loop(train_dataloader, model, activation, loss_fn_ivt, optimizers,
                       lr_schedulers, epoch, writer, wandb, alpha)

            # val
            if epoch % val_interval == 0:
                start = time.time()
                mAP.reset_global()
                mAPi.reset_global()
                mAPv.reset_global()
                mAPt.reset_global()
                print("Evaluating @ epoch: ", epoch, file=open(logfile, 'a+'))
                for i, val_dataloader in enumerate(val_dataloaders):
                    test_loop(val_dataloader, model, activation, writer, wandb, final_eval=False)
                if FLAGS.loss_type == 'i':
                    behaviour = weight_mgt(mAPi.compute_video_AP()['mAP'], epoch=epoch)
                elif FLAGS.loss_type == 'v':
                    behaviour = weight_mgt(mAPv.compute_video_AP()['mAP'], epoch=epoch)
                elif FLAGS.loss_type == 't':
                    behaviour = weight_mgt(mAPt.compute_video_AP()['mAP'], epoch=epoch)
                elif FLAGS.loss_type == 'single':
                    mean_mAP = (mAPi.compute_video_AP()['mAP'] + mAPv.compute_video_AP()['mAP'] +
                                mAPt.compute_video_AP()['mAP']) / 3
                    behaviour = weight_mgt(mean_mAP, epoch=epoch)
                else:
                    behaviour = weight_mgt(mAP.compute_video_AP()['mAP'], epoch=epoch)
                if FLAGS.loss_type in ['i', 'v', 't', 'single']:
                    mAP_i = mAPi.compute_video_AP(ignore_null=set_chlg_eval)
                    mAP_v = mAPv.compute_video_AP(ignore_null=set_chlg_eval)
                    mAP_t = mAPt.compute_video_AP(ignore_null=set_chlg_eval)
                else:
                    mAP_i = mAP.compute_video_AP('i', ignore_null=set_chlg_eval)
                    mAP_v = mAP.compute_video_AP('v', ignore_null=set_chlg_eval)
                    mAP_t = mAP.compute_video_AP('t', ignore_null=set_chlg_eval)

                mAP_iv = mAP.compute_video_AP('iv', ignore_null=set_chlg_eval)
                mAP_it = mAP.compute_video_AP('it', ignore_null=set_chlg_eval)
                mAP_ivt = mAP.compute_video_AP('ivt', ignore_null=set_chlg_eval)
                info_mAP = {
                    'mAP_i': mAP_i["mAP"],
                    'mAP_v': mAP_v["mAP"],
                    'mAP_t': mAP_t["mAP"],
                    'mAP_iv': mAP_iv["mAP"],
                    'mAP_it': mAP_it["mAP"],
                    'mAP_ivt': mAP_ivt["mAP"],
                }

                wandb.log({
                    'val_mAP_i': mAP_i["mAP"],
                    'val_mAP_v': mAP_v["mAP"],
                    'val_mAP_t': mAP_t["mAP"],
                    'val_mAP_iv': mAP_iv["mAP"],
                    'val_mAP_it': mAP_it["mAP"],
                    'val_mAP_ivt': mAP_ivt["mAP"],
                },
                step=epoch)
                writer.add_scalars('val/mAP', info_mAP, epoch)
                print(
                    "\t\t\t\t\t\t\t video-wise | eta {:.2f} secs | mAP => ivt: [{:.5f}] ".format((time.time() - start),
                                                                                                 mAP.compute_video_AP(
                                                                                                     'ivt',
                                                                                                     ignore_null=set_chlg_eval)[
                                                                                                     'mAP']),
                    file=open(logfile, 'a+'))

                start = time.time()
                mAP.reset_global()
                mAPi.reset_global()
                mAPv.reset_global()
                mAPt.reset_global()
                print("Test @ epoch: ", epoch, file=open(logfile, 'a+'))
                if FLAGS.profile_inference_fps:
                    inference_fps_totals['test'] = {'time': 0.0, 'frames': 0, 'videos': 0}
                for i, test_dataloader in enumerate(test_dataloaders):
                    test_loop(test_dataloader, model, activation, writer, wandb, final_eval=False, mode='test')
                behaviour = weight_mgt(mAP.compute_video_AP()['mAP'], epoch=epoch)
                if FLAGS.loss_type in ['i', 'v', 't', 'single']:
                    mAP_i = mAPi.compute_video_AP(ignore_null=set_chlg_eval)
                    mAP_v = mAPv.compute_video_AP(ignore_null=set_chlg_eval)
                    mAP_t = mAPt.compute_video_AP(ignore_null=set_chlg_eval)
                else:
                    mAP_i = mAP.compute_video_AP('i', ignore_null=set_chlg_eval)
                    mAP_v = mAP.compute_video_AP('v', ignore_null=set_chlg_eval)
                    mAP_t = mAP.compute_video_AP('t', ignore_null=set_chlg_eval)

                mAP_iv = mAP.compute_video_AP('iv', ignore_null=set_chlg_eval)
                mAP_it = mAP.compute_video_AP('it', ignore_null=set_chlg_eval)
                mAP_ivt = mAP.compute_video_AP('ivt', ignore_null=set_chlg_eval)
                print('----ivt----' )
                print(mAP_ivt["AP"])
                print(mAP_ivt["mAP"])
                MAP_i_head = np.nanmean([mAP_i["AP"][idx] for idx in i_head])
                MAP_v_head = np.nanmean([mAP_v["AP"][idx] for idx in v_head])
                MAP_t_head = np.nanmean([mAP_t["AP"][idx] for idx in t_head])
                MAP_ivt_head = mean_ap_by_indices(mAP_ivt["AP"], ivt_head)
                MAP_i_tail = np.nanmean([mAP_i["AP"][idx] for idx in i_tail])
                MAP_v_tail = np.nanmean([mAP_v["AP"][idx] for idx in v_tail])
                MAP_t_tail = np.nanmean([mAP_t["AP"][idx] for idx in t_tail])
                MAP_ivt_tail = mean_ap_by_indices(mAP_ivt["AP"], ivt_tail)
                MAP_i_medium = np.nanmean([mAP_i["AP"][idx] for idx in i_medium])
                MAP_v_medium = np.nanmean([mAP_v["AP"][idx] for idx in v_medium])
                MAP_t_medium = np.nanmean([mAP_t["AP"][idx] for idx in t_medium])
                MAP_ivt_medium = mean_ap_by_indices(mAP_ivt["AP"], ivt_medium)
                print('----i----' )
                print(mAP_i["AP"])
                print(mAP_i["mAP"])
                print('----v----' )
                print(mAP_v["AP"])
                print(mAP_v["mAP"])
                print('----t----' )
                print(mAP_t["AP"])
                print(mAP_t["mAP"])
                info_mAP = {
                    'mAP_i': mAP_i["mAP"],
                    'mAP_v': mAP_v["mAP"],
                    'mAP_t': mAP_t["mAP"],
                    'mAP_iv': mAP_iv["mAP"],
                    'mAP_it': mAP_it["mAP"],
                    'mAP_ivt': mAP_ivt["mAP"],
                }
                wandb.log({
                    'test_mAP_i': mAP_i["mAP"],
                    'test_mAP_v': mAP_v["mAP"],
                    'test_mAP_t': mAP_t["mAP"],
                    'test_mAP_iv': mAP_iv["mAP"],
                    'test_mAP_it': mAP_it["mAP"],
                    'test_mAP_ivt': mAP_ivt["mAP"],
                },
                step=epoch)
                writer.add_scalars('test/mAP', info_mAP, epoch)
                print(
                    "\t\t\t\t\t\t\t video-wise | eta {:.2f} secs | mAP => ivt: [{:.5f}] | AP {}".format((time.time() - start),
                                                                                                 mAP_ivt["mAP"], mAP_ivt["AP"]),
                    file=open(logfile, 'a+'))
        except KeyboardInterrupt:
            print(f'>> Process cancelled by user at {time.ctime()}, ...', file=open(logfile, 'a+'))
            sys.exit(1)
    test_ckpt = ckpt_path

# %% eval
if is_test:
    print("Test weight: ", test_ckpt)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    if is_train:
        tags = ['latest', 'best']
    else:
        tags = ['best']
    for tag in tags:
        print('========', tag, '==============')
        if tag == 'best':
            model.load_state_dict(torch.load(test_ckpt))

        mAP.reset_global()
        mAPi.reset_global()
        mAPv.reset_global()
        mAPt.reset_global()
        allstep = 0
        if FLAGS.profile_inference_fps:
            inference_fps_totals['test'] = {'time': 0.0, 'frames': 0, 'videos': 0}
        for test_dataloader in test_dataloaders:
            test_loop(test_dataloader, model, activation, writer, wandb, final_eval=True, mode='test')

        mAPs = {'ivt': mAP, 'i': mAPi, 'v': mAPv, 't': mAPt}
        import pickle

        f = open(model_dir + '/mAPs_' + tag + '_k' + str(kfold) + '.pckl', 'wb')
        pickle.dump(mAPs, f)
        f.close()
        if FLAGS.loss_type in ['i', 'v', 't']:
            mAP_i = mAPi.compute_video_AP(ignore_null=set_chlg_eval)
            mAP_v = mAPv.compute_video_AP(ignore_null=set_chlg_eval)
            mAP_t = mAPt.compute_video_AP(ignore_null=set_chlg_eval)
        else:
            mAP_i = mAP.compute_video_AP('i', ignore_null=set_chlg_eval)
            mAP_v = mAP.compute_video_AP('v', ignore_null=set_chlg_eval)
            mAP_t = mAP.compute_video_AP('t', ignore_null=set_chlg_eval)

        mAP_iv = mAP.compute_video_AP('iv', ignore_null=set_chlg_eval)
        mAP_it = mAP.compute_video_AP('it', ignore_null=set_chlg_eval)
        mAP_ivt = mAP.compute_video_AP('ivt', ignore_null=set_chlg_eval)
        print('-' * 50, file=open(logfile, 'a+'))
        print(tag, file=open(logfile, 'a+'))
        print('Test Results\nPer-category AP: ', file=open(logfile, 'a+'))
        print(f'I   : {mAP_i["AP"]}', file=open(logfile, 'a+'))
        print(f'V   : {mAP_v["AP"]}', file=open(logfile, 'a+'))
        print(f'T   : {mAP_t["AP"]}', file=open(logfile, 'a+'))
        print(f'IV  : {mAP_iv["AP"]}', file=open(logfile, 'a+'))
        print(f'IT  : {mAP_it["AP"]}', file=open(logfile, 'a+'))
        print(f'IVT : {mAP_ivt["AP"]}', file=open(logfile, 'a+'))
        print('-' * 50, file=open(logfile, 'a+'))
        print(f'Mean AP:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {mAP_i["mAP"]:.4f} | {mAP_v["mAP"]:.4f} | {mAP_t["mAP"]:.4f} | {mAP_iv["mAP"]:.4f} | {mAP_it["mAP"]:.4f} | {mAP_ivt["mAP"]:.4f} ',
            file=open(logfile, 'a+'))
        top5 = [mAP.topK(5, 'i'), mAP.topK(5, 'v'), mAP.topK(5, 't'), mAP.topK(5, 'iv'), mAP.topK(5, 'it'),
                mAP.topK(5, 'ivt')]
        top10 = [mAP.topK(10, 'i'), mAP.topK(10, 'v'), mAP.topK(10, 't'), mAP.topK(10, 'iv'), mAP.topK(10, 'it'),
                 mAP.topK(10, 'ivt')]
        top20 = [mAP.topK(20, 'i'), mAP.topK(20, 'v'), mAP.topK(20, 't'), mAP.topK(20, 'iv'), mAP.topK(20, 'it'),
                 mAP.topK(20, 'ivt')]
        print(f'top 5:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {top5[0]:.4f} | {top5[1]:.4f} | {top5[2]:.4f} | {top5[3]:.4f} | {top5[4]:.4f} | {top5[5]:.4f} ',
            file=open(logfile, 'a+'))
        print(f'top 10:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {top10[0]:.4f} | {top10[1]:.4f} | {top10[2]:.4f} | {top10[3]:.4f} | {top10[4]:.4f} | {top10[5]:.4f} ',
            file=open(logfile, 'a+'))
        print(f'top 20:  I  |  V  |  T  |  IV  |  IT  |  IVT ', file=open(logfile, 'a+'))
        print(
            f':::::: : {top20[0]:.4f} | {top20[1]:.4f} | {top20[2]:.4f} | {top20[3]:.4f} | {top20[4]:.4f} | {top20[5]:.4f} ',
            file=open(logfile, 'a+'))
        print('=' * 50, file=open(logfile, 'a+'))

# %% End
print("All done!\nShutting done...\nIt is what it is ...\nC'est finis! {}".format("-" * maxlen),
      file=open(logfile, 'a+'))
