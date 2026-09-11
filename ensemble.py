#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %% import libraries
from ipaddress import v4_int_to_packed
from itertools import count
from math import log
import os
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
parser.add_argument('--data_dir', type=str, default='/home/student/Documents/data/CholecT45/',
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
parser.add_argument('--coe', default=0.5, type=float)
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
is_test = True
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
activation = nn.Sigmoid()
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
    

def test_loop(dataloader_T, dataloader_B, model_T, model_B, activation, writer, wb, final_eval=False, mode='val'):
    global allstep, all_val_step
    coe = FLAGS.coe
    print(f"coe: {coe}")
    mAP.reset()
    mAPv.reset()
    mAPt.reset()
    mAPi.reset()

    with torch.no_grad():   
        for (img_T, (y1, y2, y3, y4,_), _), (img_B, (_, _, _, _,_), _) in zip(dataloader_T, dataloader_B):

            model_T.eval()
            model_B.eval()
            predicted_list_T, predicted_list_i_T, predicted_list_v_T, predicted_list_t_T, _, _, _ = model_T(img_T, False)
            predicted_list_B, predicted_list_i_B, predicted_list_v_B, predicted_list_t_B, _, _, _ = model_B(img_B, False)

            if FLAGS.fpn == 'p1':
                logit_ivt_T = []
                logit_ivt_B = []
                logit_ivt = []
                for logits, mask in [predicted_list_T[0]]:
                    l = mask.sum().item()
                    logit_ivt_T.append(logits[:,:,:l])
                for logits, mask in [predicted_list_B[0]]:
                    l = mask.sum().item()
                    logit_ivt_B.append(logits[:,:,:l])
                logit_ivt.append(logit_ivt_T[0] * coe + logit_ivt_B[0] * (1 - coe))
                logit_i_T = []
                logit_i_B = []
                logit_i = []
                for logits, mask in [predicted_list_i_T[0]]:
                    l = mask.sum().item()
                    logit_i_T.append(logits[:,:,:l])
                for logits, mask in [predicted_list_i_B[0]]:
                    l = mask.sum().item()
                    logit_i_B.append(logits[:,:,:l])
                logit_i.append(logit_i_T[0] * coe + logit_i_B[0] * (1 - coe))
                logit_v_T = []
                logit_v_B = []
                logit_v = []
                for logits, mask in [predicted_list_v_T[0]]:
                    l = mask.sum().item()
                    logit_v_T.append(logits[:,:,:l])
                for logits, mask in [predicted_list_v_B[0]]:
                    l = mask.sum().item()
                    logit_v_B.append(logits[:,:,:l])
                logit_v.append(logit_v_T[0] * coe + logit_v_B[0] * (1 - coe))
                logit_t_T = []
                logit_t_B = []
                logit_t = []
                for logits, mask in [predicted_list_t_T[0]]:
                    l = mask.sum().item()
                    logit_t_T.append(logits[:,:,:l])
                for logits, mask in [predicted_list_t_B[0]]:
                    l = mask.sum().item()
                    logit_t_B.append(logits[:,:,:l])
                logit_t.append(logit_t_T[0] * coe + logit_t_B[0] * (1 - coe))

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
    model_T = network.VideoTrans(FLAGS, num_f_maps, 768).cuda()
    model_B = network.VideoTrans(FLAGS, num_f_maps, 1024).cuda()
elif FLAGS.model == 'clip':
    model = network.VideoTrans_clip(FLAGS, num_f_maps, dim).cuda()

pytorch_total_params = sum(p.numel() for p in model_T.parameters())
pytorch_train_params = sum(p.numel() for p in model_T.parameters() if p.requires_grad)
print('BackBone: Total params: %.2fM' % (sum(p.numel() for p in model_T.parameters()) / 1000000.0))

# %% performance tracker for hp tuning
benchmark = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
print("Model built ...")
allstep = 0
all_val_step = 0


print("Model's weight loaded ...")


dataset = dataloader.CholecT50(
    args=FLAGS,
    dataset_dir=data_dir,
    dataset_variant=dataset_variant,
    test_fold=kfold,
    augmentation_list=data_augmentations,
    model=None
)


# build dataset
test_dataset_T = dataset.build_T()
test_dataset_B = dataset.build_B()


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

test_dataloaders_T = []
for video_dataset in test_dataset_T:
    test_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False,
                                 prefetch_factor=5 * 1,
                                 num_workers=1, pin_memory=True, persistent_workers=True, drop_last=False)
    test_dataloaders_T.append(test_dataloader)

test_dataloaders_B = []
for video_dataset in test_dataset_B:
    test_dataloader = DataLoader(video_dataset, batch_size=1, shuffle=False,
                                 prefetch_factor=5 * 1,
                                 num_workers=1, pin_memory=True, persistent_workers=True, drop_last=False)
    test_dataloaders_B.append(test_dataloader)
print("Dataset loaded ...")

# maxlen = max(len(header1), len(header2))
writer = SummaryWriter(model_dir)


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
        #"name": params.arch+"_"+str(params.lr)+"_"+str(params.bs)
        }
    )
# %% run


# %% eval
if is_test:

    tags = ['best']

    for tag in tags:
        print('========', tag, '==============')
        if tag == 'latest':
            model_T.load_state_dict(torch.load("/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/__checkpoint__/run_t45Tclip_"+str(FLAGS.kfold)+"/actionformer_l8_cholectcholect45-crossval_k"+str(FLAGS.kfold)+"_batchnorm_lowres_latest.pth"), strict=False)
            model_B.load_state_dict(torch.load("/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/__checkpoint__/run_t45Bclip_"+str(FLAGS.kfold)+"/actionformer_l8_cholectcholect45-crossval_k"+str(FLAGS.kfold)+"_batchnorm_lowres_latest.pth"), strict=False)
        elif tag == 'best':
            model_T.load_state_dict(torch.load("/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/__checkpoint__/run_t45Tclip_"+str(FLAGS.kfold)+"/actionformer_l8_cholectcholect45-crossval_k"+str(FLAGS.kfold)+"_batchnorm_lowres.pth"), strict=False)
            model_B.load_state_dict(torch.load("/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/__checkpoint__/run_t45Bclip_"+str(FLAGS.kfold)+"/actionformer_l8_cholectcholect45-crossval_k"+str(FLAGS.kfold)+"_batchnorm_lowres.pth"), strict=False)

        mAP.reset_global()
        mAPi.reset_global()
        mAPv.reset_global()
        mAPt.reset_global()
        allstep = 0
        print("Testing...")
        time_start = time.time()
        for test_dataloader_t, test_dataloader_b in zip(test_dataloaders_T, test_dataloaders_B):
            test_loop(test_dataloader_t, test_dataloader_b, model_T, model_B, activation, writer, wandb, final_eval=True, mode='test')
        print("Testing done...")

        mAPs = {'ivt': mAP, 'i': mAPi, 'v': mAPv, 't': mAPt}
        
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
        time_end = time.time()
        print("Time taken: ", time_end - time_start, "seconds")

# %% End
print("All done!\nShutting done...\nIt is what it is ...\nC'est finis! {}".format("-" * 100),
      file=open(logfile, 'a+'))
