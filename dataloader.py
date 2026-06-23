# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CODE RELEASE TO SUPPORT RESEARCH.
COMMERCIAL USE IS NOT PERMITTED.
#==============================================================================
An implementation based on:
***
    C.I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, N. Padoy. 
    Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. 
    Medical Image Analysis, 78 (2022) 102433.
***  
Created on Thu Oct 21 15:38:36 2021
#==============================================================================  
Copyright 2021 The Research Group CAMMA Authors All Rights Reserved.
(c) Research Group CAMMA, University of Strasbourg, France
@ Laboratory: CAMMA - ICube
@ Author: Chinedu Innocent Nwoye
@ Website: http://camma.u-strasbg.fr
#==============================================================================
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
#==============================================================================
"""

import os
import random
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import pickle
import json
from matrix import soft_label 
import matplotlib.pyplot as plt
from gmm import GaussianMixture
#from pycave.bayes import GaussianMixture as GMM
# global kfold_feats, load_feats, fold_num
# load_feats = False
# kfold_feats = {}
# # feats_dir = '../0-5fold/data_feats/run_KD_spatial_all_feat_ra111_KLT_t4'
# # feats_dir = '../0-5fold/data_feats/run_KD_spatial_all_feat_ra111_KLT'
# # feats_dir = '../0-5fold/data_feats/run_KD18_chal'
# # feats_dir = '../0-5fold/data_feats/res18'
# feats_dir = '../0-5fold/data_feats/run_res18_all'
# # feats_dir = '../0-5fold/data_feats/Q2L'
# # feats_dir = '../0-5fold/data_feats/run_KD_spatial_all_feat_ra111_KLT_t5'
# # feats_dir = '../0-5fold/data_feats/run_KD_spatial_all_feat_ra111_KLT_t2'
# # feats_dir = '../0-5fold/data_feats/run_KD_spatial_all_feat_ra111_KLT_t1'
# # feats_dir = '../0-5fold/data_feats/run_KD_spatial_all_feat_ra111_KLT_t4_tenco'
# # feats_dir = '../0-5fold/data_feats/run_KD_spatial_all_feat_ra111_KLT_res50_seed11'
# # feats_dir = '../0-5fold/data_feats/run_KD_spatial_all_feat_ra111_KLT_singletea'
# os.makedirs(feats_dir, exist_ok=True)

instruments = {
    "grasper": {
        "tip": "forked, rectangular, hollow",
        "shaft": "cylindrical, dark, matte",
        "wrist": "jointed, flexible"
    },
    "bipolar": {
        "tip": "forked, curved, blue",
        "shaft": "cylindrical, metallic",
        "wrist": "jointed, cylindrical"
    },
    "clipper": {
        "tip": "forked, serrated, sharp",
        "shaft": "cylindrical, metallic, smooth",
        "wrist": "rigid"
    },
    "scissors": {
        "tip": "forked, sharp, pointed",
        "shaft": "cylindrical, metallic",
        "wrist": "rigid"
    },
    "hook": {
        "tip": "curved, pointed, tapered",
        "wrist": "not present",
        "shaft": "cylindrical, dark, matte"
    },
    "irrigator": {
        "tip": "not present",
        "wrist": "not present",
        "shaft": "cylindrical, perforated, metallic"
    },
    "null": {
        "tip": "not present",
        "shaft": "not present",
        "wrist": "not present"
    }   
}

ins = {
    0: "grasper",
    1: "bipolar",
    2: "hook",
    3: "scissors",
    4: "clipper",
    5: "irrigator",
    6: "null"
}

target = {
    0: "gallbladder",
    1: "cystic_plate",
    2: "cystic_duct",
    3: "cystic_artery",
    4: "cystic_pedicle",
    5: "blood_vessel",
    6: "fluid",
    7: "abdominal_wall_cavity",
    8: "liver",
    9: "adhesion",
    10: "omentum",
    11: "peritoneum",
    12: "gut",
    13: "specimen_bag",
    14: "null_target"
}

verb = {
    0: "grasp",
    1: "retract",
    2: "dissect",
    3: "coagulate",
    4: "clip",
    5: "cut",
    6: "aspirate",
    7: "irrigate",
    8: "pack",
    9: "null_verb"
}

class CholecT50():
    def __init__(self,
                 args,
                 dataset_dir,
                 dataset_variant="cholect45-crossval",
                 test_fold=1,
                 augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90'],
                 model=None):
        """ Args
                dataset_dir : common path to the dataset (excluding videos, output)
                list_video  : list video IDs, e.g:  ['VID01', 'VID02']
                aug         : data augumentation style
                split       : data split ['train', 'val', 'test']
            Call
                batch_size: int, 
                shuffle: True or False
            Return
                tuple ((image), (tool_label, verb_label, target_label, triplet_label))
        """
        self.args = args
        self.dataset_dir = dataset_dir
        self.list_dataset_variant = {
            "cholect45-crossval": "for CholecT45 dataset variant with the official cross-validation splits.",
            "cholect50-crossval": "for CholecT50 dataset variant with the official cross-validation splits",
            "cholect50-challenge": "for CholecT50 dataset variant as used in CholecTriplet challenge",
            "cholect45-challenge": "for CholecT45 dataset variant as used in CholecTriplet challenge",
            "cholect50": "for the CholecT50 dataset with original splits used in rendezvous paper",
            "cholect45": "a pointer to cholect45-crossval",
        }
        assert dataset_variant in self.list_dataset_variant.keys(), print(dataset_variant,
                                                                          "is not a valid dataset variant")
        video_split = self.split_selector(case=dataset_variant)
        train_videos = sum([v for k, v in video_split.items() if k != self.args.kfold],
                           []) if 'crossval' in dataset_variant else video_split['train']
        test_videos = sum([v for k, v in video_split.items() if k == self.args.kfold],
                          []) if 'crossval' in dataset_variant else video_split['test']
        if 'crossval' in dataset_variant:
            val_videos = train_videos[-5:]
            train_videos = train_videos[:-5]
        else:
            val_videos = video_split['val']
        self.train_records = ['VID{}'.format(str(v).zfill(2)) for v in train_videos]
        self.val_records = ['VID{}'.format(str(v).zfill(2)) for v in val_videos]
        self.test_records = ['VID{}'.format(str(v).zfill(2)) for v in test_videos]
        # self.test_records = ['VID{}'.format(str(v).zfill(2)) for v in train_videos]
        self.augmentations = {
            'original': self.no_augumentation,
            'vflip': transforms.RandomVerticalFlip(0.4),
            'hflip': transforms.RandomHorizontalFlip(0.4),
            'contrast': transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0, hue=0),
            'rot90': transforms.RandomRotation(90, expand=True),
            'brightness': transforms.RandomAdjustSharpness(sharpness_factor=1.6, p=0.5),
            'contrast': transforms.RandomAutocontrast(p=0.5),
        }
        self.augmentation_list = []
        for aug in augmentation_list:
            self.augmentation_list.append(self.augmentations[aug])
        trainform, testform = self.transform()

        self.build_train_dataset(testform, model=model)
        self.build_val_dataset(testform, model=model)
        self.build_test_dataset(testform, model=model)
        #self.build_test_train_dataset(testform, model=model)
        # self.build_test_T(testform, model=model)
        # self.build_test_B(testform, model=model)
        self.feats_dir = '../0-5fold/data_feats/run_{}'.format(self.args.version1)
        os.makedirs(self.feats_dir, exist_ok=True)

    def list_dataset_variants(self):
        print(self.list_dataset_variant)

    def list_augmentations(self):
        print(self.augmentations.keys())

    def split_selector(self, case='cholect50'):
        switcher = {
            'cholect50': {
                'train': [1, 15, 26, 40, 52, 65, 79, 2, 18, 27, 43, 56, 66, 92, 4, 22, 31, 47, 57, 68, 96, 5, 23, 35,
                          48, 60, 70, 103, 13, 25, 36, 49, 62, 75, 110],
                'val': [8, 12, 29, 50, 78],
                'test': [6, 51, 10, 73, 14, 74, 32, 80, 42, 111]
            },
            'cholect50-challenge': {
                'train': [1, 15, 26, 40, 52, 79, 2, 27, 43, 56, 66, 4, 22, 31, 47, 57, 68, 23, 35, 48, 60, 70, 13, 25,
                          49, 62, 75, 8, 12, 29, 50, 78, 6, 51, 10, 73, 14, 32, 80, 42],
                'val': [5, 18, 36, 65, 74],
                'test': [92, 96, 103, 110, 111]
            },
            'cholect45-challenge': {
                'train': [1, 15, 26, 40, 52, 79, 2, 27, 43, 56, 66, 4, 22, 31, 47, 57, 5, 23, 35, 48, 60, 18, 13, 25,
                          49, 62, 65, 8, 12, 29, 50, 78, 6, 51, 10, 36, 14, 32, 80, 42],
                'val': [68, 70, 73, 74, 75],
                # 'test': [92, 96, 103, 110, 111]
                'test': [68, 70, 73, 74, 75]
            },
            'cholect45-crossval': {
                1: [79, 2, 51, 6, 25, 14, 66, 23, 50, ],
                2: [80, 32, 5, 15, 40, 47, 26, 48, 70, ],
                3: [31, 57, 36, 18, 52, 68, 10, 8, 73, ],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12, ],
                5: [78, 43, 62, 35, 74, 1, 56, 4, 13, ],
            },
            'cholect50-crossval': {
                1: [79, 2, 51, 6, 25, 14, 66, 23, 50, 111],
                2: [80, 32, 5, 15, 40, 47, 26, 48, 70, 96],
                3: [31, 57, 36, 18, 52, 68, 10, 8, 73, 103],
                4: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
                5: [78, 43, 62, 35, 74, 1, 56, 4, 13, 92],
            },
        }
        return switcher.get(case)

    def no_augumentation(self, x):
        return x

    def transform(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        op_test = [transforms.Resize((256, 448)), transforms.Resize((256, 448)), transforms.ToTensor(), normalize, ]
        op_train = [transforms.Resize((256, 448))] + self.augmentation_list + [transforms.Resize((256, 448)),
                                                                               transforms.ToTensor(), normalize, ]
        testform = transforms.Compose(op_test)
        trainform = transforms.Compose(op_train)
        return trainform, testform

    def get_data_statics(self, video):
        with open('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/all_data_t50.json', 'r') as f:
            data = json.load(f)
        return data[video]
    
    def generate_miu_var(self, iterable_dataset):
        # read from instrument_tip_descriptors_clip_768d_embeddings.npy
        instrument_tip_descriptors = np.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/clip/instrument_tip_descriptors_clip_1024d_embeddings.npy', allow_pickle=True).item()
        # read from instrument_shaft_descriptors_clip_768d_embeddings.npy
        instrument_shaft_descriptors = np.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/clip/instrument_shaft_descriptors_clip_1024d_embeddings.npy', allow_pickle=True).item()
        # read from instrument_wrist_descriptors_clip_768d_embeddings.npy
        instrument_wrist_descriptors = np.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/clip/instrument_wrist_descriptors_clip_1024d_embeddings.npy', allow_pickle=True).item()
        # build a dictionary to store the mean and variance of each item in the descriptor tip
        tip_descriptors_dict = {}
        shaft_descriptors_dict = {}
        wrist_descriptors_dict = {}
        for descriptor, embedding in instrument_tip_descriptors.items():
            tip_descriptors_dict[descriptor] = {
                'frames': [],
                'mean': np.zeros_like(embedding[0]), # initialize as zero
                'miu': np.mean(embedding, axis=0),
                'cov': np.zeros_like(embedding[0])
            }
        for descriptor, embedding in instrument_shaft_descriptors.items():
            shaft_descriptors_dict[descriptor] = {
                'frames': [],
                'mean': np.zeros_like(embedding[0]), # initialize as zero
                'miu': np.mean(embedding, axis=0),
                'cov': np.zeros_like(embedding[0])
            }
        for descriptor, embedding in instrument_wrist_descriptors.items():
            wrist_descriptors_dict[descriptor] = {
                'frames': [],
                'mean': np.zeros_like(embedding[0]), # initialize as zero
                'miu': np.mean(embedding, axis=0),
                'cov': np.zeros_like(embedding[0])
            }
        
        for dataset in iterable_dataset:
            tool_labels = dataset.tool_labels
            tool_labels = [np.where(label == 1)[0] for label in tool_labels]
            tool_names = [[ins[label] for label in labels] for labels in tool_labels]
            # if tool names is null, then use the null descriptor
            for i in range(len(tool_names)):
                if tool_names[i] == []:
                    tool_names[i] = ["null"]

            feats = dataset.foundation_feats
            # compute the cosine similarity betwwen feats and its corresponding descriptor
            for i in range(len(tool_names)):
                # find all sub items in tool_name from instruments
                for instrument, attributes in instruments.items():
                    for tool_name in tool_names[i]:
                        if tool_name == instrument:
                            for items in attributes['tip'].split(", "):
                                tip_descriptors_dict[items]['frames'].append(feats[i])
                            for items in attributes['shaft'].split(", "):
                                shaft_descriptors_dict[items]['frames'].append(feats[i])
                            for items in attributes['wrist'].split(", "):
                                wrist_descriptors_dict[items]['frames'].append(feats[i])
        
        for descriptor, data in tip_descriptors_dict.items():
            tip_descriptors_dict[descriptor]['miu'] = tip_descriptors_dict[descriptor]['miu'] / np.linalg.norm(tip_descriptors_dict[descriptor]['miu'])
            #normalize the frames
            data['frames'] = data['frames'] / np.linalg.norm(data['frames'], axis=1, keepdims=True)
            # compute the covariance matrix
            tip_descriptors_dict[descriptor]['mean'] = np.mean(data['frames'], axis=0)
            dif = data['frames'] - tip_descriptors_dict[descriptor]['miu']
            cov = np.sum(dif * dif, axis=0) / len(data['frames'])
            tip_descriptors_dict[descriptor]['cov'] = cov
            
        for descriptor, data in shaft_descriptors_dict.items():
            shaft_descriptors_dict[descriptor]['miu'] = shaft_descriptors_dict[descriptor]['miu'] / np.linalg.norm(shaft_descriptors_dict[descriptor]['miu'])
            #normalize the frames
            data['frames'] = data['frames'] / np.linalg.norm(data['frames'], axis=1, keepdims=True)
            shaft_descriptors_dict[descriptor]['mean'] = np.mean(data['frames'], axis=0)
            dif = data['frames'] - shaft_descriptors_dict[descriptor]['miu']
            # compute the covariance matrix in diagonal form
            cov = np.sum(dif * dif, axis=0) / len(data['frames'])
            #cov = np.diag(cov)
            shaft_descriptors_dict[descriptor]['cov'] = cov
            
        for descriptor, data in wrist_descriptors_dict.items():
            wrist_descriptors_dict[descriptor]['miu'] = wrist_descriptors_dict[descriptor]['miu'] / np.linalg.norm(wrist_descriptors_dict[descriptor]['miu'])
            #normalize the frames
            data['frames'] = data['frames'] / np.linalg.norm(data['frames'], axis=1, keepdims=True)
            wrist_descriptors_dict[descriptor]['mean'] = np.mean(data['frames'], axis=0)
            dif = data['frames'] - wrist_descriptors_dict[descriptor]['miu']
            # compute the covariance matrix in diagonal form
            cov = np.sum(dif * dif, axis=0) / len(data['frames'])
            #cov = np.diag(cov)
            wrist_descriptors_dict[descriptor]['cov'] = cov
        
        # append the miu and cov from each item in tip_descriptors_dict
        miu_tip = []
        cov_tip = []
        for descriptor, data in tip_descriptors_dict.items():
            miu_tip.append(data['miu'])
            cov_tip.append(data['cov'])
        miu_tip = np.array(miu_tip)
        cov_tip = np.array(cov_tip)
        miu_shaft = []
        cov_shaft = []
        for descriptor, data in shaft_descriptors_dict.items():
            miu_shaft.append(data['miu'])
            cov_shaft.append(data['cov'])
        miu_shaft = np.array(miu_shaft)
        cov_shaft = np.array(cov_shaft)
        miu_wrist = []
        cov_wrist = []
        for descriptor, data in wrist_descriptors_dict.items():
            miu_wrist.append(data['miu'])
            cov_wrist.append(data['cov'])
        miu_wrist = np.array(miu_wrist)
        cov_wrist = np.array(cov_wrist)
        
        # Reshape miu_tip to match required dimensions (1, n_components, n_features)
        miu_tip = miu_tip.reshape((1, miu_tip.shape[0], -1))
        cov_tip = cov_tip.reshape((1, cov_tip.shape[0], -1))
        # covert miu_tip and cov_tip to torch tensor
        miu_tip = torch.from_numpy(miu_tip)
        cov_tip = torch.from_numpy(cov_tip)
        miu_shaft = miu_shaft.reshape((1, miu_shaft.shape[0], -1))
        cov_shaft = cov_shaft.reshape((1, cov_shaft.shape[0], -1))
        miu_wrist = miu_wrist.reshape((1, miu_wrist.shape[0], -1))
        cov_wrist = cov_wrist.reshape((1, cov_wrist.shape[0], -1))
        miu_shaft = torch.from_numpy(miu_shaft)
        cov_shaft = torch.from_numpy(cov_shaft)
        miu_wrist = torch.from_numpy(miu_wrist)
        cov_wrist = torch.from_numpy(cov_wrist)

        self.gmm_tip = GaussianMixture(n_components=miu_tip.shape[1], #10
                               n_features=miu_tip.shape[2], 
                               covariance_type="diag", 
                               init_params='kmeans',
                               mu_init=miu_tip,
                               var_init=cov_tip)
        self.gmm_shaft = GaussianMixture(n_components=miu_shaft.shape[1], #7
                               n_features=miu_shaft.shape[2], 
                               covariance_type="diag", 
                               init_params='kmeans',
                               mu_init=miu_shaft,
                               var_init=cov_shaft)
        self.gmm_wrist = GaussianMixture(n_components=miu_wrist.shape[1], #5
                               n_features=miu_wrist.shape[2], 
                               covariance_type="diag", 
                               init_params='kmeans',
                               mu_init=miu_wrist,
                               var_init=cov_wrist)
        
        
        self.gmm_tip.save('gmm_tip_t45_clip_1024_f'+str(self.args.kfold)+'.pth')
        self.gmm_shaft.save('gmm_shaft_t45_clip_1024_f'+str(self.args.kfold)+'.pth')
        self.gmm_wrist.save('gmm_wrist_t45_clip_1024_f'+str(self.args.kfold)+'.pth')

        for i in range(100):
            prob = self.gmm_tip.predict_proba(torch.from_numpy(data['frames'][i]))
            # get the top 3 probability's index
            top_k_index = torch.topk(prob, 3)[1]
            print(top_k_index)

        # target_prompt = np.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/clip/verb_prompt_clip_768d_embeddings.npy', allow_pickle=True).item()
        # target_descriptors_dict = {}
        # for descriptor, embedding in target_prompt.items():
        #     target_descriptors_dict[descriptor] = {
        #         'frames': [],
        #         'mean': np.zeros_like(embedding[0]), # initialize as zero
        #         'miu': np.mean(embedding, axis=0),
        #         'cov': np.zeros_like(embedding[0])
        #     }
        # for dataset in iterable_dataset:
        #     target_labels = dataset.verb_labels
        #     target_labels = [np.where(label == 1)[0] for label in target_labels]
        #     target_names = [[verb[label] for label in labels] for labels in target_labels]
        #     for i in range(len(target_names)):
        #         if target_names[i] == []:
        #             target_names[i] = ["null_verb"]

        #     feats = dataset.feats
        #     for i in range(len(target_names)):
        #         for descriptor, data in target_descriptors_dict.items():
        #             for target_name in target_names[i]:
        #                 if target_name == descriptor:
        #                     target_descriptors_dict[descriptor]['frames'].append(feats[i])
                            
        # for descriptor, data in target_descriptors_dict.items():
        #     target_descriptors_dict[descriptor]['miu'] = target_descriptors_dict[descriptor]['miu'] / np.linalg.norm(target_descriptors_dict[descriptor]['miu'])
        #     if len(data['frames']) > 0:
        #         data['frames'] = data['frames'] / np.linalg.norm(data['frames'], axis=1, keepdims=True)
        #         target_descriptors_dict[descriptor]['mean'] = np.mean(data['frames'], axis=0)
        #         dif = data['frames'] - target_descriptors_dict[descriptor]['miu']
        #         cov = np.sum(dif * dif, axis=0) / len(data['frames'])
        #         target_descriptors_dict[descriptor]['cov'] = cov

        # miu_target = []
        # cov_target = []
        # for descriptor, data in target_descriptors_dict.items():
        #     miu_target.append(data['miu'])
        #     cov_target.append(data['cov'])
        # miu_target = np.array(miu_target)
        # cov_target = np.array(cov_target)
        # miu_target = miu_target.reshape((1, miu_target.shape[0], -1))
        # cov_target = cov_target.reshape((1, cov_target.shape[0], -1))
        # miu_target = torch.from_numpy(miu_target)
        # cov_target = torch.from_numpy(cov_target)

        # self.gmm_target = GaussianMixture(n_components=miu_target.shape[1],
        #                                  n_features=miu_target.shape[2],
        #                                  covariance_type="diag",
        #                                  init_params='kmeans',
        #                                  mu_init=miu_target,
        #                                  var_init=cov_target)
        
        # self.gmm_target.save('gmm_verb.pth')

        # for i in range(100):
        #     prob = self.gmm_target.predict_proba(torch.from_numpy(data['frames'][i]))
        #     # get the top 3 probability's index
        #     top_k_index = torch.topk(prob, 2)[1]
        #     print(top_k_index)
            

    def build_train_dataset(self, transform, model):
        iterable_dataset = []
        data_statics = []
        for video in self.train_records:
            dataset = T50(args=self.args, split='train', img_dir=os.path.join(self.dataset_dir, 'data', video),
                          triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                          tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                          verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                          target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                          transform=transform,
                          model=model)
            print("building train dataset:", video)
            data_statics.append(self.get_data_statics(video))
            iterable_dataset.append(dataset)
        self.train_dataset = ConcatDataset(iterable_dataset)

        # self.generate_miu_var(iterable_dataset)
            
        combined_statics = {}
        for d in data_statics:
            for key, value in d.items():
                if key in combined_statics:
                    combined_statics[key] += value
                else:
                    combined_statics[key] = value
        self.train_data_statics = combined_statics

    def build_val_dataset(self, transform, model):
        iterable_dataset = []
        data_statics = []
        for video in self.val_records:
            dataset = T50(args=self.args, split='val', img_dir=os.path.join(self.dataset_dir, 'data', video),
                          triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                          tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                          verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                          target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                          transform=transform,
                          model=model)
            iterable_dataset.append(dataset)
            print("building val dataset:", video)
            data_statics.append(self.get_data_statics(video))
        self.val_dataset = iterable_dataset
        combined_statics = {}
        for d in data_statics:
            for key, value in d.items():
                if key in combined_statics:
                    combined_statics[key] += value
                else:
                    combined_statics[key] = value
        self.val_data_statics = combined_statics

    def build_test_dataset(self, transform, model):
        iterable_dataset = []
        data_statics = []
        for video in self.test_records:
            dataset = T50(args=self.args, split='test', img_dir=os.path.join(self.dataset_dir, 'data', video),
                          triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                          tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                          verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                          target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                          transform=transform,
                          model=model)
            iterable_dataset.append(dataset)
            print("building test dataset:", video)
            data_statics.append(self.get_data_statics(video))
        self.test_dataset = iterable_dataset
        combined_statics = {}
        for d in data_statics:
            for key, value in d.items():
                if key in combined_statics:
                    combined_statics[key] += value
                else:
                    combined_statics[key] = value
        self.test_data_statics = combined_statics
    
    def build_test_T(self, transform, model):
        iterable_dataset = []
        data_statics = []
        for video in self.test_records:
            dataset = T50_T(args=self.args, split='test', img_dir=os.path.join(self.dataset_dir, 'data', video),
                          triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                          tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                          verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                          target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                          transform=transform,
                          model=model)
            iterable_dataset.append(dataset)
            print("building test dataset:", video)
            data_statics.append(self.get_data_statics(video))
        self.test_dataset_T = iterable_dataset
        combined_statics = {}
        for d in data_statics:
            for key, value in d.items():
                if key in combined_statics:
                    combined_statics[key] += value
                else:
                    combined_statics[key] = value
        self.test_data_statics_T = combined_statics

    def build_test_B(self, transform, model):
        iterable_dataset = []
        data_statics = []
        for video in self.test_records:
            dataset = T50_B(args=self.args, split='test', img_dir=os.path.join(self.dataset_dir, 'data', video),
                          triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                          tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                          verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                          target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                          transform=transform,
                          model=model)
            iterable_dataset.append(dataset)
            print("building test dataset:", video)
            data_statics.append(self.get_data_statics(video))
        self.test_dataset_B = iterable_dataset
        combined_statics = {}
        for d in data_statics:
            for key, value in d.items():
                if key in combined_statics:
                    combined_statics[key] += value
                else:
                    combined_statics[key] = value
        self.test_data_statics_B = combined_statics

    def build_test_train_dataset(self, transform, model):
        iterable_dataset = []
        for video in self.train_records[-9:]:
            dataset = T50(args=self.args, split='test', img_dir=os.path.join(self.dataset_dir, 'data', video),
                          triplet_file=os.path.join(self.dataset_dir, 'triplet', '{}.txt'.format(video)),
                          tool_file=os.path.join(self.dataset_dir, 'instrument', '{}.txt'.format(video)),
                          verb_file=os.path.join(self.dataset_dir, 'verb', '{}.txt'.format(video)),
                          target_file=os.path.join(self.dataset_dir, 'target', '{}.txt'.format(video)),
                          transform=transform,
                          model=model)
            iterable_dataset.append(dataset)
        self.test_train_dataset = iterable_dataset

    def build(self):
        return (self.train_dataset, self.val_dataset, self.test_dataset)
    
    def build_T(self):
        return self.test_dataset_T
    
    def build_B(self):
        return self.test_dataset_B
    
    def cls_weights(self):
        info_train = {}
        for d in self.train_data_statics:
            cid = int(d.split(' ')[0])
            info_train[cid] = self.train_data_statics[d]
        info_test = {}

        info_train_array = [0] * 101
        for key, value in info_train.items():
            if key == -1:
                info_train_array[100] = value
            else:
                info_train_array[key] = value
        
        for d in self.test_data_statics:
            cid = int(d.split(' ')[0])
            info_test[cid] = self.test_data_statics[d]

        info_test_array = [0] * 101
        for key, value in info_test.items():
            if key == -1:
                info_test_array[100] = value
            else:
                info_test_array[key] = value

        return info_train_array, info_test_array


class T50(Dataset):
    def __init__(self, args, split, img_dir, triplet_file, tool_file, verb_file, target_file, transform=None,
                 target_transform=None, model=None):
        self.args = args
        self.split = split
        self.img_dir = img_dir
        self.feats_dir = '/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0-5fold/data_feats/run_{}'.format(self.args.version1)

        with open(os.path.join(self.feats_dir, 'k' + str(self.args.kfold) + '_feats.pkl'), 'rb') as f:
            parts = self.img_dir.split('/')  # 按 '/' 分割路径
            last_part = parts[-1]    # 取最后一部分 'VID01'
            vid_number = last_part.replace('VID', '')
            self.terl_feats = pickle.load(f)[vid_number]
        if args.input_dim == 768:
            self.feats_dir = '/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/clip/clip_features_clip_image768_t50.pkl'
        else:
            self.feats_dir = '/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/clip/clip_features_cliprn50_image1024_t50.pkl'
            # self.feats_dir = '/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/clip/clip_features_ViT-L-14_feats_32.pkl'
        with open (self.feats_dir, 'rb') as f:
            parts = self.img_dir.split('/')  # 按 '/' 分割路径
            last_part = parts[-1]    # 取最后一部分 'VID01'
            self.foundation_feats = pickle.load(f)[last_part]

        # if args.clip_image:
        #     ff = args.clip_i_feature #'TERL/0_5fold_TCN_black/clip_features_ViT-L-14_feats.pkl'
        #     with open(ff, 'rb') as f:
        #         self.feats_i = pickle.load(f)[self.img_dir[-5:]]
        #         #self.feats = self.feats / np.linalg.norm(self.feats, axis=-1, keepdims=True)
        #         if args.fuse == 'add':
        #             self.feats += self.feats_i
        #         elif args.fuse == 'concat':
        #             self.feats = np.concatenate((self.feats, self.feats_i), axis=-1)
        #         elif args.fuse == 'replace':
        #             self.feats = self.feats_i
        #         elif args.fuse == 'multiply':
        #             self.feats = self.feats * self.feats_i
        # if args.clip_text:
        #     ff = args.clip_t_feature #'TERL/0_5fold_TCN_black/clip_features_ViT-L-14_feats.pkl'
        #     with open(ff, 'rb') as f:
        #         self.feats_t = pickle.load(f)[self.img_dir[-5:]]
        #         #print(self.feats_t.shape)
                
        #         #self.feats = self.feats / np.linalg.norm(self.feats, axis=-1, keepdims=True)
        #         if args.fuse == 'add':
        #             self.feats += self.feats_t
        #         elif args.fuse == 'concat':
        #             self.feats = np.concatenate((self.feats, self.feats_t), axis=-1)
        #         elif args.fuse == 'replace':
        #             self.feats = self.feats_t
        #         elif args.fuse == 'multiply':
        #             self.feats = self.feats * self.feats_t
        

        if args.clip_text and args.fuse == 'replace':
            idx = [i for i in range(len(self.terl_feats))]
        else:
            sub1 = self.terl_feats[1:, :] - self.terl_feats[:-1, :]
            idx1 = np.where(np.sum(sub1, axis=-1) == 0)[0]
            idx2 = np.unique(np.concatenate((idx1, idx1 + 1)))
            idx = [i for i in range(len(self.terl_feats)) if i not in list(idx2)]

        # if args.norm:
        #     self.feats = self.feats / np.linalg.norm(self.feats, axis=-1, keepdims=True)
        #     self.feats_i = self.feats_i / np.linalg.norm(self.feats_i, axis=-1, keepdims=True)
        #     self.feats_t = self.feats_t / np.linalg.norm(self.feats_t, axis=-1, keepdims=True)
        self.idx = [1 if i in idx else 0 for i in range(len(self.terl_feats))]
        self.terl_feats = self.terl_feats[idx, :]
        self.foundation_feats = self.foundation_feats[idx, :]
        print(self.terl_feats.shape)
        # if args.clip_image:
        #     self.feats_i = self.feats_i[idx, :]
        # if args.clip_text:
        #     self.feats_t = self.feats_t[idx, :]
        self.matrix = soft_label()

        self.triplet_labels = np.loadtxt(triplet_file, dtype=int, delimiter=',', )[idx, 1:]
        self.tool_labels = np.loadtxt(tool_file, dtype=int, delimiter=',', )[idx, 1:]
        self.verb_labels = np.loadtxt(verb_file, dtype=int, delimiter=',', )[idx, 1:]
        self.target_labels = np.loadtxt(target_file, dtype=int, delimiter=',', )[idx, 1:]
        self.triplet_soft_labels = self.generate_soft_labels(self.triplet_labels)

        self.transform = transform
        self.target_transform = target_transform

        if args.ins_prompt != -1:
            ins_prompt_source = getattr(args, 'ins_prompt_source', 'gmm')
            if ins_prompt_source == 'random':
                self._build_random_ins_prompts()
            elif ins_prompt_source in ['gmm', 'random_gmm']:
                self._build_gmm_ins_prompts(random_activation=(ins_prompt_source == 'random_gmm'))
            elif ins_prompt_source == 'gt_attribute':
                self._build_gt_attribute_ins_prompts()
            else:
                raise ValueError(f"Unsupported --ins_prompt_source: {ins_prompt_source}")

        # if args.target_prompt != -1:
        #     gmm_target = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm_target.pth', weights_only=True)
        #     self.gmm_target = GaussianMixture(
        #         n_components=gmm_target['state_dict']['mu'].shape[1],
        #         n_features=gmm_target['state_dict']['mu'].shape[2],
        #         covariance_type="diag",
        #         init_params='kmeans',
        #         mu_init=gmm_target['state_dict']['mu'],
        #         var_init=gmm_target['state_dict']['var']
        #     )
        #     feats_tensor = torch.from_numpy(self.feats)  # (N, D)
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     feats_tensor = feats_tensor.to(device)
        #     self.gmm_target = self.gmm_target.to(device)
        #     prob_target = self.gmm_target.predict_proba(feats_tensor)  # (N, n_components)
        #     topk_target = torch.topk(prob_target, 2, dim=1)[1]    # (N, 2)
        #     self.target_prompt = self.gmm_target.state_dict()['mu'].to(device)[0]
        #     self.target_prompt = self.target_prompt[topk_target].cpu().numpy()
        
        # if args.verb_prompt != -1:
        #     gmm_verb = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm_verb.pth', weights_only=True)
        #     self.gmm_verb = GaussianMixture(
        #         n_components=gmm_verb['state_dict']['mu'].shape[1],
        #         n_features=gmm_verb['state_dict']['mu'].shape[2],
        #         covariance_type="diag",
        #         init_params='kmeans',
        #         mu_init=gmm_verb['state_dict']['mu'],
        #         var_init=gmm_verb['state_dict']['var']
        #     )
        #     feats_tensor = torch.from_numpy(self.feats)  # (N, D)
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     feats_tensor = feats_tensor.to(device)
        #     self.gmm_verb = self.gmm_verb.to(device)
        #     prob_verb = self.gmm_verb.predict_proba(feats_tensor)  # (N, n_components)
        #     topk_verb = torch.topk(prob_verb, 2, dim=1)[1]    # (N, 2)
        #     self.verb_prompt = self.gmm_verb.state_dict()['mu'].to(device)[0]
        #     self.verb_prompt = self.verb_prompt[topk_verb].cpu().numpy()
        
    


    def _build_gmm_ins_prompts(self, random_activation=False):
        args = self.args
        if args.dataset_variant == 'cholect50':
            if args.input_dim == 768:
                gmm_tip = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_tip_t50_clip_768_f' + str(args.kfold) + '.pth', weights_only=True)
                gmm_shaft = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_shaft_t50_clip_768_f' + str(args.kfold) + '.pth', weights_only=True)
                gmm_wrist = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_wrist_t50_clip_768_f' + str(args.kfold) + '.pth', weights_only=True)
            else:
                gmm_tip = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_tip_t50_clip_1024_f' + str(args.kfold) + '.pth', weights_only=True)
                gmm_shaft = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_shaft_t50_clip_1024_f' + str(args.kfold) + '.pth', weights_only=True)
                gmm_wrist = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_wrist_t50_clip_1024_f' + str(args.kfold) + '.pth', weights_only=True)
        elif args.dataset_variant == 'cholect45-crossval':
            if args.input_dim == 768:
                gmm_tip = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_tip_t45_clip_768_f' + str(args.kfold) + '.pth', weights_only=True)
                gmm_shaft = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_shaft_t45_clip_768_f' + str(args.kfold) + '.pth', weights_only=True)
                gmm_wrist = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_wrist_t45_clip_768_f' + str(args.kfold) + '.pth', weights_only=True)
            else:
                gmm_tip = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_tip_t45_clip_1024_f' + str(args.kfold) + '.pth', weights_only=True)
                gmm_shaft = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_shaft_t45_clip_1024_f' + str(args.kfold) + '.pth', weights_only=True)
                gmm_wrist = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_wrist_t45_clip_1024_f' + str(args.kfold) + '.pth', weights_only=True)
        else:
            raise ValueError(f"Unsupported dataset_variant for GMM instrument prompts: {args.dataset_variant}")

        self.gmm_tip = GaussianMixture(
            n_components=gmm_tip['state_dict']['mu'].shape[1],
            n_features=gmm_tip['state_dict']['mu'].shape[2],
            covariance_type="diag",
            init_params='kmeans',
            mu_init=gmm_tip['state_dict']['mu'],
            var_init=gmm_tip['state_dict']['var']
        )
        self.gmm_shaft = GaussianMixture(
            n_components=gmm_shaft['state_dict']['mu'].shape[1],
            n_features=gmm_shaft['state_dict']['mu'].shape[2],
            covariance_type="diag",
            init_params='kmeans',
            mu_init=gmm_shaft['state_dict']['mu'],
            var_init=gmm_shaft['state_dict']['var']
        )
        self.gmm_wrist = GaussianMixture(
            n_components=gmm_wrist['state_dict']['mu'].shape[1],
            n_features=gmm_wrist['state_dict']['mu'].shape[2],
            covariance_type="diag",
            init_params='kmeans',
            mu_init=gmm_wrist['state_dict']['mu'],
            var_init=gmm_wrist['state_dict']['var']
        )

        feats_tensor = torch.from_numpy(self.foundation_feats)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        feats_tensor = feats_tensor.to(device)
        self.gmm_tip = self.gmm_tip.to(device)
        self.gmm_shaft = self.gmm_shaft.to(device)
        self.gmm_wrist = self.gmm_wrist.to(device)

        prob_tip = self.gmm_tip.predict_proba(feats_tensor)
        prob_shaft = self.gmm_shaft.predict_proba(feats_tensor)
        prob_wrist = self.gmm_wrist.predict_proba(feats_tensor)

        k = args.topk
        if random_activation:
            video_name = os.path.basename(self.img_dir)
            try:
                video_id = int(video_name.replace('VID', ''))
            except ValueError:
                video_id = sum(ord(ch) for ch in video_name)
            split_offset = {'train': 0, 'val': 10000, 'test': 20000}.get(self.split, 30000)
            seed = int(args.seed) + int(args.kfold) * 100000 + video_id * 100 + split_offset + 777
            rng = np.random.default_rng(seed)
            topk_tip = torch.as_tensor(
                rng.integers(0, prob_tip.shape[1], size=(prob_tip.shape[0], k)),
                dtype=torch.long,
                device=device,
            )
            topk_shaft = torch.as_tensor(
                rng.integers(0, prob_shaft.shape[1], size=(prob_shaft.shape[0], k)),
                dtype=torch.long,
                device=device,
            )
            topk_wrist = torch.as_tensor(
                rng.integers(0, prob_wrist.shape[1], size=(prob_wrist.shape[0], k)),
                dtype=torch.long,
                device=device,
            )
            print(f"Using randomly activated GMM instrument prompts for {video_name}: topk={k}, seed={seed}")
        else:
            topk_tip = torch.topk(prob_tip, k, dim=1)[1]
            topk_shaft = torch.topk(prob_shaft, k, dim=1)[1]
            topk_wrist = torch.topk(prob_wrist, k, dim=1)[1]

        mu_tip = self.gmm_tip.state_dict()['mu'].to(device)[0]
        mu_shaft = self.gmm_shaft.state_dict()['mu'].to(device)[0]
        mu_wrist = self.gmm_wrist.state_dict()['mu'].to(device)[0]

        self.tip_prompt = mu_tip[topk_tip].cpu().numpy()
        self.shaft_prompt = mu_shaft[topk_shaft].cpu().numpy()
        self.wrist_prompt = mu_wrist[topk_wrist].cpu().numpy()


    def _load_descriptor_index(self, part):
        descriptor_path = (
            '/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/'
            f'0_5fold_TCN_black/clip/instrument_{part}_descriptors_clip_{self.args.input_dim}d_embeddings.npy'
        )
        descriptor_embeddings = np.load(descriptor_path, allow_pickle=True).item()
        return {descriptor: idx for idx, descriptor in enumerate(descriptor_embeddings.keys())}

    def _gt_attribute_indices(self, tool_label, part, descriptor_to_idx):
        active_tool_ids = np.where(tool_label == 1)[0].tolist()
        if not active_tool_ids:
            active_tool_ids = [6]

        attrs = []
        for tool_id in active_tool_ids:
            tool_name = ins.get(int(tool_id), 'null')
            for attr in instruments[tool_name][part].split(', '):
                if attr not in attrs:
                    attrs.append(attr)

        indices = []
        missing = []
        for attr in attrs:
            if attr in descriptor_to_idx:
                indices.append(descriptor_to_idx[attr])
            else:
                missing.append(attr)
        if missing:
            print(f"Warning: missing {part} descriptors for GT attributes {missing}; falling back when needed.")
        if not indices:
            fallback = descriptor_to_idx.get('not present', 0)
            indices = [fallback]

        k = self.args.topk
        while len(indices) < k:
            indices.append(indices[-1])
        return indices[:k]

    def _build_gt_attribute_ins_prompts(self):
        self._build_gmm_ins_prompts(random_activation=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tip_to_idx = self._load_descriptor_index('tip')
        shaft_to_idx = self._load_descriptor_index('shaft')
        wrist_to_idx = self._load_descriptor_index('wrist')

        topk_tip = np.asarray([
            self._gt_attribute_indices(label, 'tip', tip_to_idx)
            for label in self.tool_labels
        ], dtype=np.int64)
        topk_shaft = np.asarray([
            self._gt_attribute_indices(label, 'shaft', shaft_to_idx)
            for label in self.tool_labels
        ], dtype=np.int64)
        topk_wrist = np.asarray([
            self._gt_attribute_indices(label, 'wrist', wrist_to_idx)
            for label in self.tool_labels
        ], dtype=np.int64)

        mu_tip = self.gmm_tip.state_dict()['mu'].to(device)[0]
        mu_shaft = self.gmm_shaft.state_dict()['mu'].to(device)[0]
        mu_wrist = self.gmm_wrist.state_dict()['mu'].to(device)[0]

        self.tip_prompt = mu_tip[torch.as_tensor(topk_tip, dtype=torch.long, device=device)].cpu().numpy()
        self.shaft_prompt = mu_shaft[torch.as_tensor(topk_shaft, dtype=torch.long, device=device)].cpu().numpy()
        self.wrist_prompt = mu_wrist[torch.as_tensor(topk_wrist, dtype=torch.long, device=device)].cpu().numpy()
        print(
            f"Using GT attribute oracle instrument prompts for {os.path.basename(self.img_dir)}: "
            f"shape={self.tip_prompt.shape}, topk={self.args.topk}"
        )


    def _build_random_ins_prompts(self):
        num_frames = self.foundation_feats.shape[0]
        feat_dim = self.foundation_feats.shape[1]
        topk = self.args.topk
        video_name = os.path.basename(self.img_dir)
        try:
            video_id = int(video_name.replace('VID', ''))
        except ValueError:
            video_id = sum(ord(ch) for ch in video_name)
        split_offset = {'train': 0, 'val': 10000, 'test': 20000}.get(self.split, 30000)
        seed = int(self.args.seed) + int(self.args.kfold) * 100000 + video_id * 100 + split_offset
        rng = np.random.default_rng(seed)

        def make_prompt(offset):
            prompt = rng.normal(
                loc=0.0,
                scale=getattr(self.args, 'random_ins_prompt_std', 1.0),
                size=(num_frames, topk, feat_dim),
            ).astype(np.float32)
            if not getattr(self.args, 'random_ins_prompt_no_normalize', False):
                norm = np.linalg.norm(prompt, axis=-1, keepdims=True)
                prompt = prompt / np.clip(norm, 1e-6, None)
            return prompt

        self.tip_prompt = make_prompt(0)
        self.shaft_prompt = make_prompt(1)
        self.wrist_prompt = make_prompt(2)
        print(
            f"Using random instrument prompt features for {video_name}: "
            f"shape={(num_frames, topk, feat_dim)}, seed={seed}"
        )

    def generate_soft_labels(self, triplet_labels):
        ivt_head = [17, 60, 19]
        n_samples, n_classes = triplet_labels.shape
        soft_labels = np.zeros_like(triplet_labels, dtype=np.float32)
        for i in range(n_samples):
            true_indices = np.where(triplet_labels[i] == 1)[0]
            if len(true_indices) == 0:
                continue  # 处理无标注样本的情况
            for index in true_indices:
                # if index not in ivt_head:
                #     soft_labels[i] = triplet_labels[i]
                # else:
                # 计算soft label
                sim_vectors = self.matrix[index]
                soft_labels[i] = np.maximum(soft_labels[i], sim_vectors)
        return soft_labels

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # basename = "{}.png".format(str(self.triplet_labels[index, 0]).zfill(6))
        # if self.split == 'train' and random.random() > 0.7:
        #     num_clips = random.choice(range(10, 1000 if len(self.feats) > 1000 else len(self.feats)))
        #     random_index = random.choice(range(0, len(self.feats) - num_clips))
        #     idx = [random_index + i for i in range(num_clips)]
        # else:
        #     idx = [i for i in range(len(self.feats))]
        random_rate = self.args.random
        idx = [i for i in range(len(self.terl_feats))]
        if self.args.scale_factor == 1:
            start = 10
        else:
            start = 100
        if self.split == 'train' and random.random() > random_rate:
            num_clips = random.choice(range(start, 1000 if len(self.terl_feats) > 1000 else len(self.terl_feats)))
            random_index = random.choice(range(0, len(self.terl_feats) - num_clips))
            idx = [random_index + i for i in range(num_clips)]
        # implement random reverse to the image and labels
        if self.split == 'train' and random.random() > random_rate and self.args.reverse:
            reverse_idx = idx[::-1]
            idx = reverse_idx
        # if self.split == 'train' and random.random() > 0.5 and self.args.step:
        #     step = random.choice(range(1, 10))
        #     idx = idx[::step]
        

        triplet_label = self.triplet_labels[idx, :]
        triplet_soft_label = self.triplet_soft_labels[idx, :]
        tool_label = self.tool_labels[idx, :]
        verb_label = self.verb_labels[idx, :]
        target_label = self.target_labels[idx, :]

        weights = torch.ones(triplet_label.shape[0], dtype=torch.float32)
        # sub1 = triplet_label[1:, :] - triplet_label[:-1, :]
        # sub1 = sub1[:,1:]
        # idx1 = np.where(np.any(sub1 != 0, axis=-1))[0]
        # idx2 = np.unique(np.concatenate((idx1, idx1 + 1)))
        # weights[idx2] = self.args.transit
        # idx3 = np.where(~np.isin(np.arange(len(weights)), idx2))[0]
        # weights[idx3] = 1 - self.args.transit

        feats = self.terl_feats[idx]
        features = feats
        # convert prompt to numpy
        if self.args.ins_prompt != -1:
            tip_prompt = self.tip_prompt[idx,:,:]
            shaft_prompt = self.shaft_prompt[idx,:,:]
            wrist_prompt = self.wrist_prompt[idx,:,:]
            features = [feats, tip_prompt, shaft_prompt, wrist_prompt]
        if self.args.target_prompt != -1:
            target_prompt = self.target_prompt[idx,:,:]
            features = [feats, target_prompt]
        if self.args.verb_prompt != -1:
            verb_prompt = self.verb_prompt[idx,:,:]
            features = [feats, verb_prompt]
        if self.args.clip_text and self.args.clip_image:
            clip_i = self.feats_i[idx]
            clip_t = self.feats_t[idx]
            features = [feats, clip_i, clip_t]
        if self.args.ins_prompt != -1 and self.args.target_prompt != -1 and self.args.verb_prompt != -1:
            features = [feats, tip_prompt, shaft_prompt, wrist_prompt,verb_prompt, target_prompt]
        # if self.args.target_prompt != -1:
        #     features = [features, target_prompt]
        # if self.args.verb_prompt != -1:
        #     features = [features, verb_prompt]

        if self.target_transform:
            triplet_label = self.target_transform(triplet_label)
        # print("000000")
        # print(triplet_label.shape)
        return features, (tool_label, verb_label, target_label, triplet_label, triplet_soft_label), weights

class T50_T(Dataset):
    def __init__(self, args, split, img_dir, triplet_file, tool_file, verb_file, target_file, transform=None,
                 target_transform=None, model=None):
        self.args = args
        self.split = split
        self.img_dir = img_dir
        self.feats_dir = '/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0-5fold/data_feats/run_241014_6_baseline_learnT_swinT_div2_p1_con1_k'+str(self.args.kfold)+'_seed20000912'

        with open(os.path.join(self.feats_dir, 'k' + str(self.args.kfold) + '_feats.pkl'), 'rb') as f:
            parts = self.img_dir.split('/')  # 按 '/' 分割路径
            last_part = parts[-1]    # 取最后一部分 'VID01'
            vid_number = last_part.replace('VID', '')
            self.terl_feats = pickle.load(f)[vid_number]
        
        self.feats_dir = '/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/clip/clip_features_clip_image768_t50.pkl'
        with open (self.feats_dir, 'rb') as f:
            parts = self.img_dir.split('/')  # 按 '/' 分割路径
            last_part = parts[-1]    # 取最后一部分 'VID01'
            self.foundation_feats = pickle.load(f)[last_part]


        if args.clip_text and args.fuse == 'replace':
            idx = [i for i in range(len(self.terl_feats))]
        else:
            sub1 = self.terl_feats[1:, :] - self.terl_feats[:-1, :]
            idx1 = np.where(np.sum(sub1, axis=-1) == 0)[0]
            idx2 = np.unique(np.concatenate((idx1, idx1 + 1)))
            idx = [i for i in range(len(self.terl_feats)) if i not in list(idx2)]


        self.terl_feats = self.terl_feats[idx, :]
        self.foundation_feats = self.foundation_feats[idx, :]
        print(self.terl_feats.shape)

        self.matrix = soft_label()

        self.triplet_labels = np.loadtxt(triplet_file, dtype=int, delimiter=',', )[idx, 1:]
        self.tool_labels = np.loadtxt(tool_file, dtype=int, delimiter=',', )[idx, 1:]
        self.verb_labels = np.loadtxt(verb_file, dtype=int, delimiter=',', )[idx, 1:]
        self.target_labels = np.loadtxt(target_file, dtype=int, delimiter=',', )[idx, 1:]
        self.triplet_soft_labels = self.generate_soft_labels(self.triplet_labels)

        self.transform = transform
        self.target_transform = target_transform

        if args.ins_prompt != -1:
            #load gmm model

            gmm_tip = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_tip_t45_clip_768_f'+ str(self.args.kfold)+'.pth', weights_only=True)
            gmm_shaft = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_shaft_t45_clip_768_f'+ str(self.args.kfold)+'.pth', weights_only=True)
            gmm_wrist = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_wrist_t45_clip_768_f'+ str(self.args.kfold)+'.pth', weights_only=True)

            self.gmm_tip = GaussianMixture(
                n_components=gmm_tip['state_dict']['mu'].shape[1],
                n_features=gmm_tip['state_dict']['mu'].shape[2],
                covariance_type="diag",
                init_params='kmeans',
                mu_init=gmm_tip['state_dict']['mu'],
                var_init=gmm_tip['state_dict']['var']
            )
            self.gmm_shaft = GaussianMixture(
                n_components=gmm_shaft['state_dict']['mu'].shape[1],
                n_features=gmm_shaft['state_dict']['mu'].shape[2],
                covariance_type="diag",
                init_params='kmeans',
                mu_init=gmm_shaft['state_dict']['mu'],
                var_init=gmm_shaft['state_dict']['var']
            )
            self.gmm_wrist = GaussianMixture(
                n_components=gmm_wrist['state_dict']['mu'].shape[1],
                n_features=gmm_wrist['state_dict']['mu'].shape[2],
                covariance_type="diag",
                init_params='kmeans',
                mu_init=gmm_wrist['state_dict']['mu'],
                var_init=gmm_wrist['state_dict']['var']
            )

            feats_tensor = torch.from_numpy(self.foundation_feats)  # (N, D)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            feats_tensor = feats_tensor.to(device)
            self.gmm_tip = self.gmm_tip.to(device)
            self.gmm_shaft = self.gmm_shaft.to(device)
            self.gmm_wrist = self.gmm_wrist.to(device)

            # Batch predict_proba for all features
            prob_tip = self.gmm_tip.predict_proba(feats_tensor)  # (N, n_components)
            prob_shaft = self.gmm_shaft.predict_proba(feats_tensor)
            prob_wrist = self.gmm_wrist.predict_proba(feats_tensor)

            # Get top-3 indices for each sample
            k = 3
            topk_tip = torch.topk(prob_tip, k, dim=1)[1]    # (N, 3)
            topk_shaft = torch.topk(prob_shaft, k, dim=1)[1]
            topk_wrist = torch.topk(prob_wrist, k, dim=1)[1]


            # Get mu for each GMM
            mu_tip = self.gmm_tip.state_dict()['mu'].to(device)[0]      # (n_components, D)
            mu_shaft = self.gmm_shaft.state_dict()['mu'].to(device)[0]
            mu_wrist = self.gmm_wrist.state_dict()['mu'].to(device)[0]

            # Gather the top-3 mus for each sample efficiently
            self.tip_prompt = mu_tip[topk_tip].cpu().numpy()       # (N, 3, D)
            self.shaft_prompt = mu_shaft[topk_shaft].cpu().numpy()
            self.wrist_prompt = mu_wrist[topk_wrist].cpu().numpy()
        
    
    def generate_soft_labels(self, triplet_labels):
        ivt_head = [17, 60, 19]
        n_samples, n_classes = triplet_labels.shape
        soft_labels = np.zeros_like(triplet_labels, dtype=np.float32)
        for i in range(n_samples):
            true_indices = np.where(triplet_labels[i] == 1)[0]
            if len(true_indices) == 0:
                continue  # 处理无标注样本的情况
            for index in true_indices:
                # if index not in ivt_head:
                #     soft_labels[i] = triplet_labels[i]
                # else:
                # 计算soft label
                sim_vectors = self.matrix[index]
                soft_labels[i] = np.maximum(soft_labels[i], sim_vectors)
        return soft_labels

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # basename = "{}.png".format(str(self.triplet_labels[index, 0]).zfill(6))
        # if self.split == 'train' and random.random() > 0.7:
        #     num_clips = random.choice(range(10, 1000 if len(self.feats) > 1000 else len(self.feats)))
        #     random_index = random.choice(range(0, len(self.feats) - num_clips))
        #     idx = [random_index + i for i in range(num_clips)]
        # else:
        #     idx = [i for i in range(len(self.feats))]
        random_rate = self.args.random
        idx = [i for i in range(len(self.terl_feats))]
        if self.args.scale_factor == 1:
            start = 10
        else:
            start = 100
        if self.split == 'train' and random.random() > random_rate:
            num_clips = random.choice(range(start, 1000 if len(self.terl_feats) > 1000 else len(self.terl_feats)))
            random_index = random.choice(range(0, len(self.terl_feats) - num_clips))
            idx = [random_index + i for i in range(num_clips)]
        # implement random reverse to the image and labels
        if self.split == 'train' and random.random() > random_rate and self.args.reverse:
            reverse_idx = idx[::-1]
            idx = reverse_idx
        # if self.split == 'train' and random.random() > 0.5 and self.args.step:
        #     step = random.choice(range(1, 10))
        #     idx = idx[::step]
        

        triplet_label = self.triplet_labels[idx, :]
        triplet_soft_label = self.triplet_soft_labels[idx, :]
        tool_label = self.tool_labels[idx, :]
        verb_label = self.verb_labels[idx, :]
        target_label = self.target_labels[idx, :]

        weights = torch.ones(triplet_label.shape[0], dtype=torch.float32)


        feats = self.terl_feats[idx]
        features = feats
        # convert prompt to numpy
        if self.args.ins_prompt != -1:
            tip_prompt = self.tip_prompt[idx,:,:]
            shaft_prompt = self.shaft_prompt[idx,:,:]
            wrist_prompt = self.wrist_prompt[idx,:,:]
            features = [feats, tip_prompt, shaft_prompt, wrist_prompt]
        if self.args.target_prompt != -1:
            target_prompt = self.target_prompt[idx,:,:]
            features = [feats, target_prompt]
        if self.args.verb_prompt != -1:
            verb_prompt = self.verb_prompt[idx,:,:]
            features = [feats, verb_prompt]
        if self.args.clip_text and self.args.clip_image:
            clip_i = self.feats_i[idx]
            clip_t = self.feats_t[idx]
            features = [feats, clip_i, clip_t]
        if self.args.ins_prompt != -1 and self.args.target_prompt != -1 and self.args.verb_prompt != -1:
            features = [feats, tip_prompt, shaft_prompt, wrist_prompt,verb_prompt, target_prompt]


        if self.target_transform:
            triplet_label = self.target_transform(triplet_label)

        return features, (tool_label, verb_label, target_label, triplet_label, triplet_soft_label), weights


class T50_B(Dataset):
    def __init__(self, args, split, img_dir, triplet_file, tool_file, verb_file, target_file, transform=None,
                 target_transform=None, model=None):
        self.args = args
        self.split = split
        self.img_dir = img_dir
        self.feats_dir = '/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0-5fold/data_feats/run_241014_6_baseline_learnT_swinB_div4_p1_con1_k'+str(self.args.kfold)+'_seed20000912'

        with open(os.path.join(self.feats_dir, 'k' + str(self.args.kfold) + '_feats.pkl'), 'rb') as f:
            parts = self.img_dir.split('/')  # 按 '/' 分割路径
            last_part = parts[-1]    # 取最后一部分 'VID01'
            vid_number = last_part.replace('VID', '')
            self.terl_feats = pickle.load(f)[vid_number]
        
        self.feats_dir = '/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/clip/clip_features_cliprn50_image1024_t50.pkl'
        with open (self.feats_dir, 'rb') as f:
            parts = self.img_dir.split('/')  # 按 '/' 分割路径
            last_part = parts[-1]    # 取最后一部分 'VID01'
            self.foundation_feats = pickle.load(f)[last_part]


        if args.clip_text and args.fuse == 'replace':
            idx = [i for i in range(len(self.terl_feats))]
        else:
            sub1 = self.terl_feats[1:, :] - self.terl_feats[:-1, :]
            idx1 = np.where(np.sum(sub1, axis=-1) == 0)[0]
            idx2 = np.unique(np.concatenate((idx1, idx1 + 1)))
            idx = [i for i in range(len(self.terl_feats)) if i not in list(idx2)]


        self.terl_feats = self.terl_feats[idx, :]
        self.foundation_feats = self.foundation_feats[idx, :]
        print(self.terl_feats.shape)

        self.matrix = soft_label()

        self.triplet_labels = np.loadtxt(triplet_file, dtype=int, delimiter=',', )[idx, 1:]
        self.tool_labels = np.loadtxt(tool_file, dtype=int, delimiter=',', )[idx, 1:]
        self.verb_labels = np.loadtxt(verb_file, dtype=int, delimiter=',', )[idx, 1:]
        self.target_labels = np.loadtxt(target_file, dtype=int, delimiter=',', )[idx, 1:]
        self.triplet_soft_labels = self.generate_soft_labels(self.triplet_labels)

        self.transform = transform
        self.target_transform = target_transform

        if args.ins_prompt != -1:
            #load gmm model

            gmm_tip = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_tip_t45_clip_1024_f'+ str(self.args.kfold)+'.pth', weights_only=True)
            gmm_shaft = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_shaft_t45_clip_1024_f'+ str(self.args.kfold)+'.pth', weights_only=True)
            gmm_wrist = torch.load('/home/student/PycharmProjects/ComputerVision_Codes/TERL-Vision1/0_5fold_TCN_black/gmm/gmm_wrist_t45_clip_1024_f'+ str(self.args.kfold)+'.pth', weights_only=True)

            self.gmm_tip = GaussianMixture(
                n_components=gmm_tip['state_dict']['mu'].shape[1],
                n_features=gmm_tip['state_dict']['mu'].shape[2],
                covariance_type="diag",
                init_params='kmeans',
                mu_init=gmm_tip['state_dict']['mu'],
                var_init=gmm_tip['state_dict']['var']
            )
            self.gmm_shaft = GaussianMixture(
                n_components=gmm_shaft['state_dict']['mu'].shape[1],
                n_features=gmm_shaft['state_dict']['mu'].shape[2],
                covariance_type="diag",
                init_params='kmeans',
                mu_init=gmm_shaft['state_dict']['mu'],
                var_init=gmm_shaft['state_dict']['var']
            )
            self.gmm_wrist = GaussianMixture(
                n_components=gmm_wrist['state_dict']['mu'].shape[1],
                n_features=gmm_wrist['state_dict']['mu'].shape[2],
                covariance_type="diag",
                init_params='kmeans',
                mu_init=gmm_wrist['state_dict']['mu'],
                var_init=gmm_wrist['state_dict']['var']
            )

            feats_tensor = torch.from_numpy(self.foundation_feats)  # (N, D)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            feats_tensor = feats_tensor.to(device)
            self.gmm_tip = self.gmm_tip.to(device)
            self.gmm_shaft = self.gmm_shaft.to(device)
            self.gmm_wrist = self.gmm_wrist.to(device)

            # Batch predict_proba for all features
            prob_tip = self.gmm_tip.predict_proba(feats_tensor)  # (N, n_components)
            prob_shaft = self.gmm_shaft.predict_proba(feats_tensor)
            prob_wrist = self.gmm_wrist.predict_proba(feats_tensor)

            # Get top-3 indices for each sample
            k = 3
            topk_tip = torch.topk(prob_tip, k, dim=1)[1]    # (N, 3)
            topk_shaft = torch.topk(prob_shaft, k, dim=1)[1]
            topk_wrist = torch.topk(prob_wrist, k, dim=1)[1]


            # Get mu for each GMM
            mu_tip = self.gmm_tip.state_dict()['mu'].to(device)[0]      # (n_components, D)
            mu_shaft = self.gmm_shaft.state_dict()['mu'].to(device)[0]
            mu_wrist = self.gmm_wrist.state_dict()['mu'].to(device)[0]

            # Gather the top-3 mus for each sample efficiently
            self.tip_prompt = mu_tip[topk_tip].cpu().numpy()       # (N, 3, D)
            self.shaft_prompt = mu_shaft[topk_shaft].cpu().numpy()
            self.wrist_prompt = mu_wrist[topk_wrist].cpu().numpy()
        
    
    def generate_soft_labels(self, triplet_labels):
        ivt_head = [17, 60, 19]
        n_samples, n_classes = triplet_labels.shape
        soft_labels = np.zeros_like(triplet_labels, dtype=np.float32)
        for i in range(n_samples):
            true_indices = np.where(triplet_labels[i] == 1)[0]
            if len(true_indices) == 0:
                continue  # 处理无标注样本的情况
            for index in true_indices:
                # if index not in ivt_head:
                #     soft_labels[i] = triplet_labels[i]
                # else:
                # 计算soft label
                sim_vectors = self.matrix[index]
                soft_labels[i] = np.maximum(soft_labels[i], sim_vectors)
        return soft_labels

    def __len__(self):
        return 1

    def __getitem__(self, index):
        # basename = "{}.png".format(str(self.triplet_labels[index, 0]).zfill(6))
        # if self.split == 'train' and random.random() > 0.7:
        #     num_clips = random.choice(range(10, 1000 if len(self.feats) > 1000 else len(self.feats)))
        #     random_index = random.choice(range(0, len(self.feats) - num_clips))
        #     idx = [random_index + i for i in range(num_clips)]
        # else:
        #     idx = [i for i in range(len(self.feats))]
        random_rate = self.args.random
        idx = [i for i in range(len(self.terl_feats))]
        if self.args.scale_factor == 1:
            start = 10
        else:
            start = 100
        if self.split == 'train' and random.random() > random_rate:
            num_clips = random.choice(range(start, 1000 if len(self.terl_feats) > 1000 else len(self.terl_feats)))
            random_index = random.choice(range(0, len(self.terl_feats) - num_clips))
            idx = [random_index + i for i in range(num_clips)]
        # implement random reverse to the image and labels
        if self.split == 'train' and random.random() > random_rate and self.args.reverse:
            reverse_idx = idx[::-1]
            idx = reverse_idx
        # if self.split == 'train' and random.random() > 0.5 and self.args.step:
        #     step = random.choice(range(1, 10))
        #     idx = idx[::step]
        

        triplet_label = self.triplet_labels[idx, :]
        triplet_soft_label = self.triplet_soft_labels[idx, :]
        tool_label = self.tool_labels[idx, :]
        verb_label = self.verb_labels[idx, :]
        target_label = self.target_labels[idx, :]

        weights = torch.ones(triplet_label.shape[0], dtype=torch.float32)


        feats = self.terl_feats[idx]
        features = feats
        # convert prompt to numpy
        if self.args.ins_prompt != -1:
            tip_prompt = self.tip_prompt[idx,:,:]
            shaft_prompt = self.shaft_prompt[idx,:,:]
            wrist_prompt = self.wrist_prompt[idx,:,:]
            features = [feats, tip_prompt, shaft_prompt, wrist_prompt]
        if self.args.target_prompt != -1:
            target_prompt = self.target_prompt[idx,:,:]
            features = [feats, target_prompt]
        if self.args.verb_prompt != -1:
            verb_prompt = self.verb_prompt[idx,:,:]
            features = [feats, verb_prompt]
        if self.args.clip_text and self.args.clip_image:
            clip_i = self.feats_i[idx]
            clip_t = self.feats_t[idx]
            features = [feats, clip_i, clip_t]
        if self.args.ins_prompt != -1 and self.args.target_prompt != -1 and self.args.verb_prompt != -1:
            features = [feats, tip_prompt, shaft_prompt, wrist_prompt,verb_prompt, target_prompt]


        if self.target_transform:
            triplet_label = self.target_transform(triplet_label)

        return features, (tool_label, verb_label, target_label, triplet_label, triplet_soft_label), weights


if __name__ == "__main__":
    print("Refers to https://github.com/CAMMA-public/cholect45 for the usage guide.")
