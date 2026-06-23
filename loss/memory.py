import torch
from torch import nn
import math
import numpy as np
from util import AverageMeter

class PrototypeMemory(nn.Module):
    """
    memory buffer that stores prototypes of each category.
    """
    def __init__(self, featSize=512, classSize=100, proto_path=None, momentum=0.9):
        super(PrototypeMemory, self).__init__()

        #stdv = 1. / math.sqrt(inputSize / 3)
        self.momentum = momentum
        #self.register_buffer('memory', torch.rand(classSize, featSize).mul_(2 * stdv).add_(-stdv))
        self.memory = torch.zeros(classSize, featSize)
        #proto = np.load('prototype_leafseg_DenseNet121_v2.npy', allow_pickle=True).item()
        #proto = np.load('prototype_leafseg_resnet50_v2.npy', allow_pickle=True).item()
        # print(proto_path)
        # proto = np.load(proto_path, allow_pickle=True).item()
        #proto = np.load('prototype_cotton80_DenseNet121_train.npy', allow_pickle=True).item()
        #proto = np.load('prototype_soylocal_resnet50_train.npy', allow_pickle=True).item()
        #proto = np.load('prototype_soylocal_DenseNet121_train.npy', allow_pickle=True).item()
        #proto = np.load('prototype_soyglo_resnet50_train.npy', allow_pickle=True).item()
        #proto = np.load('prototype_soyglo_DenseNet121_train.npy', allow_pickle=True).item()
        #proto = np.load('prototype_soygen_DenseNet121_train.npy', allow_pickle=True).item()
        #proto = np.load('prototype_soyage_resnet50_train.npy', allow_pickle=True).item()
        #proto = np.load('prototype_soygen_DenseNet121_train.npy', allow_pickle=True).item()
        #proto = np.load('prototype_CUB_resnet50_train.npy', allow_pickle=True).item()
        #proto = np.load('prototype_CUB_DenseNet121_train.npy', allow_pickle=True).item()
        #proto = np.load('prototype_soyageR6_DenseNet121_train.npy', allow_pickle=True).item()
        #proto = np.load('prototype_soyageR1_resnet50_train.npy', allow_pickle=True).item()
        #proto = np.load('prototype_Soy200_resnet50_train.npy', allow_pickle=True).item()
        #proto = np.load('prototype_Soy200_DenseNet121_train.npy', allow_pickle=True).item()
        #proto = np.load('prototype_Soy100_resnet50.npy', allow_pickle=True).item()
        #proto = np.load('prototype_Soy100_DenseNet121.npy', allow_pickle=True).item()
        #proto = np.load('prototype_cherry_resnet50.npy', allow_pickle=True).item()
        #proto = np.load('prototype_cherry_DenseNet121.npy', allow_pickle=True).item()

        # for k in proto:
        #     # 对特征进行归一化，并保存到矩阵memory中
        #     norm = np.power(np.sum(np.power(proto[k].avg,  2), axis=0), 0.5)
        #     # k与品种相对应，如k=0，则self.memory[k]的值为品种1的特征，k=1,则self.memory[k]的值为品种2的特征，以此类推
        #     self.memory[k] = torch.from_numpy(np.divide(proto[k].avg, norm))

        self.memory = self.memory.cuda()


    def forward(self, feature,  label):
        # normalize feature
        # f_norm = feature.pow(2).sum(1, keepdim=True).pow(0.5)
        # feature = feature.div(f_norm)

        with torch.no_grad():
            # update memory
            self.memory = self.update_memory_bank(feature, label)

        #out_f = torch.mm(feature, self.memory.transpose(0,1))

        return self.memory
    
    def update_memory_bank(self, new_features, labels):
        """
        Update class prototypes in memory bank with new batch features
        
        Args:
            memory_bank: tensor of shape [num_classes, feature_dim]
            new_features: tensor of shape [batch_size, feature_dim]
            labels: one-hot tensor of shape [batch_size, num_classes]
            alpha: momentum factor (higher values retain more of old prototype)
        """
        batch_size, num_classes = labels.shape
        # normalize new_features
        new_features = new_features.transpose(0, 1)
        new_features = new_features.div(new_features.pow(2).sum(1, keepdim=True).pow(0.5))
        
        # Aggregate features for each class in current batch
        new_prototypes = torch.zeros_like(self.memory)
        class_counts = torch.zeros(num_classes, device=self.memory.device)
        
        for i in range(batch_size):
            for c in range(num_classes):
                if labels[i, c] > 0:  # If this sample belongs to class c
                    new_prototypes[c] += new_features[i]
                    class_counts[c] += 1
        
        # Average the features for classes that appeared in this batch
        valid_indices = class_counts > 0
        new_prototypes[valid_indices] /= class_counts[valid_indices].unsqueeze(1)
            
        # Update memory bank with momentum      
        self.memory[valid_indices] = self.momentum * self.memory[valid_indices] + (1 - self.momentum) * new_prototypes[valid_indices]
            
        return self.memory