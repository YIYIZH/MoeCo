import json
import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from .memory import PrototypeMemory
class EQL(nn.Module):
    def __init__(self,
                 train_weights,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 lambda_=0.00177,
                 version="v0_5",
                 cgl_split_source='full',
                 cgl_split_mode='absolute',
                 cgl_head_threshold=10000,
                 cgl_tail_threshold=1000,
                 cgl_head_percent=0.10,
                 cgl_tail_percent=0.10,
                 cgl_head_ratio=0.07,
                 cgl_tail_ratio=0.007,
                 cgl_full_stats_path=None):
        super(EQL, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.lambda_ = lambda_
        self.version = version
        self.train_weights = torch.tensor(train_weights, dtype=torch.float32)[:100]
        self.lambda_ = 999 / self.train_weights.sum()  # keep the original EQL frequency threshold for reference
        self.freq_info = torch.FloatTensor(self.train_weights / self.train_weights.sum()).cuda()
        self.num_class_included = torch.sum(self.freq_info < self.lambda_)
        self.temperature = 1
        self.prototype = PrototypeMemory(512, 100)

        self.cgl_split_source = cgl_split_source
        self.cgl_split_mode = cgl_split_mode
        self.cgl_head_threshold = cgl_head_threshold
        self.cgl_tail_threshold = cgl_tail_threshold
        self.cgl_head_percent = cgl_head_percent
        self.cgl_tail_percent = cgl_tail_percent
        self.cgl_head_ratio = cgl_head_ratio
        self.cgl_tail_ratio = cgl_tail_ratio
        self.cgl_full_stats_path = cgl_full_stats_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'all_data.json')

        split_counts = self._load_cgl_counts()
        self.ivt_head, self.ivt_medium, self.ivt_tail = self._build_cgl_groups(split_counts)
        self.cgl_tail_mask_cpu = torch.zeros(100, dtype=torch.float32)
        self.cgl_tail_mask_cpu[self.ivt_tail] = 1.0

        print(f"set up EQL (version {version}), {self.num_class_included} classes included by original EQL threshold.")
        print(
            f"CGL split source={self.cgl_split_source}, mode={self.cgl_split_mode}, "
            f"head_ratio={self.cgl_head_ratio}, tail_ratio={self.cgl_tail_ratio}, "
            f"head={len(self.ivt_head)} {self.ivt_head}, "
            f"medium={len(self.ivt_medium)} {self.ivt_medium}, "
            f"tail={len(self.ivt_tail)} classes"
        )

    def _load_cgl_counts(self):
        if self.cgl_split_source == 'full':
            try:
                return self._load_full_dataset_counts(self.cgl_full_stats_path)
            except Exception as exc:
                print(f"Warning: failed to load full CGL stats from {self.cgl_full_stats_path}: {exc}. Falling back to train_weights.")
        return self.train_weights.detach().cpu()

    @staticmethod
    def _load_full_dataset_counts(stats_path):
        with open(stats_path, 'r') as f:
            all_data = json.load(f)
        counts = torch.zeros(100, dtype=torch.float32)
        for video_stats in all_data.values():
            for label, count in video_stats.items():
                match = re.match(r'^(-?\d+)\s+', label)
                if match is None:
                    continue
                idx = int(match.group(1))
                if 0 <= idx < 100:
                    counts[idx] += float(count)
        return counts

    def _build_cgl_groups(self, counts):
        counts = torch.as_tensor(counts, dtype=torch.float32)[:100]
        ranked = sorted(range(100), key=lambda idx: (-float(counts[idx]), idx))

        if self.cgl_split_mode == 'absolute':
            head = [idx for idx in ranked if float(counts[idx]) > self.cgl_head_threshold]
            tail = [idx for idx in ranked if float(counts[idx]) < self.cgl_tail_threshold]
            head_set, tail_set = set(head), set(tail)
            medium = [idx for idx in ranked if idx not in head_set and idx not in tail_set]
        elif self.cgl_split_mode == 'percentile':
            head_n = max(1, int(round(100 * self.cgl_head_percent)))
            tail_n = max(1, int(round(100 * self.cgl_tail_percent)))
            if head_n + tail_n >= 100:
                raise ValueError('cgl_head_percent + cgl_tail_percent must leave at least one medium class.')
            head = ranked[:head_n]
            tail = ranked[-tail_n:]
            head_set, tail_set = set(head), set(tail)
            medium = [idx for idx in ranked if idx not in head_set and idx not in tail_set]
        elif self.cgl_split_mode == 'ratio':
            total = float(counts.sum())
            if total <= 0:
                raise ValueError('Cannot build ratio-based CGL groups because total class count is zero.')
            ratios = counts / total
            head = [idx for idx in ranked if float(ratios[idx]) > self.cgl_head_ratio]
            tail = [idx for idx in ranked if float(ratios[idx]) < self.cgl_tail_ratio]
            head_set, tail_set = set(head), set(tail)
            medium = [idx for idx in ranked if idx not in head_set and idx not in tail_set]
        else:
            raise ValueError(f"Unsupported cgl_split_mode: {self.cgl_split_mode}. Use 'absolute', 'percentile', or 'ratio'.")

        return head, medium, tail

    def kld_loss(self, pred, target):
        input = F.softmax(pred / self.temperature, dim=1)
        # find sum in dim=1 of target is 0
        # delete the rows in target and input where sum is 0
        non_zero_indices = torch.where(torch.sum(target, dim=1) != 0)[0]
        input = input[non_zero_indices]
        target = target[non_zero_indices]


        target = F.softmax(target / self.temperature, dim=1)

        loss = F.kl_div(torch.log(input), target, reduction='none')
        loss = loss * (self.temperature**2)  # Add a small epsilon to avoid division by zero

        batch_size = input.shape[0]

        if self.reduction == 'sum':
            # Change view to calculate instance-wise sum
            loss = loss.view(batch_size, -1)
            return torch.sum(loss, dim=1)

        elif self.reduction == 'mean':
            # Change view to calculate instance-wise mean
            loss = loss.view(batch_size, -1)
            return torch.sum(loss) / batch_size


    def forward(self,
                feature,
                cls_score,
                label,
                wb,
                epoch,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score

        #kld_loss = self.kld_loss(cls_score, label)

        weight_tail, weight_nontail = self.threshold_func()
        eql_w_neg = 1 - self.exclude_func() * weight_tail * (1 - label)
        eql_w_pos = 1 - weight_nontail * label
        eql_w = eql_w_neg * eql_w_pos
        #eql_w = eql_w_pos
        

        bce_loss = F.binary_cross_entropy_with_logits(cls_score, label,
                                                      reduction='none')
        bce_loss = bce_loss * eql_w

        pos_mask = (label == 1.0)
        neg_mask = (label == 0.0)
        
        # Calculate positive and negative components
        # pos_loss = bce_loss[pos_mask].mean()
        # neg_loss = bce_loss[neg_mask].mean()
        # Create a mask for ivt_head classes
        ivt_head_mask = torch.zeros_like(bce_loss, dtype=torch.bool)
        for idx in self.ivt_head:
            ivt_head_mask[:, idx] = True
        ivt_medium_mask = torch.zeros_like(bce_loss, dtype=torch.bool)
        for idx in self.ivt_medium:  
            ivt_medium_mask[:, idx] = True
        ivt_tail_mask = torch.zeros_like(bce_loss, dtype=torch.bool)
        for idx in self.ivt_tail:
            ivt_tail_mask[:, idx] = True
        
        prob = torch.sigmoid(cls_score)
        prob_head_pos = prob[ivt_head_mask & pos_mask].mean() # 2570
        prob_medium_pos = prob[ivt_medium_mask & pos_mask].mean() # 711
        prob_tail_pos = prob[ivt_tail_mask & pos_mask].mean() # 176
        prob_head_neg = 1.0 - prob[ivt_head_mask & neg_mask].mean() # 4576
        prob_medium_neg = 1.0 - prob[ivt_medium_mask & neg_mask].mean() # 30255
        prob_tail_neg = 1.0 - prob[ivt_tail_mask & neg_mask].mean() # 199912

        # Apply the mask to select only logits from ivt_head classes
        head_loss = bce_loss[ivt_head_mask].mean() # 7146
        medium_loss = bce_loss[ivt_medium_mask].mean() # 30966
        tail_loss = bce_loss[ivt_tail_mask].mean() # 200088
        pos_head_loss = bce_loss[pos_mask & ivt_head_mask] # 2570
        pos_medium_loss = bce_loss[pos_mask & ivt_medium_mask] # 711
        pos_tail_loss = bce_loss[pos_mask & ivt_tail_mask] # 176
        neg_head_loss = bce_loss[neg_mask & ivt_head_mask] # 4576
        neg_medium_loss = bce_loss[neg_mask & ivt_medium_mask] # 30255
        neg_tail_loss = bce_loss[neg_mask & ivt_tail_mask] # 199912
        pos_loss = bce_loss[pos_mask] # 3457
        neg_loss = bce_loss[neg_mask] # 234743
        wb.log({'pos_head_loss': pos_head_loss.mean().item(),
                'pos_medium_loss': pos_medium_loss.mean().item(),
                'pos_tail_loss': pos_tail_loss.mean().item(),
                'neg_head_loss': neg_head_loss.mean().item(),
                'neg_medium_loss': neg_medium_loss.mean().item(),
                'neg_tail_loss': neg_tail_loss.mean().item(),
                'pos_loss': pos_loss.mean().item(),
                'neg_loss': neg_loss.mean().item(),
                'head_loss': head_loss.mean().item(),
                'medium_loss': medium_loss.mean().item(),
                'tail_loss': tail_loss.mean().item(),
                'prob_head_pos': prob_head_pos.mean().item(),
                'prob_medium_pos': prob_medium_pos.mean().item(),
                'prob_tail_pos': prob_tail_pos.mean().item(),
                'prob_head_neg': prob_head_neg.mean().item(),
                'prob_medium_neg': prob_medium_neg.mean().item(),
                'prob_tail_neg': prob_tail_neg.mean().item()},
                step=epoch)

        #alpha = 0.1
        cls_loss = (pos_loss.mean() + neg_loss.mean()) / 2.0
        #cls_loss = torch.sum(bce_loss) / self.n_i
        #cls_loss = 0.001 * kld_loss + 0.5* pos_loss.mean()+ 0.5 * neg_loss.mean()
        #cls_loss = torch.sum(cls_loss * eql_w) / self.n_i
        #cls_loss = torch.mean(bce_loss * eql_w)
        # cls_loss = bce_loss.mean()
        # print(torch.mean(bce_loss * eql_w))
        #cls_loss = (pos_head_loss.mean() + neg_head_loss.mean()+ pos_medium_loss.mean() + neg_medium_loss.mean() + pos_tail_loss.mean() + neg_tail_loss.mean()) / 6.0 
        #cls_loss = (head_loss + medium_loss + tail_loss)/3
        #self.memory = self.prototype(feature, label)
        #fea_loss = self.prototype_consistency_loss(feature, label, self.memory)
        #wb.log({'fea_loss': fea_loss.item()}, step=epoch)
        #knn_loss = self.knn_loss(feature, label, 5)
        #wb.log({'knn_loss': knn_loss.item()}, step=epoch)
        #loss = cls_loss + 0.1 *fea_loss #+ knn_loss
        return cls_loss
    
    # def fairgrad(self, loss, alpha):
    #     return (loss ** (1-alpha)) / (1-alpha)
    
    # def knn_loss(self,features, labels, k=5, temperature=0.1, margin=1.0):
    #     """
    #     KNN loss that pulls together instances with the same label
        
    #     Args:
    #         features: tensor of shape [batch_size, feature_dim]
    #         labels: one-hot tensor of shape [batch_size, num_classes]
    #         k: number of nearest neighbors to consider
    #         temperature: scaling factor for similarity
    #         margin: margin for negative pairs
            
    #     Returns:
    #         loss: scalar tensor
    #     """
    #     features = features.transpose(0, 1)
    #     batch_size = features.shape[0]
        
    #     # Normalize features for cosine similarity
    #     norm_features = F.normalize(features, p=2, dim=1)
        
    #     # Compute pairwise cosine similarities
    #     sim_matrix = torch.mm(norm_features, norm_features.t())
        
    #     # Create label similarity matrix (1 if same class, 0 if different)
    #     #label_sim = torch.mm(labels, labels.t())

    #     # For multi-label classification where only exact matches count as same label
    #     # Calculate Hamming distance (0 = exact match, >0 = different)
    #     label_sim = (labels.unsqueeze(1) == labels.unsqueeze(0)).all(dim=2).float()
        
    #     loss = 0
    #     for i in range(batch_size):
    #         if labels[i].sum() == 0:
    #             continue  # Skip samples with no labels
                
    #         # For each instance, find similarities with other instances
    #         similarities = sim_matrix[i]
    #         is_same_label = label_sim[i] > 0
            
    #         # Exclude self
    #         mask = torch.ones(batch_size, dtype=torch.bool, device=features.device)
    #         mask[i] = False
    #         similarities = similarities[mask]
    #         is_same_label = is_same_label[mask]
            
    #         # If no positive samples, skip
    #         if is_same_label.sum() == 0:
    #             continue
                
    #         # Find bottom-k similar samples with same label
    #         pos_sim = similarities[is_same_label]
    #         if len(pos_sim) > k:
    #             _, bottomk_indices = torch.topk(-pos_sim, k)
    #             pos_sim = pos_sim[bottomk_indices]
            
    #         # Find negative samples (different label) (hardest negatives)
    #         neg_sim = similarities[~is_same_label]
    #         if len(neg_sim) > k:
    #             _, topk_indices = torch.topk(neg_sim, k)
    #             neg_sim = neg_sim[topk_indices]
            
    #         # Compute positive loss: pull positives closer than pos_margin
    #         pos_margin = 0.9 # 同类样本的相似度高于这个阈值,值越高，对特征相似度的要求越严格
    #         neg_margin = 0.3 # 不同类样本的相似度低于这个阈值,值越低，对特征相似度的要求越严格
    #         if len(pos_sim) > 0:
    #             pos_loss = torch.mean(torch.relu(pos_margin - pos_sim))
    #         else:
    #             pos_loss = 0.0
    #         if len(neg_sim) > 0:
    #             neg_loss = torch.mean(torch.relu(neg_sim - neg_margin))
    #         else:
    #             neg_loss = 0.0
            
    #         loss += pos_loss + neg_loss
            
    #     return loss / batch_size if batch_size > 0 else 0
        
        
    def exclude_func(self):
        # instance-level weight
        tail_class = self.pred_class_logits.new_zeros(self.n_c)
        tail_indices = torch.as_tensor(self.ivt_tail, dtype=torch.long, device=self.pred_class_logits.device)
        tail_class[tail_indices] = 1
        #bg_ind = self.n_c # background class

        weight1 = (self.gt_classes == tail_class).float()
        weight2 = weight1 * tail_class
        # set the value to 1 if the sum of first dim in weight2 is 0
        weight3 = (weight2.sum(dim=1) == 0).float() #  1 means instance i has no tail class
        weight = weight3.view(self.n_i, 1).expand(self.n_i, self.n_c)
        # find how many 0 in weight
        return weight

    def threshold_func(self):
        # class-level weight
        # mask = torch.rand(100).cuda()
        # mask = torch.where(mask > 0.5, torch.tensor(1, device=mask.device), torch.tensor(0, device=mask.device))
        weight = self.pred_class_logits.new_zeros(self.n_c)
        tail_indices = torch.as_tensor(self.ivt_tail, dtype=torch.long, device=self.pred_class_logits.device)
        weight[tail_indices] = 1
        # cereate a mask of shape (self.n_c,self.n_i) that 30% of the items are 1
        # Create mask using random values
        #mask = (torch.rand(self.n_c, self.n_i, device=self.pred_class_logits.device) < 0.1).float()
        mask = torch.bernoulli(torch.full((self.n_c,self.n_i), 0.1)) # randomly select percent of the tail classes to be supressed, will effect the random()
        # expand weight to the shape of mask
        weight = weight.expand(self.n_i, self.n_c).transpose(0, 1)
        weight_tail = (weight * mask.cuda()).transpose(0, 1)
        weight_nontail = ((1-weight) * mask.cuda()).transpose(0, 1)

        # weight_tail = weight_tail.view(1, self.n_c).expand(self.n_i, self.n_c)
        # weight_nontail = weight_nontail.view(1, self.n_c).expand(self.n_i, self.n_c)
        return [weight_tail, weight_nontail]
    
    def prototype_consistency_loss(self, feat, labels, memory_bank, temperature=0.5):
        """
        Encourages features to be close to their class prototypes
        
        Args:
            features: tensor of shape [batch_size, feature_dim]
            labels: one-hot tensor of shape [batch_size, num_classes]
            memory_bank: tensor of shape [num_classes, feature_dim] containing class prototypes
            temperature: scaling factor for similarity scores
        
        Returns:
            loss: scalar tensor
        """
        feat = feat.transpose(0, 1)
        batch_size = feat.shape[0]
        
        # Normalize features and prototypes for cosine similarity
        norm_features = F.normalize(feat, p=2, dim=1)
        norm_prototypes = F.normalize(memory_bank, p=2, dim=1)
        
        # Compute cosine similarity between features and all prototypes
        similarity = torch.mm(norm_features, norm_prototypes.t()) / temperature

        loss = 0
        for i in range(batch_size):
            if labels[i].sum() > 0:  # Skip samples with no labels
                # Weighted cross-entropy loss
                pos_sim = similarity[i] * labels[i]
                pos_sim = pos_sim[labels[i] > 0].sum()
                
                # Negative log-likelihood of positive similarity
                denom = torch.exp(similarity[i]).sum()
                loss -= torch.log(torch.exp(pos_sim) / denom)
        
        return loss / batch_size if batch_size > 0 else 0