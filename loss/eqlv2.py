import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
#from mmdet.utils import get_root_logger
from functools import partial

ivt_head = [17, 60, 19]
ivt_medium = [58, 7, 20, 12, 94, 61, 96, 82, 59, 57, 29, 79, 16]  # Medium frequency IVT indices
ivt_tail = [78, 69, 1, 18, 68, 95, 99, 63, 14, 27, 88, 4, 22, 92, 36, 28, 62, 98, 21, 30, 51, 10, 13, 52, 64, 37, 23, 97, 44, 6, 66, 34, 90, 33, 87, 39, 76, 71, 84, 93, 40, 0, 53, 26, 3, 32, 45, 24, 9, 31, 25, 73, 35, 81, 11, 75, 15, 48, 83, 77, 43, 2, 91, 86, 89, 5, 72, 46, 56, 67, 70, 65, 49, 80, 74, 47, 85, 42, 50, 8, 38, 41, 54, 55]

class EQLv2(nn.Module):
    def __init__(self,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 num_classes=100,  # 1203 for lvis v1.0, 1230 for lvis v0.5
                 gamma=12,
                 mu=0.8,
                 alpha=4.0,
                 vis_grad=False,
                 test_with_obj=True):
        super().__init__()
        self.use_sigmoid = True
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.num_classes = num_classes
        self.group = True

        # cfg for eqlv2
        self.vis_grad = vis_grad
        self.gamma = gamma
        self.mu = mu
        self.alpha = alpha

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(self.num_classes).cuda())
        self.register_buffer('neg_grad', torch.zeros(self.num_classes).cuda())
        # At the beginning of training, we set a high value (eg. 100)
        # for the initial gradient ratio so that the weight for pos gradients and neg gradients are 1.
        self.register_buffer('pos_neg', torch.ones(self.num_classes).cuda() * 100)

        self.test_with_obj = test_with_obj

        def _func(x, gamma, mu):
            return 1 / (1 + torch.exp(-gamma * (x - mu)))
        self.map_func = partial(_func, gamma=self.gamma, mu=self.mu)
        # logger = get_root_logger()
        # logger.info(f"build EQL v2, gamma: {gamma}, mu: {mu}, alpha: {alpha}")

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                wb=None,
                epoch=None,
                **kwargs):
        self.n_i, self.n_c = cls_score.size()

        self.gt_classes = label
        self.pred_class_logits = cls_score

        def expand_label(pred, gt_classes):
            target = pred.new_zeros(self.n_i, self.n_c)
            target[torch.arange(self.n_i), gt_classes] = 1
            return target

        target = label

        pos_w, neg_w = self.get_weight(cls_score)

        weight = pos_w * target + neg_w * (1 - target)

        bce_loss = F.binary_cross_entropy_with_logits(cls_score, target,
                                                      reduction='none')

        pos_mask = (label == 1.0)
        neg_mask = (label == 0.0)
        
        # Calculate positive and negative components
        # pos_loss = bce_loss[pos_mask].mean()
        # neg_loss = bce_loss[neg_mask].mean()
        # Create a mask for ivt_head classes
        ivt_head_mask = torch.zeros_like(bce_loss, dtype=torch.bool)
        for idx in ivt_head:
            ivt_head_mask[:, idx] = True
        ivt_medium_mask = torch.zeros_like(bce_loss, dtype=torch.bool)
        for idx in ivt_medium:  
            ivt_medium_mask[:, idx] = True
        ivt_tail_mask = torch.zeros_like(bce_loss, dtype=torch.bool)
        for idx in ivt_tail:
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
        neg_loss = bce_loss[neg_mask]
        pos_neg_head = self.pos_neg[ivt_head].mean()
        pos_neg_medium = self.pos_neg[ivt_medium].mean()
        pos_neg_tail = self.pos_neg[ivt_tail].mean()
        pos_grad_head = self.pos_grad[ivt_head].mean()
        pos_grad_medium = self.pos_grad[ivt_medium].mean()
        pos_grad_tail = self.pos_grad[ivt_tail].mean()
        neg_grad_head = self.neg_grad[ivt_head].mean()
        neg_grad_medium = self.neg_grad[ivt_medium].mean()
        neg_grad_tail = self.neg_grad[ivt_tail].mean()
        pos_w_head = pos_w[ivt_head_mask].mean()
        pos_w_medium = pos_w[ivt_medium_mask].mean()
        pos_w_tail = pos_w[ivt_tail_mask].mean()
        neg_w_head = neg_w[ivt_head_mask].mean()
        neg_w_medium = neg_w[ivt_medium_mask].mean()
        neg_w_tail = neg_w[ivt_tail_mask].mean()
        if wb is not None:
            wb.log({'pos_grad_head': pos_grad_head.item(),
                    'pos_grad_medium': pos_grad_medium.item(),
                    'pos_grad_tail': pos_grad_tail.item(),
                    'neg_grad_head': neg_grad_head.item(),
                    'neg_grad_medium': neg_grad_medium.item(),
                    'neg_grad_tail': neg_grad_tail.item(),
                    'pos_neg_head': pos_neg_head.item(),
                    'pos_neg_medium': pos_neg_medium.item(),
                    'pos_neg_tail': pos_neg_tail.item(),
                    'pos_w_head': pos_w_head.item(),
                    'pos_w_medium': pos_w_medium.item(),
                    'pos_w_tail': pos_w_tail.item(),
                    'neg_w_head': neg_w_head.item(),
                    'neg_w_medium': neg_w_medium.item(),
                    'neg_w_tail': neg_w_tail.item()},
                    step=epoch)

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

        cls_loss = torch.sum(bce_loss * weight) / (self.n_i * self.n_c)
        #cls_loss = torch.sum(bce_loss) / self.n_i

        self.collect_grad(cls_score.detach(), target.detach(), weight.detach())
        return self.loss_weight * cls_loss

    def get_channel_num(self, num_classes):
        num_channel = num_classes + 1
        return num_channel

    def get_activation(self, cls_score):
        cls_score = torch.sigmoid(cls_score)
        n_i, n_c = cls_score.size()
        bg_score = cls_score[:, -1].view(n_i, 1)
        if self.test_with_obj:
            cls_score[:, :-1] *= (1 - bg_score)
        return cls_score

    def collect_grad(self, cls_score, target, weight):
        prob = torch.sigmoid(cls_score)
        grad = target * (prob - 1) + (1 - target) * prob
        grad = torch.abs(grad)

        # do not collect grad for objectiveness branch [:-1]
        pos_grad = torch.sum(grad * target * weight, dim=0)
        neg_grad = torch.sum(grad * (1 - target) * weight, dim=0)
        # pos_grad = torch.sum(grad * target, dim=0)
        # neg_grad = torch.sum(grad * (1 - target), dim=0)

        if dist.is_initialized():
            dist.all_reduce(pos_grad)
            dist.all_reduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

    def get_weight(self, cls_score):
        neg_w = torch.cat([self.map_func(self.pos_neg), cls_score.new_ones(1)])[:-1]
        pos_w = 1 + self.alpha * (1 - neg_w)
        neg_w = neg_w.view(1, -1).expand(self.n_i, self.n_c)
        pos_w = pos_w.view(1, -1).expand(self.n_i, self.n_c)
        return pos_w, neg_w