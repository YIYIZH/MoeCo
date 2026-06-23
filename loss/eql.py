import torch
import torch.nn as nn
import torch.nn.functional as F

class EQL(nn.Module):
    def __init__(self,
                 train_weights,
                 use_sigmoid=True,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 lambda_=0.00177,
                 version="v0_5"):
        super(EQL, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight
        self.lambda_ = lambda_
        self.version = version
        self.train_weights = torch.tensor(train_weights, dtype=torch.float32)[:100]
        self.lambda_ = 999/ self.train_weights.sum() # 999
        self.freq_info = torch.FloatTensor(self.train_weights / self.train_weights.sum()).cuda()
        self.num_class_included = torch.sum(self.freq_info < self.lambda_)
        self.temperature = 1
        print(f"set up EQL (version {version}), {self.num_class_included} classes included.")
        
        # Define class indices for head, medium, and tail classes
        self.ivt_head = [17, 60, 19]
        self.ivt_medium = [58, 7, 20, 12, 94, 61, 96, 82, 59, 57, 29, 79, 16]  # Medium frequency IVT indices
        self.ivt_tail = [78, 69, 1, 18, 68, 95, 99, 63, 14, 27, 88, 4, 22, 92, 36, 28, 62, 98, 21, 30, 51, 10, 13, 52, 64, 37, 23, 97, 44, 6, 66, 34, 90, 33, 87, 39, 76, 71, 84, 93, 40, 0, 53, 26, 3, 32, 45, 24, 9, 31, 25, 73, 35, 81, 11, 75, 15, 48, 83, 77, 43, 2, 91, 86, 89, 5, 72, 46, 56, 67, 70, 65, 49, 80, 74, 47, 85, 42, 50, 8, 38, 41, 54, 55]
    
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
        #weight_tail = self.threshold_func()
        eql_w_neg = 1 - self.exclude_func() * weight_tail * (1 - label)
        eql_w_pos = 1 - weight_nontail * label
        eql_w = eql_w_neg * eql_w_pos
        # eql_w = eql_w_pos
        #eql_w = eql_w_neg

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

        cls_loss = (pos_loss.mean() + neg_loss.mean()) / 2.0 
        #cls_loss = 0.001 * kld_loss + 0.5* pos_loss.mean()+ 0.5 * neg_loss.mean()
        #cls_loss = torch.sum(cls_loss * eql_w) / self.n_i
        #cls_loss = torch.mean(bce_loss * eql_w)
        #cls_loss = bce_loss.mean()
        # print(torch.mean(bce_loss * eql_w))
        #cls_loss = (pos_head_loss.mean() + neg_head_loss.mean()+ pos_medium_loss.mean() + neg_medium_loss.mean() + pos_tail_loss.mean() + neg_tail_loss.mean()) / 6.0 
        #cls_loss = (head_loss + medium_loss + tail_loss)/3
        return cls_loss

    def exclude_func_ours(self):
        # instance-level weight
        tail_class = self.pred_class_logits.new_zeros(self.n_c)
        tail_class[self.freq_info < self.lambda_] = 1
        #bg_ind = self.n_c # background class

        weight1 = (self.gt_classes == tail_class).float()
        weight2 = weight1 * tail_class
        # set the value to 1 if the sum of first dim in weight2 is 0
        weight3 = (weight2.sum(dim=1) == 0).float() #  1 means instance i has no tail class
        weight = weight3.view(self.n_i, 1).expand(self.n_i, self.n_c)
        # find how many 0 in weight
        return weight

    def threshold_func_ours(self):
        # class-level weight
        # mask = torch.rand(100).cuda()
        # mask = torch.where(mask > 0.5, torch.tensor(1, device=mask.device), torch.tensor(0, device=mask.device))
        weight= self.pred_class_logits.new_zeros(self.n_c)
   
        weight[self.freq_info < self.lambda_] = 1
        mask = torch.bernoulli(torch.full((self.n_c,), 0.3)) # randomly select percent of the tail classes to be supressed, will effect the random()
        weight_tail = weight * mask.cuda()
        weight_nontail = (1-weight) * mask.cuda()
        weight_tail = weight_tail.view(1, self.n_c).expand(self.n_i, self.n_c)
        weight_nontail = weight_nontail.view(1, self.n_c).expand(self.n_i, self.n_c)
        return [weight_tail, weight_nontail]

    def exclude_func(self):
        # instance-level weight
        #bg_ind = self.pred_class_logits.new_zeros(self.n_c)
        # Set background index (all zeros in one-hot label) to 1, others to 0
        indx = (self.gt_classes.sum(dim=1) == 0).float()
        #bg_ind = bg_ind.view(-1, 1).expand(-1, self.n_c)
        weight = (indx != 0).float()
        weight = weight.view(self.n_i, 1).expand(self.n_i, self.n_c)
        return weight

    def threshold_func(self):
        # class-level weight
        weight = self.pred_class_logits.new_zeros(self.n_c)
        weight[self.freq_info < self.lambda_] = 1
        mask = torch.bernoulli(torch.full((self.n_c,), 0.9)) # randomly select percent of the tail classes to be supressed, will effect the random()
        weight = weight * mask.cuda()
        weight = weight.view(1, self.n_c).expand(self.n_i, self.n_c)
        return weight