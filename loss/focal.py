        
class MultiLabelFocalLoss(nn.Module):
    """
    Multi-label focal loss implementation for PyTorch.
    Args:
        alpha (list or None): Weight for each label to handle imbalance, e.g., [0.9, 0.1] for two labels.
        gamma (float): Focusing parameter, typically set to 2 for standard focal loss.
    """
    def __init__(self, alpha=0.5, gamma=2):
        super(MultiLabelFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # Per-label weights for imbalance

    def forward(self, logits, targets, wb=None, epoch=None):
        """
        Compute multi-label focal loss.
        Args:
            logits (torch.Tensor): Model outputs (logits), shape [batch_size, num_classes].
            targets (torch.Tensor): Ground truth labels, shape [batch_size, num_classes], binary (0 or 1).
        Returns:
            torch.Tensor: Mean focal loss over the batch.
        """
        print('using focal loss')
        lbd = 1
        weights = 1.0
        logit = logits * (1 - targets) * lbd  + logits * targets
        weight = (weights / lbd) * (1 - targets) + weights * targets
        bce_loss = F.binary_cross_entropy_with_logits(
            logit, targets, weight, reduction='none'
        )
        
        # Create masks for positive and negative samples
        pos_mask = (targets == 1.0)
        neg_mask = (targets == 0.0)
        
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
        
        prob = torch.sigmoid(logit)
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
        # pos_loss_fg = fairgrad(pos_loss, 0)
        # neg_loss_fg = fairgrad(neg_loss, self.alpha)
        # #neg_loss_fg = fairgrad(neg_loss, (1-self.alpha))
        # wb.log({'pos_loss': pos_loss.mean().item(),
        #     'neg_loss': neg_loss.mean().item(),
        #     'pos_loss_fg': pos_loss_fg.mean().item(),
        #     'neg_loss_fg': neg_loss_fg.mean().item()},
        #     step=epoch)
        # loss = torch.sum(pos_loss_fg) + torch.sum(neg_loss_fg)
        # loss = loss / (pos_loss.size(0) + neg_loss.size(0))
        #loss = (pos_loss.mean() + neg_loss.mean()) / 2.0
        loss = bce_loss.mean()
        #loss = (pos_head_loss.mean() + neg_head_loss.mean()+ pos_medium_loss.mean() + neg_medium_loss.mean() + pos_tail_loss.mean() + neg_tail_loss.mean()) / 6.0 
        #loss = (head_loss + medium_loss + tail_loss)/3
        return loss
    

        # old = False
        # if old:
        #     # Apply sigmoid to get probabilities
        #     inputs = torch.sigmoid(logits)
            
        #     # Set alpha weights if provided, otherwise use uniform weights
        #     # if self.alpha is None:
        #     #     alpha = torch.ones(targets.size(1), device=targets.device)
        #     # else:
        #     #     alpha = torch.as_tensor(self.alpha).to(targets.device)
            
        #     # Expand alpha to match batch dimension for broadcasting
        #     #alpha = alpha.view(1, -1).expand_as(targets)
            
        #     # Compute positive part (for labels present, y=1)
        #     # alpha = train_weights[:100]
        #     # alpha = torch.as_tensor(alpha).to(targets.device)
        #     # total = alpha.sum()
        #     # alpha = 1.0 - (alpha/ total)  # Inverse frequency, adjust as needed
        #     # alpha = alpha / alpha.sum()
        #     # #positive_part = targets * alpha * - (1 - inputs)**self.gamma * torch.log(inputs)
            
        #     positive_part = targets * - (1 - inputs)**self.gamma * torch.log(inputs + 1e-8)
            
        #     # # Compute negative part (for labels absent, y=0)
        #     # #negative_part = (1 - targets) * (1 - alpha) * - inputs**self.gamma * torch.log(1 - inputs)
        #     negative_part = (1 - targets) * - inputs**self.gamma * torch.log(1 - inputs + 1e-8)
            
        #     # # Sum positive and negative parts for total loss per sample
        #     total_loss = positive_part + negative_part
            
        #     # Sum over classes and average over batch
        #     return total_loss.mean()
        #     #return total_loss.sum(dim=1).mean()
        # else:

        #     if not (0 <= self.alpha <= 1) and self.alpha != -1:
        #         raise ValueError(f"Invalid alpha value: {alpha}. alpha must be in the range [0,1] or -1 for ignore.")

        #     p = torch.sigmoid(logits)
        #     ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        #     p_t = p * targets + (1 - p) * (1 - targets)
        #     loss = ce_loss * ((1 - p_t) ** self.gamma)

        #     if alpha >= 0:
        #         alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        #         loss = alpha_t * loss
        #     loss = loss.mean()
        #     return loss