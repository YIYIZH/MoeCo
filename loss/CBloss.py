class ClassBalancedBCELoss(nn.Module):
    def __init__(self, beta=0.9999, epsilon=1, max_weight=100.0):
        super(ClassBalancedBCELoss, self).__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.max_weight = max_weight
    def focal_loss(self, labels, logits, alpha, gamma):
        """Compute the focal loss between `logits` and the ground truth `labels`.

        Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
        where pt is the probability of being classified to the true class.
        pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

        Args:
        labels: A float tensor of size [batch, num_classes].
        logits: A float tensor of size [batch, num_classes].
        alpha: A float tensor of size [batch_size]
            specifying per-example weight for balanced cross entropy.
        gamma: A float scalar modulating loss from hard and easy examples.

        Returns:
        focal_loss: A float32 scalar representing normalized total loss.
        """    
        BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
                torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss

        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)

        focal_loss /= torch.sum(labels)
        return focal_loss

    def forward(self, logits, targets, class_counts, gamma=2):
        # Compute effective number of samples per class
        print('using class balanced loss')
        # class_counts = torch.tensor(class_counts[:100], dtype=torch.float32).cuda()
        # effective_num = torch.zeros_like(class_counts)
        # effective_num = (1.0 - torch.pow(self.beta, class_counts)) / (1.0 - self.beta) 
        # effective_num = torch.clamp(effective_num, min=self.epsilon)

        # # Compute weights as inverse of effective number
        # #self.weights = torch.zeros_like(effective_num)
        # self.weights = 1.0 / effective_num
        # self.weights = torch.clamp(self.weights, max=self.max_weight)

        # # Normalize weights
        # if self.weights.sum() > 0:
        #     self.weights = self.weights / self.weights.sum() * len(class_counts)
        # #weights = self.weights.to(logits.device)
        # # Compute BCE loss for each class independently
        # bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # # Apply weights per class
        # weighted_loss = self.weights.unsqueeze(0) * bce_loss  # Broadcasting weights across batch
        # return weighted_loss.mean()
        samples_per_cls = torch.tensor(class_counts[:100], dtype=torch.float32)
        no_of_classes = len(samples_per_cls)
        samples_per_cls = torch.clamp(samples_per_cls, min=self.epsilon)
        effective_num = 1.0 - np.power(self.beta, samples_per_cls)
        weights = (1.0 - self.beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * no_of_classes

        weights = torch.tensor(weights).float().cuda()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(targets.shape[0],1) * targets
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1,no_of_classes)

        ls="focal"
        if ls == "focal":
            cb_loss = self.focal_loss(targets, logits, weights, gamma)
        elif ls == "sigmoid":
            cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = targets, weight=weights)# weight for sampels, pos_weight for classes
        elif ls == "softmax":
            pred = logits.softmax(dim = 1)
            cb_loss = F.binary_cross_entropy(input = pred, target = targets, weight = weights)
        return cb_loss