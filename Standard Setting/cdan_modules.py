import torch
import torch.nn as nn
import torch.nn.functional as F
import config_cdan as config


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_
        return output, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class ConditionalDomainDiscriminator(nn.Module):
    def __init__(self, feature_dim, num_classes=2, hidden_dim=None, dropout=None):
        super().__init__()
        hidden_dim = hidden_dim or config.CDAN_DISCRIMINATOR_HIDDEN_DIM
        dropout = dropout or config.DROPOUT_RATE
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.conditional_dim = feature_dim * num_classes
        
        self.discriminator = nn.Sequential(
            nn.Linear(self.conditional_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features, predictions):
        pred_probs = F.softmax(predictions, dim=1)
        
        features_expanded = features.unsqueeze(2)
        pred_expanded = pred_probs.unsqueeze(1)
        
        conditional_features = features_expanded * pred_expanded
        conditional_features = conditional_features.view(-1, self.conditional_dim)
        
        domain_pred = self.discriminator(conditional_features)
        return domain_pred
