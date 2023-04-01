import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.nn_utils import MLP


class MultimodalClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.model = MLP(2 * input_dim, hidden_dims, output_dim)


    def forward(self, x, y):
        logits = self.model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none").sum(dim=1).mean()
        return {
            "loss": loss,
            "logits": logits
        }


class UnimodalClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.image_model = MLP(input_dim, hidden_dims, output_dim)
        self.text_model = MLP(input_dim, hidden_dims, output_dim)


    def forward(self, x_image, x_text, y):
        pred_image = torch.sigmoid(self.image_model(x_image))
        pred_text = torch.sigmoid(self.text_model(x_text))
        pred_avg = (pred_image + pred_text) / 2
        loss = F.binary_cross_entropy(pred_avg, y, reduction="none").sum(dim=1).mean()
        return {
            "loss": loss,
            "logits": pred_avg.log()
        }