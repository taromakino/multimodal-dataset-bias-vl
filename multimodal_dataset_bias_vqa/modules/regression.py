import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.nn_utils import MLP


class MultimodalRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.model = MLP(2 * input_dim, hidden_dims, output_dim)


    def forward(self, x, y):
        logits = self.model(x)
        return {
            "loss": F.binary_cross_entropy_with_logits(logits, y),
            "logits": logits
        }


class UnimodalRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.image_model = MLP(input_dim, hidden_dims, output_dim)
        self.text_model = MLP(input_dim, hidden_dims, output_dim)


    def forward(self, x_image, x_text, y):
        pred_image = torch.sigmoid(self.image_model(x_image))
        pred_text = torch.sigmoid(self.text_model(x_text))
        pred_avg = (pred_image + pred_text) / 2
        return {
            "loss": F.binary_cross_entropy(pred_avg, y),
            "logits": pred_avg.log()
        }