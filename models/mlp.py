import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_value=0.25, hidden_dim=1536):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_fc = nn.Linear(input_dim, self.hidden_dim)
        self.output_fc = nn.Linear(self.hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        h_1 = F.relu(self.input_fc(x))

        # no final sigmoid because alredy in the BCE loss function
        y_pred = self.output_fc(h_1)

        return y_pred
