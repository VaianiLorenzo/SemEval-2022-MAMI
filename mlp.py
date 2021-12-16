import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_value = 0.25, hidden_dim=1536):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_fc = nn.Linear(input_dim, self.hidden_dim)
        #self.hidden_fc_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        #self.hidden_fc_2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_fc = nn.Linear(self.hidden_dim, output_dim)

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        '''
        h_1 = F.relu(self.input_fc(x))
        h_1 = self.dropout(h_1)
        h_2 = F.relu(self.hidden_fc_1(h_1))
        h_2 = self.dropout(h_2)
        h_3 = F.relu(self.hidden_fc_2(h_2))
        '''
        h_1 = F.relu(self.input_fc(x))
        #y_pred = torch.sigmoid(self.output_fc(h_1))

        # no final sigmoid because alredy in the BCE loss function
        y_pred = self.output_fc(h_1)

        return y_pred


'''
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_value = 0.25, n_hidden=3, hidden_dim=1536):
        super().__init__()

        self.n_hidden = n_hidden
        self.hidden_dim = hidden_dim

        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.list_fc_layers = []
        for i in range(0, n_hidden):
            self.list_fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        #batch_size = x.shape[0]
        #x = x.view(batch_size, -1)
        h_1 = F.relu(self.input_fc(x))
        h_1 = self.dropout(h_1)
        
        for i, l in enumerate(self.list_fc_layers):
            h_1 = F.relu(l(h_1))
            if i != len(self.list_fc_layers) - 1:
                h_1 = self.dropout(h_1)

        y_pred = F.sigmoid(self.output_fc(h_1))
        #y_pred = F.relu(self.output_fc(h_5))
        #y_pred = self.output_fc(h_5)
        return y_pred
'''
