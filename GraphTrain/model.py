import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GraphConv, TopKPooling
from torch_geometric.nn import global_mean_pool as g_mean
from torch_geometric.nn import global_max_pool as g_max


class Net(torch.nn.Module):
    def __init__(self, num_features,hidden, dropout, num_classes):
        super(Net, self).__init__()

        self.conv1 = GraphConv(num_features, hidden)
        self.pool1 = TopKPooling(hidden, ratio=0.8)
        
        self.conv2 = GraphConv(hidden, hidden)
        self.pool2 = TopKPooling(hidden, ratio=0.8)

        self.lin1 = nn.Linear(hidden*2, hidden)
        self.lin2 = nn.Linear(hidden, num_classes)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([g_max(x, batch), g_mean(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([g_max(x, batch), g_mean(x, batch)], dim=1)

        x = x1+x2
        x = F.relu(self.lin1(x))
        # print("*"*10)
        # print('dropout value')
        # print(self.dropout)
        # print("*"*10)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.log_softmax(self.lin2(x), dim=-1)

        return x